#!/usr/bin/env python3
"""CLI: validate NRIC (MyKad) image uploads. See nric-image-validation-tools.md."""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytesseract

CARD_ASPECT_RATIO = 85.60 / 54.00  # CR80 ID card standard
ASPECT_TOLERANCE = 0.18
BLUR_VARIANCE_THRESHOLD = 100.0
SATURATION_THRESHOLD = 15.0  # mean HSV saturation below this ~= grayscale
NRIC_RE = re.compile(r"(\d{6})[\s.\-–—]?(\d{2})[\s.\-–—]?(\d{4})")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OCR_MIN_WIDTH = 1200  # upscale crops below this so small combined/cropped uploads stay readable
# ponytail: neither a fixed upscale factor nor the default page-segmentation
# mode reads every real-world card; escalate through a few cheap combinations
# and keep the first that yields a clean NRIC match, else the first result.
OCR_ATTEMPTS = (
    {"target_width": OCR_MIN_WIDTH},
    {"target_width": 1800},
    {"target_width": OCR_MIN_WIDTH, "psm": 6},
    {"target_width": 1800, "sharpen": True},  # last resort: soft/compressed crops with single-char misreads (e.g. "I" -> "!")
)
DUAL_CARD_TOLERANCE = 0.25  # relative tolerance on whole-image ratio vs. 2 cards side-by-side/stacked
# ponytail: no backend watermark detection. Keyword OCR only ever caught
# plain-text stock watermarks anyway (a stylized logo one needs perceptual-
# hash/ML matching, out of scope here) -- reject those earlier instead, as a
# front-end upload rule ("no watermark allowed"), not a best-effort OCR guess.

# ponytail: plain correlation template matching (cv2.matchTemplate), not ML --
# consistent with the project's "simple heuristics" scope. Templates were
# cropped from a flat, axis-aligned gen6 front/back pair (samples/OIP_6.jpg);
# the flag/MyKad/banner art is shared across generations so these also match
# gen7. A rotated/perspective photo (like the gen7 marketing sample) won't
# score well -- that needs feature matching (ORB/homography), not done here.
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAMES = ("flag", "mykad_logo", "banner", "jata_negara")
TEMPLATE_SCALES = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
TEMPLATE_MATCH_THRESHOLD = 0.45

# ponytail: the chip is detected by color+texture, not template correlation --
# its grayscale silhouette false-matched the card's rounded corner, and plain
# gold-hue color alone false-matched warm skin tone in the ID photo. The grid
# of metal contact pads gives it much higher local edge variance than either.
CHIP_HUE_RANGE = ((8, 60, 60), (30, 255, 220))  # HSV gold/bronze
CHIP_AREA_FRACTION = (0.006, 0.06)  # of the card crop's area
CHIP_MAX_ASPECT = 1.6
CHIP_EDGE_VARIANCE_THRESHOLD = 200.0

# ponytail: layouts read off the two reference cards (OIP_6.jpg flat gen6 scan,
# gen7_front.png gen7 marketing render) -- gen6 puts personal text on the left
# with the photo on the right, gen7 mirrors that (photo bottom-left, text
# center-right, small ghost photo bottom-right). Boxes are (x0,y0,x1,y1) as
# fractions of the front-side crop, chosen to dodge both photos.
FRONT_TEXT_ROI = {
    "gen6": (0.00, 0.12, 0.58, 0.85),
    "gen7": (0.16, 0.26, 0.85, 0.92),
}
OLD_GEN_KEYWORDS = ("TOUCH", "MEPS", "CASH", "ATM")


def largest_card_contour(gray):
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _score_vs_threshold(value, threshold):
    """0-100: 100 at/above threshold, scaled down proportionally below it --
    so a near-miss scores close to 100, not a cliff-edge 0."""
    return round(min(100.0, 100.0 * value / threshold), 1)


def check_aspect_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = largest_card_contour(gray)
    if contour is None:
        return {"detected": False, "ratio": None, "expected": round(CARD_ASPECT_RATIO, 3), "pass": False, "score": 0.0}
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if w == 0 or h == 0:
        return {"detected": False, "ratio": None, "expected": round(CARD_ASPECT_RATIO, 3), "pass": False, "score": 0.0}
    ratio = float(max(w, h) / min(w, h))
    passed = abs(ratio - CARD_ASPECT_RATIO) <= ASPECT_TOLERANCE
    score = round(100.0 * min(ratio, CARD_ASPECT_RATIO) / max(ratio, CARD_ASPECT_RATIO), 1)
    return {"detected": True, "ratio": round(ratio, 3), "expected": round(CARD_ASPECT_RATIO, 3), "pass": bool(passed), "score": score}


def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "variance": round(variance, 1), "threshold": BLUR_VARIANCE_THRESHOLD,
        "pass": bool(variance >= BLUR_VARIANCE_THRESHOLD), "score": _score_vs_threshold(variance, BLUR_VARIANCE_THRESHOLD),
    }


def check_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    return {
        "mean_saturation": round(mean_saturation, 1), "threshold": SATURATION_THRESHOLD,
        "pass": bool(mean_saturation >= SATURATION_THRESHOLD), "score": _score_vs_threshold(mean_saturation, SATURATION_THRESHOLD),
    }


_template_cache = {}


def _load_template(name):
    if name not in _template_cache:
        img = cv2.imread(str(TEMPLATES_DIR / f"{name}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"missing template: {name}.png")
        _template_cache[name] = img
    return _template_cache[name]


def match_template_score(bgr, name):
    """Best multi-scale cv2.matchTemplate correlation score of a template in a color image.

    ponytail: matching in color (not grayscale) matters here -- the flag's
    red/white/blue/yellow and the chip's gold are the templates' most
    distinctive signal against the card's blue background; grayscale throws
    that away and both start false-matching the opposite side's artwork.
    """
    template = _load_template(name)
    img_h, img_w = bgr.shape[:2]
    best = 0.0
    for scale in TEMPLATE_SCALES:
        th, tw = int(template.shape[0] * scale), int(template.shape[1] * scale)
        if th < 8 or tw < 8 or th >= img_h or tw >= img_w:
            continue
        resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(bgr, resized, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(result.max()))
    return round(best, 3)


def classify_side(bgr):
    """front: flag/MyKad-logo/banner art; back: the Jata Negara emblem."""
    front_score = max(match_template_score(bgr, n) for n in ("flag", "mykad_logo", "banner"))
    back_score = match_template_score(bgr, "jata_negara")
    if max(front_score, back_score) < TEMPLATE_MATCH_THRESHOLD:
        side = "unknown"
    else:
        side = "front" if front_score >= back_score else "back"
    return {"side": side, "front_score": front_score, "back_score": back_score}


def has_qr_code(bgr):
    found, _ = cv2.QRCodeDetector().detect(bgr)
    return bool(found)


def detect_chip(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *CHIP_HUE_RANGE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_area = bgr.shape[0] * bgr.shape[1]
    lo, hi = CHIP_AREA_FRACTION
    for c in contours:
        area = cv2.contourArea(c)
        if not (lo * card_area <= area <= hi * card_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) / max(1, min(w, h)) > CHIP_MAX_ASPECT:
            continue
        patch = cv2.cvtColor(bgr[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(patch, cv2.CV_64F).var() >= CHIP_EDGE_VARIANCE_THRESHOLD:
            return True
    return False


def classify_generation(side, bgr, text):
    """gen6 had the chip on the front and Touch 'n Go/MEPS/ATM badges on the
    back; gen7 moved the chip to the back and added a QR code there instead.
    Evidence: samples/OIP_6.jpg (gen6 front+back) vs samples/gen7_front.png
    and samples/gen7_back.png (gen7, marketing renders -- best-effort only)."""
    has_chip = detect_chip(bgr)
    if side == "front":
        return "gen6" if has_chip else "gen7"
    if side == "back":
        if has_qr_code(bgr) or has_chip:
            return "gen7"
        if any(kw in text.upper() for kw in OLD_GEN_KEYWORDS):
            return "gen6"
    return "unknown"


def crop_fraction(image, box):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    return image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def ocr_text_region(image):
    """Escalate through the known front-text layouts (gen6, gen7) before
    falling back to OCR-ing the whole side -- dodges the ID photo(s) that
    otherwise pollute the "line after the NRIC" name/address heuristic."""
    candidates = [crop_fraction(image, FRONT_TEXT_ROI["gen6"]), crop_fraction(image, FRONT_TEXT_ROI["gen7"]), image]
    best = None
    for candidate in candidates:
        text, accuracy, fields = ocr_and_parse(candidate)
        if best is None:
            best = (text, accuracy, fields)
        if fields["nric_number"] and fields["nric_date_valid"] and fields["text"]:
            return text, accuracy, fields
    return best


def run_ocr(image, target_width=OCR_MIN_WIDTH, psm=None, sharpen=False):
    """OCR a crop, returning text in top-to-bottom visual order plus mean confidence.

    ponytail: image_to_string's block order isn't reliably top-to-bottom on
    small/sparse crops (e.g. a half-card split), which silently breaks the
    "name is the line after the NRIC" heuristic. Rebuilding lines from
    image_to_data's word boxes sorted by Y avoids that, and reuses the one
    OCR pass already needed for confidence instead of calling tesseract twice.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, target_width / gray.shape[1])
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if sharpen:
        # unsharp mask: only tried as a late OCR_ATTEMPTS fallback since it also
        # amplifies webp/jpeg compression artifacts on already-clean crops.
        gray = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
    config = f"--psm {psm}" if psm else ""
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

    lines = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = lines.setdefault(key, {"top": data["top"][i], "left": data["left"][i], "words": []})
        line["words"].append(word)
    ordered_lines = sorted(lines.values(), key=lambda l: (l["top"], l["left"]))
    text = "\n".join(" ".join(l["words"]) for l in ordered_lines)

    confidences = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
    accuracy = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    return text, accuracy


def ocr_and_parse(image):
    """Run run_ocr/parse_fields across OCR_ATTEMPTS, keeping the highest-confidence
    attempt among those that agree with the first valid NRIC match; falls back to
    the first attempt's result if none match.

    ponytail: OCR confidence is not a correctness signal for digits -- a later
    (especially sharpened) pass can hallucinate a different but still
    regex-valid NRIC with HIGHER reported confidence than the correct one
    (seen on samples/OIP_6.jpg: "860109" misread as "360109"). Anchoring to the
    first match's NRIC and only comparing confidence among attempts that agree
    with it lets a later pass clean up the name/address text without risking
    a silent digit swap.
    """
    first, anchor_nric, candidates = None, None, []
    for attempt in OCR_ATTEMPTS:
        text, accuracy = run_ocr(image, **attempt)
        fields = parse_fields(text)
        if first is None:
            first = (text, accuracy, fields)
        if fields["nric_number"] and fields["nric_date_valid"]:
            anchor_nric = anchor_nric or fields["nric_number"]
            if fields["nric_number"] == anchor_nric:
                candidates.append((text, accuracy, fields))
    if candidates:
        return max(candidates, key=lambda c: c[1])
    return first


def is_dual_card_photo(image):
    """True if this looks like one photo containing both the front and back of
    the card -- rejected rather than split: front-end should ask for front and
    back as two separate uploads (--side front / --side back) instead.

    Pure geometry (whole-image ratio vs. 2 cards side-by-side/stacked) over-
    triggers on plain backgroundy photos and on a single card whose padding
    happens to make the whole image ~half the card ratio (seen on
    samples/OIP_1.webp: a single front card, ratio 0.794 vs. the stacked
    target 0.793). So geometry alone isn't enough -- only flag it when the two
    halves also actually classify as different sides (one front, one back).
    """
    h, w = image.shape[:2]
    ratio = w / h
    side_by_side, stacked = 2 * CARD_ASPECT_RATIO, CARD_ASPECT_RATIO / 2

    if abs(ratio - side_by_side) / side_by_side <= DUAL_CARD_TOLERANCE:
        halves = [image[:, : w // 2], image[:, w // 2 :]]
    elif abs(ratio - stacked) / stacked <= DUAL_CARD_TOLERANCE:
        halves = [image[: h // 2, :], image[h // 2 :, :]]
    else:
        return False

    return {classify_side(half)["side"] for half in halves} == {"front", "back"}


def is_valid_nric_date(yymmdd):
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12):
        return False
    for year in (1900 + yy, 2000 + yy):
        try:
            date(year, mm, dd)
            return True
        except ValueError:
            continue
    return False


def parse_fields(ocr_text):
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    nric_number, nric_line_idx, date_valid = None, None, False
    for i, line in enumerate(lines):
        match = NRIC_RE.search(line.replace(" ", ""))
        if match:
            nric_number = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            nric_line_idx = i
            date_valid = is_valid_nric_date(match.group(1))
            break

    # ponytail: no layout model to tell name apart from address lines, and
    # guessing wrong is worse than not guessing -- hand back every line after
    # the NRIC (minus short OCR-noise lines) and let the caller eyeball it.
    text = None
    if nric_line_idx is not None:
        rest = lines[nric_line_idx + 1:]
        text_lines = [l for l in rest if sum(c.isalpha() for c in l) >= 4]
        text = " / ".join(text_lines) or None

    return {"nric_number": nric_number, "nric_date_valid": date_valid, "text": text}


def validate_image(path, side_override=None):
    image = cv2.imread(str(path))
    if image is None:
        return {"file": str(path), "error": "could not read image"}

    # ponytail: a photo with both sides in one shot can't give a clean ROI/OCR
    # read of either -- reject it and let the front-end ask for front and back
    # as two separate uploads (--side front / --side back) instead of trying
    # to salvage a split (the previous approach, dropped: not worth the risk
    # of a bad split silently losing part of the address).
    if is_dual_card_photo(image):
        return {"file": str(path), "error": "Image appears to contain both sides of the card. Upload front and back as separate images."}

    classification = classify_side(image)
    if side_override:
        classification["side"] = side_override
    # ROI escalation is front-layout-specific; a back (or unclassified) image
    # gets the full image so classify_generation sees its full text (e.g. the
    # Touch 'n Go/MEPS badges), not a front-text-shaped partial.
    if classification["side"] == "front":
        text, ocr_accuracy, fields = ocr_text_region(image)
    else:
        text, ocr_accuracy, fields = ocr_and_parse(image)
    generation = classify_generation(classification["side"], image, text)

    aspect = check_aspect_ratio(image)
    blur = check_blur(image)
    color = check_color(image)

    # ponytail: same signals the old pass/fail gate ANDed, now scored instead
    # of all-or-nothing -- a near-miss (e.g. one soft-fail check) reads very
    # differently from a photo that fails everything, which a boolean hides.
    # No watermark check here -- reject those earlier as a front-end upload
    # rule instead (plain-text OCR keyword matching never caught the stylized
    # logo watermarks that actually matter anyway).
    passed_checks = (aspect["pass"], blur["pass"], color["pass"], fields["nric_number"] is not None, fields["nric_date_valid"])
    score = round(100 * sum(passed_checks) / len(passed_checks), 1)

    return {
        "file": str(path),
        "side": classification["side"],
        "generation": generation,
        "nric_number": fields["nric_number"],
        "nric_date_valid": fields["nric_date_valid"],
        "text": fields["text"],
        "ocr_accuracy": ocr_accuracy,
        "checks": {
            "side_confidence": {
                "front": round(classification["front_score"] * 100, 1),
                "back": round(classification["back_score"] * 100, 1),
            },
            "aspect_ratio": aspect,
            "blur": blur,
            "color": color,
        },
        "score": score,
    }


def collect_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Validate NRIC (MyKad) image uploads.")
    parser.add_argument("path", help="Image file or folder of images")
    parser.add_argument(
        "--side", choices=["front", "back"], default=None,
        help="Tell the validator which side this is instead of guessing (e.g. from a front-end crop/rotate step "
        "that already asks the user). Applies to every image in a folder run, so use a single-file run per side "
        "if you're processing a mixed batch.",
    )
    args = parser.parse_args()

    images = collect_images(args.path)
    if not images:
        print(f"No images found at {args.path}", file=sys.stderr)
        sys.exit(1)

    results = [validate_image(img, side_override=args.side) for img in images]
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
