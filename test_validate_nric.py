#!/usr/bin/env python3
"""Self-check for validate_nric.py heuristics. Needs samples/OIP_6.jpg for the
template-matching checks (synthetic images can't stand in for real card art)."""
import cv2
import numpy as np

from validate_nric import (
    check_blur,
    check_color,
    classify_generation,
    classify_side,
    is_valid_nric_date,
    match_expected_ic_no,
    match_expected_name,
    parse_fields,
    validate_image,
)

sharp = np.zeros((200, 200, 3), dtype=np.uint8)
sharp[::2, :] = 255  # stripes -> high Laplacian variance
assert check_blur(sharp)["pass"] is True

blurry = np.full((200, 200, 3), 128, dtype=np.uint8)
assert check_blur(blurry)["pass"] is False

colorful = np.zeros((50, 50, 3), dtype=np.uint8)
colorful[:, :, 2] = 255  # pure red -> high saturation
assert check_color(colorful)["pass"] is True

gray = np.full((50, 50, 3), 128, dtype=np.uint8)
assert check_color(gray)["pass"] is False

assert is_valid_nric_date("900115") is True
assert is_valid_nric_date("901332") is False

fields = parse_fields("WARGANEGARA\n900115-14-5523\nAHMAD BIN ALI\nNO 12 JALAN BUNGA\n50000 KUALA LUMPUR")
assert fields["nric_number"] == "900115-14-5523"
assert fields["nric_date_valid"] is True
assert "AHMAD BIN ALI" in fields["text"]
assert "JALAN BUNGA" in fields["text"]

gen6 = cv2.imread("samples/OIP_6.jpg")
h, w = gen6.shape[:2]
front_bgr = gen6[:, : w // 2]
back_bgr = gen6[:, w // 2 :]

front_cls = classify_side(front_bgr)
assert front_cls["side"] == "front", front_cls
back_cls = classify_side(back_bgr)
assert back_cls["side"] == "back", back_cls

assert classify_generation("front", front_bgr, "") == "gen6"  # chip on front
assert classify_generation("back", back_bgr, "TOUCH N GO ATM 64K chip MEPS CASH") == "gen6"

# --side override must win over the auto-classifier, even on an actual front
# crop forced to "back".
result = validate_image("samples/OIP_1.webp", side_override="back")
assert result["side"] == "back", result

# a genuine front+back-in-one-photo upload must be rejected, not split.
result = validate_image("samples/OIP_6.jpg")
assert "error" in result, result

assert match_expected_ic_no("900115-14-5523", "900115145523")["match"] is True  # separators shouldn't matter
mismatch = match_expected_ic_no("900115-14-5523", "900115-14-5529")
assert mismatch["match"] is False
assert mismatch["score"] > 80  # still close, just one digit off
assert mismatch["differences"] == [{"expected": "9", "actual": "3"}], mismatch  # names the exact digit that differs

name = match_expected_name("AHMAD BIN ALI / NO 12 JALAN BUNGA", "Ahmad Bin Ali")
assert name["best_match_line"] == "AHMAD BIN ALI", name
assert name["score"] > 95, name
assert name["match"] is True, name
assert name["differences"] == [], name  # exact match once normalized -> nothing to report

# punctuation (OCR noise or a legitimate symbol like "'" / "A/L") must not cost match score.
noisy = match_expected_name("NURINA SAKINAH BINT! ANUAR / KEPONG", "Nurina Sakinah Binti Anuar")
assert noisy["match"] is True, noisy
slashed = match_expected_name("AHMAD A/L BAKAR / NO 1 JALAN", "Ahmad A/L Bakar")
assert slashed["best_match_line"] == "AHMAD A/L BAKAR", slashed  # split on " / ", not bare "/"
assert slashed["match"] is True, slashed

# a genuinely different name should be flagged, with the differing part surfaced.
wrong = match_expected_name("JOHN WICK / KAMPUNG MASJID", "John Smith")
assert wrong["match"] is False, wrong
assert wrong["differences"], wrong  # non-empty -- something concrete to point at, not just a low score

# end to end: validate_image should surface the same thing under "match".
result = validate_image("samples/sample.jpg", expected_name="Ahmad Bin Ali", expected_ic_no="900115145523")
assert result["match"]["ic_no"]["match"] is True, result["match"]
assert result["match"]["name"]["score"] > 95, result["match"]

print("OK")
