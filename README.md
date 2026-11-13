# NRIC Image Validation

Validates an uploaded NRIC (MyKad) image before trusting its text: card shape, blur, color tone, then OCR + format check. See `nric-image-validation-tools.md` for the tool rationale.

## Setup

Requires Docker (handles `tesseract-ocr` + Python deps for you).

```
docker build -t nric-validate .
```

The Dockerfile trusts a local FortiGate TLS-inspection CA cert to reach PyPI/apt in this sandbox — drop that block if building elsewhere.

## Usage

```
./run.sh samples/sample.jpg              # single file, auto-detect side
./run.sh samples/                        # every image in the folder
./run.sh samples/sample.jpg --side front # trust a front-end-supplied side instead of guessing
./run.sh samples/sample.jpg --expected-name "Ahmad Bin Ali" --expected-ic-no "900115145523" # match against what the user claims
```

Drop your own MyKad image into `samples/` (the shipped `sample.jpg` is a synthetic demo, not a real card).

`--side front|back` is meant for a front-end that already asks the user before upload — it skips `classify_side`'s template-match guess (the least-tested part of this pipeline). Cropping and rotation need no backend flag: do them client-side before the image reaches this tool, and reject uploads whose aspect ratio is obviously wrong before they're even sent.

A photo containing **both** sides of the card in one shot is rejected outright (`error` field, see below), not split — have the front-end collect front and back as two separate uploads instead.

`--expected-name`/`--expected-ic-no` are for checking the OCR'd card against what the uploader *claims* (e.g. a name/IC already on an application form) — separate from `score`, which only measures image quality. Give either or both; the result gets a `match` key only when at least one was given (see Output below).

## Output

JSON per image (or `{"file": ..., "error": "..."}` if the image can't be read or looks like a front+back composite):

| Field | Meaning |
|---|---|
| `side` | `front` / `back` / `unknown`, from template matching against `templates/` (flag, MyKad logo, banner → front; Jata Negara emblem → back), or whatever `--side` was given |
| `generation` | `gen6` / `gen7` / `unknown` — gen6 has the chip on the front and Touch 'n Go/MEPS/ATM badges on the back; gen7 moved the chip to the back and added a QR code there |
| `nric_number` / `nric_date_valid` | parsed `YYMMDD-PB-####` + date sanity check |
| `text` | every text line after the NRIC line in the detected text region, joined with ` / ` — no attempt to split name from address, compare it against the card yourself |
| `raw_text` | everything OCR found anywhere on the whole image, unfiltered — no watermark keyword matching (a fixed list can't cover arbitrary custom text like "FOR TAKAFUL USAGE ONLY"), eyeball this for anything that shouldn't be there instead |
| `ocr_accuracy` | average OCR word confidence (0-100) |
| `checks` | all the image-quality signals, each with its own 0-100 `score` — not a cliff-edge pass/fail, so a near-miss reads differently from a wildly-off photo: `side_confidence` (front/back template-match scores), `aspect_ratio` (whole-image ratio vs. CR80 standard 1.586, assumes the front-end already cropped to just the card — 100 at a perfect match, 50 exactly at the pass/fail tolerance boundary, dropping well below 50 the further off it gets), `blur` (Laplacian variance vs. threshold — 100 at/past threshold, scaled down below it), `color` (mean HSV saturation vs. threshold, catches grayscale scans — same shape as `blur`) |
| `score` | 0-100, the average of `aspect_ratio`/`blur`/`color`'s own `score` plus NRIC found and NRIC date valid (each 100 or 0, no continuous measure for those two) — the single top-line number. **Gated**: if `aspect_ratio.pass` is false, `score` is capped at `aspect_ratio`'s own score, since a badly-cropped image shouldn't be maskable by four unrelated perfect scores. No watermark check (front-end's job, see below) |
| `match` | only present if `--expected-name`/`--expected-ic-no` were given: `name` (`best_match_line` — whichever line of `text` matched best — and a 0-100 `score`) and/or `ic_no` (`actual`, a 0-100 `score`, and a boolean `match` after stripping separators from both sides) |

## How front/back and gen6/gen7 are detected

`templates/` holds 4 small crops (`flag.png`, `mykad_logo.png`, `banner.png`, `jata_negara.png`) cut from a flat, axis-aligned gen6 front+back photo (`samples/OIP_6.jpg`). `classify_side` runs `cv2.matchTemplate` (color, multi-scale) for each and picks whichever side's art scores higher — the flag/logo/banner art is shared across generations, so this also matches gen7. The IC chip isn't template-matched (its grayscale silhouette falsely matched the card's rounded corner, and gold color alone falsely matched skin tone in the ID photo) — it's found instead by a gold-hue color blob with a texture check (`detect_chip`), since the grid of metal contact pads gives it much higher local edge variance than either a plain corner or skin.

Once an image is classified `front`, OCR first tries two known text regions (`FRONT_TEXT_ROI`, read off `OIP_6.jpg` for gen6 and `gen7_front.png` for gen7) before falling back to the whole image — this dodges the ID photo(s) so photo-adjacent noise doesn't get read as part of the name/address. A `back` (or unclassified) image always gets the full image, since those ROIs are front-layout-specific.

The front+back-composite rejection (`is_dual_card_photo`) reuses the same `classify_side` call on each half of the image, geometry-gated: it only fires when the whole image's aspect ratio looks like two cards side-by-side or stacked *and* the two halves actually classify as different sides (one front, one back) — geometry alone over-triggers on a single card whose padding coincidentally halves to the right ratio.

## Front-end responsibilities

This tool only validates an image that's already been prepared right — none of this is done server-side, so the front-end must handle it before the image ever reaches this tool:

| Step | What to do | Message to show the user |
|---|---|---|
| Crop | Let the user crop to just the card, no background | "Crop the photo to just the IC card" |
| Rotate | Let the user straighten the card so it's upright | "Rotate the card so it's upright" |
| One side per upload | Capture front and back as two separate photos; pass the matching `--side front` / `--side back` | "Upload the front of your IC" → then "Now upload the back of your IC" |
| Reject an obviously bad crop before sending | Check the cropped image's width/height ratio against ~1.586 (CR80 standard) client-side; don't even upload if it's way off | "This doesn't look like a cropped IC — please try again" |
| No watermark | Refuse sample/demo/stock-photo watermarked images before upload — this tool can't reliably catch one (a fixed keyword list can never cover arbitrary custom text, and a stylized logo watermark isn't OCR-readable at all), so it has to be a front-end rule, not a backend check. `raw_text` is there to help a human reviewer spot one after the fact, not to replace this rule | "No watermark allowed — upload your own original IC photo" |
| Handle a rejected combined photo | If the user submits one photo with both sides in it, this tool returns `{"error": "..."}` instead of a result — catch that and ask them to split it | "Please upload the front and back as two separate photos" |

After getting a result back, two more things worth doing client-side, since this tool is heuristic, not a human:
- Use `score` and `checks` to drive a simple "looks good" vs. "retake photo" prompt (a low `blur`/`color`/`aspect_ratio` sub-score points at *why* to retake).
- Show `nric_number` and `text` back to the user to confirm or correct before trusting them — a digit misread (e.g. `860109` read as `360109`) still looks like a perfectly valid NRIC to this tool's checks, so only the person looking at their own card can catch that.

## Limitations

Heuristic, not ML: no layout model, so instead of guessing which line is the name vs. the address (and risking a wrong split), `text` is just every text line found after the NRIC number, left for you to read. `aspect_ratio` is just the whole image's width/height ratio, so it's only meaningful if the front-end actually cropped to the card first (see Front-end responsibilities) — on an uncropped photo it's measuring the photo's shape, not the card's. The gen6/gen7 classifier was calibrated against exactly one real flat gen6 card and one tilted gen7 *marketing* render (`samples/gen7_front.png`/`gen7_back.png`, not a real upload) — treat `generation` as a best-effort hint, not a verified fact, until tested against more real gen7 photos. Template matching is also not rotation/perspective-invariant, so a tilted photo may not match either side confidently (`side: unknown`). Not fraud-proof — catches bad uploads (wrong file, blurry, grayscale), not well-made forgeries or watermarks. See `nric-image-validation-tools.md`.
