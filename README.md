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
```

Drop your own MyKad image into `samples/` (the shipped `sample.jpg` is a synthetic demo, not a real card).

`--side front|back` is meant for a front-end that already asks the user before upload — it skips `classify_side`'s template-match guess (the least-tested part of this pipeline). Cropping and rotation need no backend flag: do them client-side before the image reaches this tool, and reject uploads whose aspect ratio is obviously wrong before they're even sent.

A photo containing **both** sides of the card in one shot is rejected outright (`error` field, see below), not split — have the front-end collect front and back as two separate uploads instead.

## Output

JSON per image (or `{"file": ..., "error": "..."}` if the image can't be read or looks like a front+back composite):

| Field | Meaning |
|---|---|
| `side` | `front` / `back` / `unknown`, from template matching against `templates/` (flag, MyKad logo, banner → front; Jata Negara emblem → back), or whatever `--side` was given |
| `generation` | `gen6` / `gen7` / `unknown` — gen6 has the chip on the front and Touch 'n Go/MEPS/ATM badges on the back; gen7 moved the chip to the back and added a QR code there |
| `nric_number` / `nric_date_valid` | parsed `YYMMDD-PB-####` + date sanity check |
| `text` | every text line after the NRIC line in the detected text region, joined with ` / ` — no attempt to split name from address, compare it against the card yourself |
| `ocr_accuracy` | average OCR word confidence (0-100) |
| `checks` | all the image-quality signals, each with its own 0-100 `score` (100 at/past threshold, scaled down below it — a near-miss scores close to 100, not a cliff-edge 0): `side_confidence` (front/back template-match scores), `aspect_ratio` (card contour ratio vs. CR80 standard 1.586), `blur` (Laplacian variance vs. threshold), `color` (mean HSV saturation vs. threshold, catches grayscale scans) |
| `score` | 0-100, the percentage of 5 pass/fail checks that passed (aspect ratio, blur, color, NRIC found, NRIC date valid) — coarser than each `checks` entry's own `score`, this is the single top-line number. No watermark check (front-end's job, see below) |

## How front/back and gen6/gen7 are detected

`templates/` holds 4 small crops (`flag.png`, `mykad_logo.png`, `banner.png`, `jata_negara.png`) cut from a flat, axis-aligned gen6 front+back photo (`samples/OIP_6.jpg`). `classify_side` runs `cv2.matchTemplate` (color, multi-scale) for each and picks whichever side's art scores higher — the flag/logo/banner art is shared across generations, so this also matches gen7. The IC chip isn't template-matched (its grayscale silhouette falsely matched the card's rounded corner, and gold color alone falsely matched skin tone in the ID photo) — it's found instead by a gold-hue color blob with a texture check (`detect_chip`), since the grid of metal contact pads gives it much higher local edge variance than either a plain corner or skin.

Once an image is classified `front`, OCR first tries two known text regions (`FRONT_TEXT_ROI`, read off `OIP_6.jpg` for gen6 and `gen7_front.png` for gen7) before falling back to the whole image — this dodges the ID photo(s) so photo-adjacent noise doesn't get read as part of the name/address. A `back` (or unclassified) image always gets the full image, since those ROIs are front-layout-specific.

The front+back-composite rejection (`is_dual_card_photo`) reuses the same `classify_side` call on each half of the image, geometry-gated: it only fires when the whole image's aspect ratio looks like two cards side-by-side or stacked *and* the two halves actually classify as different sides (one front, one back) — geometry alone over-triggers on a single card whose padding coincidentally halves to the right ratio.

## Front-end responsibilities

This tool does not check for watermarks — show the user a rule like **"No watermark allowed"** before they upload, rather than relying on OCR to catch it after the fact (keyword OCR only ever caught plain stock-photo text like "SAMPLE"/"SHUTTERSTOCK" anyway; a stylized logo watermark isn't read by Tesseract at all and needs perceptual-hash/ML matching to catch, out of scope here). Also front-end's job: crop, rotate, and collect front/back as separate uploads (see Usage above).

## Limitations

Heuristic, not ML: no layout model, so instead of guessing which line is the name vs. the address (and risking a wrong split), `text` is just every text line found after the NRIC number, left for you to read. `aspect_ratio` detection relies on Canny edge contours and is unreliable when the card isn't tightly cropped (background/padding around it) — treat a `pass: false` there as informational, not a hard reject, unless your uploads are pre-cropped. The gen6/gen7 classifier was calibrated against exactly one real flat gen6 card and one tilted gen7 *marketing* render (`samples/gen7_front.png`/`gen7_back.png`, not a real upload) — treat `generation` as a best-effort hint, not a verified fact, until tested against more real gen7 photos. Template matching is also not rotation/perspective-invariant, so a tilted photo may not match either side confidently (`side: unknown`). Not fraud-proof — catches bad uploads (wrong file, blurry, grayscale), not well-made forgeries or watermarks. See `nric-image-validation-tools.md`.
