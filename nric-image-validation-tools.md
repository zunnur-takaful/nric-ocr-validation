# NRIC Image Validation — Tool Stack (Open-Source, Self-Hosted)

Lightweight validation pipeline for checking an uploaded NRIC (MyKad) image before reading its text. Intended for a login/customer-facing flow (not strict KYC) — filters out wrong uploads, blurry photos, and obviously fake/wrong-colour scans.

## Pipeline Overview

1. Detect card shape → check aspect ratio
2. Check image sharpness (reject blurry uploads)
3. Check colour tone (reject grayscale/wrong-colour scans)
4. OCR the text → validate NRIC number format

## Tools

| Stage | Tool | License | Notes |
|---|---|---|---|
| Card detection / aspect ratio | **OpenCV** (`opencv-python`) | Apache 2.0 | Contour detection + bounding box, runs fully offline |
| Blur / quality check | **OpenCV** (`cv2.Laplacian`) | Apache 2.0 | No separate library needed — same OpenCV install |
| Colour tone check | **OpenCV** + **NumPy** | Apache 2.0 / BSD | HSV conversion + average hue check |
| OCR (text extraction) | **EasyOCR** or **PaddleOCR** | Apache 2.0 | Both run locally, no cloud API calls, GPU optional |
| OCR (alternative, lighter) | **Tesseract OCR** (`pytesseract`) | Apache 2.0 | Smaller footprint, slightly lower accuracy on ID fonts |
| Format validation | Python `re` (built-in) | — | Regex check on NRIC pattern `YYMMDD-PB-###G` + date sanity |

All tools above are open-source, free, and run entirely on your own infrastructure — no third-party API calls, no data leaves your server.

## Suggested Python Packages

```
opencv-python
numpy
easyocr        # or paddleocr / pytesseract
```

## Notes

- No model training or calibration required — this uses simple heuristics (aspect ratio, blur variance, hue range), not machine learning.
- Catches: wrong file type, blurry photos, black-and-white/grayscale scans, non-card images.
- Does **not** catch: well-made forgeries, edited images with correct colour/layout. Not intended as fraud-proof KYC — suitable for a low-stakes customer login check only.
