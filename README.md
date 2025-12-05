# Virtual Try-On System Using AI (Image-to-Dress Fit) – Level 2

## Overview
A web app that lets a user upload a person image and a cloth image, then generates two try-on results: a warping-based version (Model A, VITON-HD style) and a diffusion-inspired version (Model B). The app displays both results side-by-side, accepts ratings, stores feedback, and recommends similar dresses from a local catalog.

## Folder Structure
- `backend/` FastAPI app, models, utils, and static serving
- `frontend/` Simple HTML/CSS/JS web UI
- `training/` Dataset preparation, fine-tuning (Model A), evaluation scripts

## Requirements
- Python `>=3.10`
- Recommended: virtual environment (venv)
- Packages: `fastapi`, `uvicorn`, `python-multipart`, `pillow`, `numpy`
- Optional: `torch` (for training scripts), `mediapipe` (pose-quality enhancements)

## Setup
```
python -m venv .venv
.\.venv\Scripts\activate
pip install fastapi uvicorn python-multipart pillow numpy
# Optional
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mediapipe
```

## Run Backend
```
uvicorn backend.app:app --reload
```
- Static files served under `http://localhost:8000/static/`
- Health check: `GET http://localhost:8000/api/health`

## Run Frontend
Open `http://localhost:8000/` in a browser. The backend serves the Tailwind-based frontend.

## API Endpoints
- `POST /api/tryon` multipart form
  - `person_image`, `cloth_image`
  - `garment_type`: `Upper body` or `Full dress`
  - `background_option`: `Keep`, `Blur`, `Plain`
  - Returns URLs for person, cloth, result_A, result_B, comparison, and pose-quality flag/message
- `POST /api/feedback` JSON
  - `realism_rating`, `fit_rating`, `preferred_model`, `session_id`
- `GET /api/feedback_summary` JSON
  - Average realism/fit and model preference percentages
- `GET /api/recommendations?k=3&cloth_path=/static/outputs/...` JSON
  - Returns 2–3 similar items from `backend/static/catalog/`

## Catalog
Place cloth images in `backend/static/catalog/` (`.png/.jpg/.jpeg`). The recommendation engine uses color histograms to find similar items.

## Training (Level 2)
- `training/prepare_dataset.py`: Organizes a source folder into `train/` and `val/` splits
- `training/train_model_a.py`: Minimal fine-tuning example for a tiny warp net; saves checkpoint
- `training/evaluate_models.py`: Computes PSNR/SSIM between predictions and ground truth

### Examples
```
python training/prepare_dataset.py c:\data\viton_raw c:\data\viton_prepared --val_ratio 0.1
python training/train_model_a.py c:\data\viton_prepared\train backend\weights\model_a.pt --epochs 2
python training/evaluate_models.py c:\outputs\A c:\outputs\B c:\data\viton_prepared\val
```

## Screenshots
Add your screenshots (UI and sample results) to a `screenshots/` folder and reference them here.

## Notes
- Models are lightweight placeholders designed for demo and coursework; integrate real VITON-HD/TryOnDiffusion pipelines as needed.
- Background handling uses simple blur/plain blending for speed; add human segmentation for higher fidelity.
- Pose-quality check includes basic heuristics; integrate MediaPipe/OpenPose for production accuracy.
