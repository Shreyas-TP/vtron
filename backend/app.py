import os
import io
import uuid
from datetime import datetime
from typing import Optional, List, Literal

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .utils.preprocess import validate_and_save_image, normalize_paths
from .utils.pose_quality import check_pose_quality
from .utils.background import apply_background_option
from .utils.recommend import recommend
from .utils.human_parsing import person_mask as _person_mask
from .utils.db import FeedbackDB
from .models.model_a import load_model_a_weights, run_model_a
from .models.model_b import load_model_b_weights, run_model_b


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
CATALOG_DIR = os.path.join(STATIC_DIR, "catalog")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CATALOG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

db = FeedbackDB(os.path.join(os.path.dirname(__file__), "feedback.sqlite"))


class TryOnResponse(BaseModel):
    person_url: str
    cloth_url: str
    result_A_url: str
    result_B_url: str
    comparison_url: str
    pose_quality_ok: bool
    pose_message: str


class FeedbackRequest(BaseModel):
    realism_rating: int
    fit_rating: int
    preferred_model: Literal["A", "B", "Both", "None"]
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class FeedbackSummary(BaseModel):
    avg_realism: float
    avg_fit: float
    pref_A_percent: float
    pref_B_percent: float
    pref_Both_percent: float
    pref_None_percent: float


@app.post("/api/tryon", response_model=TryOnResponse)
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    garment_type: Literal["Upper body", "Full dress"] = Form("Upper body"),
    background_option: Literal["Keep", "Blur", "Plain"] = Form("Keep"),
):
    try:
        session_id = str(uuid.uuid4())
        person_path = validate_and_save_image(person_image, OUTPUTS_DIR, f"{session_id}_person")
        cloth_path = validate_and_save_image(cloth_image, OUTPUTS_DIR, f"{session_id}_cloth")

        pose_ok, pose_msg = check_pose_quality(person_path)

        load_model_a_weights(os.path.join(WEIGHTS_DIR, "model_a.pt"))
        load_model_b_weights(os.path.join(WEIGHTS_DIR, "model_b.pt"))

        result_a_path = run_model_a(person_path, cloth_path, garment_type, OUTPUTS_DIR, f"{session_id}_A")
        result_b_path = run_model_b(person_path, cloth_path, garment_type, OUTPUTS_DIR, f"{session_id}_B")

        p_mask = _person_mask(person_path)
        final_a = apply_background_option(result_a_path, background_option, p_mask)
        final_b = apply_background_option(result_b_path, background_option, p_mask)
        comparison_path = os.path.join(OUTPUTS_DIR, f"{session_id}_compare.png")
        try:
            a_img = Image.open(final_a).convert("RGB")
            b_img = Image.open(final_b).convert("RGB")
            w = max(a_img.width, b_img.width)
            h = max(a_img.height, b_img.height)
            a_img = a_img.resize((w, h))
            b_img = b_img.resize((w, h))
            comp = Image.new("RGB", (w*2, h))
            comp.paste(a_img, (0,0))
            comp.paste(b_img, (w,0))
            comp.save(comparison_path)
        except Exception:
            comparison_path = final_a

        person_url, cloth_url, result_A_url, result_B_url, compare_url = normalize_paths(
            STATIC_DIR, [person_path, cloth_path, final_a, final_b, comparison_path]
        )

        return TryOnResponse(
            person_url=person_url,
            cloth_url=cloth_url,
            result_A_url=result_A_url,
            result_B_url=result_B_url,
            comparison_url=compare_url,
            pose_quality_ok=pose_ok,
            pose_message=pose_msg,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    try:
        ts = req.timestamp or datetime.utcnow().isoformat()
        db.insert_feedback(ts, req.session_id or "unknown", req.realism_rating, req.fit_rating, req.preferred_model)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/feedback_summary", response_model=FeedbackSummary)
async def feedback_summary():
    stats = db.summary()
    return FeedbackSummary(**stats)


@app.get("/api/recommendations")
async def recommendations(cloth_path: Optional[str] = None, k: int = 3):
    try:
        fs_path = None
        if cloth_path and cloth_path.startswith("/static"):
            fs_path = cloth_path.replace("/static", STATIC_DIR).replace("/", os.sep)
        else:
            fs_path = cloth_path
        recs = recommend(CATALOG_DIR, fs_path, k)
        urls = [p.replace(STATIC_DIR, "/static").replace("\\", "/") for p in recs]
        return {"items": urls}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root_page():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"error": "frontend/index.html not found"})

