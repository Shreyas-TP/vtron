import os
from typing import Literal
from PIL import Image
from datetime import datetime
import numpy as np
from ..utils.human_parsing import person_mask as _person_mask, clothing_mask
from ..utils.pose_estimation import estimate_keypoints, torso_bbox_from_keypoints, pose_heatmap
from ..utils.cloth_warping import tps_warp_with_keypoints, feather_mask, color_transfer_lab, inpaint_person_by_mask
from ..utils.diffusion_tryon import run_conditional_diffusion


_MODEL_B_READY = False


def load_model_b_weights(path: str):
    global _MODEL_B_READY
    _MODEL_B_READY = True


def run_model_b(person_path: str, cloth_path: str, garment_type: Literal["Upper body", "Full dress"], out_dir: str, base_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    person_np = np.array(Image.open(person_path).convert("RGB"))
    cloth_np = np.array(Image.open(cloth_path).convert("RGB"))
    kps = estimate_keypoints(person_path)
    bbox = torso_bbox_from_keypoints(kps)
    c_mask = clothing_mask(person_path, bbox)
    person_clean = inpaint_person_by_mask(person_np, c_mask)
    warped, wmask = tps_warp_with_keypoints(cloth_np, person_np, kps, bbox)
    wmask_f = feather_mask(wmask, radius=7)
    warped_ct = color_transfer_lab(warped, person_clean, wmask_f)
    heat = pose_heatmap(person_path, kps)
    out_img = run_conditional_diffusion(Image.fromarray(person_clean), Image.fromarray(warped_ct), Image.fromarray((wmask_f*255).astype(np.uint8)), steps=3)
    out_path = os.path.join(out_dir, f"{base_name}.png")
    # Debug writes
    Image.fromarray(person_np).resize((512, 512)).save(os.path.join(out_dir, f"{ts}_person.png"))
    Image.fromarray(cloth_np).resize((512, 512)).save(os.path.join(out_dir, f"{ts}_cloth.png"))
    Image.fromarray(warped_ct).resize((512, 512)).save(os.path.join(out_dir, f"{ts}_warped_cloth.png"))
    Image.fromarray((wmask*255).astype(np.uint8)).resize((512, 512)).save(os.path.join(out_dir, f"{ts}_warped_mask.png"))
    out_img.resize((512, 512)).save(os.path.join(out_dir, f"{ts}_model_b.png"))
    # Comparison if A exists
    try:
        a_name_guess = base_name.replace("_B", "_A")
        a_path = os.path.join(out_dir, f"{a_name_guess}.png")
        if os.path.isfile(a_path):
            a_img = Image.open(a_path).convert("RGB").resize((512,512))
            b_img = out_img.resize((512,512))
            comp = Image.new("RGB", (1024, 512))
            comp.paste(a_img, (0,0))
            comp.paste(b_img, (512,0))
            comp.save(os.path.join(out_dir, f"{ts}_comparison.png"))
    except Exception:
        pass
    # Primary output
    out_img.save(out_path)
    return out_path
