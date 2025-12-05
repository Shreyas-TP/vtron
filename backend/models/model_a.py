import os
from typing import Literal
from PIL import Image
from datetime import datetime
import numpy as np
import cv2
from ..utils.pose_estimation import estimate_keypoints, torso_bbox_from_keypoints
from ..utils.human_parsing import person_mask as _person_mask, torso_mask, clothing_mask
from ..utils.cloth_warping import (
    warp_cloth_to_torso,
    blend_on_person,
    tps_warp_with_keypoints,
    feather_mask,
    color_transfer_lab,
    inpaint_person_by_mask,
)


_MODEL_A_READY = False


def load_model_a_weights(path: str):
    global _MODEL_A_READY
    _MODEL_A_READY = True


def run_model_a(person_path: str, cloth_path: str, garment_type: Literal["Upper body", "Full dress"], out_dir: str, base_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    person = cv2.cvtColor(cv2.imread(person_path), cv2.COLOR_BGR2RGB)
    cloth = cv2.cvtColor(cv2.imread(cloth_path), cv2.COLOR_BGR2RGB)
    kps = estimate_keypoints(person_path)
    bbox = torso_bbox_from_keypoints(kps)
    tmask = torso_mask(person_path, bbox)
    c_mask = clothing_mask(person_path, bbox)
    person_clean = inpaint_person_by_mask(person, c_mask)
    warped, wmask = tps_warp_with_keypoints(cloth, person, kps, bbox)
    wmask_f = feather_mask(wmask, radius=7)
    warped_ct = color_transfer_lab(warped, person_clean, wmask_f)
    synth_base = (warped_ct * wmask_f[..., None] + person_clean * (1 - wmask_f[..., None])).astype(np.uint8)
    synth = synth_base
    out_path = os.path.join(out_dir, f"{base_name}.png")
    # Debug writes
    Image.fromarray(cv2.resize(person, (512, 512))).save(os.path.join(out_dir, f"{ts}_person.png"))
    Image.fromarray(cv2.resize(cloth, (512, 512))).save(os.path.join(out_dir, f"{ts}_cloth.png"))
    Image.fromarray(cv2.resize(warped, (512, 512))).save(os.path.join(out_dir, f"{ts}_warped_cloth.png"))
    Image.fromarray((cv2.resize(wmask, (512, 512)) * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{ts}_warped_mask.png"))
    Image.fromarray(cv2.resize(synth, (512, 512))).save(os.path.join(out_dir, f"{ts}_model_a.png"))
    # Primary output
    Image.fromarray(cv2.resize(synth, (512, 512))).save(out_path)
    return out_path
