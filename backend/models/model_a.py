import os
from typing import Literal
from PIL import Image
import numpy as np
import cv2
from ..utils.pose_estimation import estimate_keypoints, torso_bbox_from_keypoints
from ..utils.human_parsing import person_mask as _person_mask, torso_mask
from ..utils.cloth_warping import warp_cloth_to_torso, blend_on_person


_MODEL_A_READY = False


def load_model_a_weights(path: str):
    global _MODEL_A_READY
    _MODEL_A_READY = True


def run_model_a(person_path: str, cloth_path: str, garment_type: Literal["Upper body", "Full dress"], out_dir: str, base_name: str) -> str:
    person = cv2.cvtColor(cv2.imread(person_path), cv2.COLOR_BGR2RGB)
    cloth = cv2.cvtColor(cv2.imread(cloth_path), cv2.COLOR_BGR2RGB)
    kps = estimate_keypoints(person_path)
    bbox = torso_bbox_from_keypoints(kps)
    tmask = torso_mask(person_path, bbox)
    warped, wmask = warp_cloth_to_torso(cloth, person, tmask)
    synth = blend_on_person(person, warped, wmask)
    out_path = os.path.join(out_dir, f"{base_name}.png")
    Image.fromarray(cv2.resize(synth, (512, 512))).save(out_path)
    return out_path
