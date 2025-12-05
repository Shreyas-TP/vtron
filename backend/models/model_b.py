import os
from typing import Literal
from PIL import Image
import numpy as np
from ..utils.human_parsing import person_mask as _person_mask
from ..utils.diffusion_tryon import run_conditional_diffusion


_MODEL_B_READY = False


def load_model_b_weights(path: str):
    global _MODEL_B_READY
    _MODEL_B_READY = True


def run_model_b(person_path: str, cloth_path: str, garment_type: Literal["Upper body", "Full dress"], out_dir: str, base_name: str) -> str:
    person = Image.open(person_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGB")
    pmask_arr = _person_mask(person_path)
    pmask_img = Image.fromarray((pmask_arr*255).astype(np.uint8))
    out_img = run_conditional_diffusion(person, cloth, pmask_img, steps=6)
    out_path = os.path.join(out_dir, f"{base_name}.png")
    out_img.save(out_path)
    return out_path
