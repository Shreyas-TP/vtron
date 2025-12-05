import os
from typing import Tuple
from PIL import Image


def check_pose_quality(person_path: str) -> Tuple[bool, str]:
    try:
        img = Image.open(person_path).convert("RGB")
        w, h = img.size
        if h < 256 or w < 256:
            return False, "Image too small; try higher resolution"
        aspect = w / h
        if aspect < 0.5 or aspect > 2.0:
            return False, "Unusual aspect ratio; prefer front-facing portrait"
        return True, "Pose likely ok"
    except Exception as e:
        return False, f"Pose check error: {e}"

