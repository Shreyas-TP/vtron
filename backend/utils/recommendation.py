import os
from typing import List, Optional
from PIL import Image
import numpy as np


def _color_histogram(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.array(img)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=32, range=(0, 255), density=True)
        hist.append(h)
    return np.concatenate(hist)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def get_recommendations(catalog_dir: str, cloth_path: Optional[str], k: int = 3) -> List[str]:
    items = []
    for fname in os.listdir(catalog_dir):
        fp = os.path.join(catalog_dir, fname)
        if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in {".png", ".jpg", ".jpeg"}:
            items.append(fp)
    if not items:
        return []
    if cloth_path and os.path.isfile(cloth_path):
        q = _color_histogram(cloth_path)
        scored = []
        for it in items:
            h = _color_histogram(it)
            scored.append((it, _similarity(q, h)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:k]]
    else:
        return items[:k]

