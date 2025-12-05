import cv2
import numpy as np
from typing import Tuple


def person_mask(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    try:
        import mediapipe as mp
        mp_selfie = mp.solutions.selfie_segmentation
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
            res = seg.process(img_rgb)
        mask = res.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)
        return mask
    except Exception:
        h, w = img.shape[:2]
        rect = (int(w*0.2), int(h*0.1), int(w*0.6), int(h*0.8))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask_gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_gc==cv2.GC_FGD) | (mask_gc==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return mask


def torso_mask(image_path: str, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    mask[y1:y2+1, x1:x2+1] = 1
    return mask
