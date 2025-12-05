import cv2
import numpy as np
from typing import Dict, Tuple


def estimate_keypoints(image_path: str) -> Dict[str, Tuple[float, float]]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    kps = {}
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=True) as pose:
            res = pose.process(img_rgb)
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                kps[str(i)] = (lm.x * w, lm.y * h)
            return kps
    except Exception:
        pass
    cx, cy = w // 2, h // 2
    kps["11"] = (cx - w * 0.15, cy - h * 0.15)
    kps["12"] = (cx + w * 0.15, cy - h * 0.15)
    kps["23"] = (cx - w * 0.12, cy + h * 0.15)
    kps["24"] = (cx + w * 0.12, cy + h * 0.15)
    return kps


def torso_bbox_from_keypoints(kps: Dict[str, Tuple[float, float]]) -> Tuple[int, int, int, int]:
    ids = [11, 12, 23, 24]  # shoulders and hips
    pts = [kps.get(str(i)) for i in ids if str(i) in kps]
    if not pts:
        return (0, 0, 0, 0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    pad_x = int(0.1 * (x2 - x1 + 1))
    pad_y = int(0.1 * (y2 - y1 + 1))
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
