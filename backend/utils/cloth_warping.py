import cv2
import numpy as np
from typing import Tuple


def _detect_and_match(img1: np.ndarray, img2: np.ndarray):
    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


def warp_cloth_to_torso(cloth: np.ndarray, person: np.ndarray, torso_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = person.shape[:2]
    target = cv2.bitwise_and(person, person, mask=torso_mask)
    dm = _detect_and_match(cloth, target)
    if dm is None or len(dm[0]) < 8:
        bbox = cv2.boundingRect(torso_mask.astype(np.uint8))
        x, y, bw, bh = bbox
        warped = cv2.resize(cloth, (bw if bw>0 else w, bh if bh>0 else h))
        mask = np.ones(warped.shape[:2], dtype=np.uint8)
        out = np.zeros_like(person)
        x2 = min(w, x + warped.shape[1]); y2 = min(h, y + warped.shape[0])
        out[y:y2, x:x2] = warped[:y2 - y, :x2 - x]
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[y:y2, x:x2] = 1
        return out, mask_full
    pts1, pts2 = dm
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(cloth, H, (w, h))
    mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    return warped, mask


def blend_on_person(person: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask3 = np.dstack([mask]*3)
    inv = (1 - mask3)
    out = warped * mask3 + person * inv
    return out.astype(np.uint8)

