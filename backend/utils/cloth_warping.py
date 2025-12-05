import cv2
import numpy as np
from typing import Tuple, Dict
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import img_as_float, img_as_ubyte


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


def tps_warp_with_keypoints(cloth: np.ndarray, person: np.ndarray, kps: Dict[str, Tuple[float, float]], bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = person.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    dst_rect = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
    src_h, src_w = cloth.shape[:2]
    grid_x = np.linspace(0, src_w - 1, 4)
    grid_y = np.linspace(0, src_h - 1, 4)
    src_pts = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float32)
    dst_pts = np.array([[x1 + (x / (src_w - 1)) * (x2 - x1), y1 + (y / (src_h - 1)) * (y2 - y1)] for y in grid_y for x in grid_x], dtype=np.float32)
    if "11" in kps and "12" in kps:
        l_sh = kps["11"]; r_sh = kps["12"]
        for i in range(dst_pts.shape[0]):
            yy = src_pts[i][1] / (src_h - 1)
            # Slightly curve upper rows towards shoulder line
            if yy < 0.5:
                alpha = (0.5 - yy)
                dst_pts[i][0] = dst_pts[i][0] * (1 - 0.1 * alpha) + (l_sh[0] + r_sh[0]) / 2 * (0.1 * alpha)
    tform = PiecewiseAffineTransform()
    tform.estimate(src_pts, dst_pts)
    cloth_f = img_as_float(cloth)
    warped = warp(cloth_f, tform, output_shape=(h, w))
    warped_u8 = img_as_ubyte(warped)
    mask = (cv2.cvtColor(warped_u8, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    return warped_u8, mask


def feather_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    m = cv2.GaussianBlur(mask.astype(np.float32), (radius * 2 + 1, radius * 2 + 1), radius)
    m = m / (m.max() + 1e-8)
    return m


def color_transfer_lab(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(np.float32)
    m3 = np.dstack([m]*3)
    src_mean = (src_lab * m3).sum(axis=(0,1)) / (m3.sum(axis=(0,1)) + 1e-8)
    src_std = np.sqrt(((src_lab - src_mean) ** 2 * m3).sum(axis=(0,1)) / (m3.sum(axis=(0,1)) + 1e-8))
    dst_mean = dst_lab.mean(axis=(0,1))
    dst_std = dst_lab.std(axis=(0,1)) + 1e-8
    matched = (dst_lab - dst_mean) / dst_std * src_std + src_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)


def inpaint_person_by_mask(person: np.ndarray, clothing_mask: np.ndarray) -> np.ndarray:
    mask = (clothing_mask.astype(np.uint8) * 255)
    person_bgr = cv2.cvtColor(person, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(person_bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
