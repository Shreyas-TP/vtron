import os
from typing import List, Tuple
from PIL import Image
import numpy as np


def _psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    ux = x.mean()
    uy = y.mean()
    vx = x.var()
    vy = y.var()
    cxy = ((x - ux) * (y - uy)).mean()
    c1 = 6.5025
    c2 = 58.5225
    return ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))


def evaluate(dir_pred_a: str, dir_pred_b: str, dir_gt: str) -> Tuple[float, float, float, float]:
    files = [f for f in os.listdir(dir_gt) if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}]
    psnr_a = []
    psnr_b = []
    ssim_a = []
    ssim_b = []
    for f in files:
        gt = np.array(Image.open(os.path.join(dir_gt, f)).convert("RGB"))
        try:
            a = np.array(Image.open(os.path.join(dir_pred_a, f)).convert("RGB").resize(gt.shape[1::-1]))
            b = np.array(Image.open(os.path.join(dir_pred_b, f)).convert("RGB").resize(gt.shape[1::-1]))
        except Exception:
            continue
        psnr_a.append(_psnr(a, gt))
        psnr_b.append(_psnr(b, gt))
        ssim_a.append(_ssim(a, gt))
        ssim_b.append(_ssim(b, gt))
    def avg(x):
        return float(np.mean(x)) if x else 0.0
    return avg(psnr_a), avg(ssim_a), avg(psnr_b), avg(ssim_b)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_pred_a")
    ap.add_argument("dir_pred_b")
    ap.add_argument("dir_gt")
    args = ap.parse_args()
    p_a, s_a, p_b, s_b = evaluate(args.dir_pred_a, args.dir_pred_b, args.dir_gt)
    print("Model A:", {"PSNR": p_a, "SSIM": s_a})
    print("Model B:", {"PSNR": p_b, "SSIM": s_b})

