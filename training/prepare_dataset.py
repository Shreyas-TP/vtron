import os
import shutil
from typing import Tuple


def prepare_viton_like_dataset(source_dir: str, out_dir: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}]
    files.sort()
    n = len(files)
    v = int(n * val_ratio)
    for i, f in enumerate(files):
        src = os.path.join(source_dir, f)
        dst_root = val_dir if i < v else train_dir
        shutil.copy2(src, os.path.join(dst_root, f))
    return train_dir, val_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()
    td, vd = prepare_viton_like_dataset(args.source_dir, args.out_dir, args.val_ratio)
    print("train:", td)
    print("val:", vd)

