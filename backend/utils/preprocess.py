import os
from typing import List, Tuple
import io
from fastapi import UploadFile
from PIL import Image


def validate_and_save_image(upload: UploadFile, out_dir: str, base_name: str) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower() or ".png"
    if ext not in [".png", ".jpg", ".jpeg"]:
        ext = ".png"
    path = os.path.join(out_dir, f"{base_name}{ext}")
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(path)
    return path


def normalize_paths(static_dir: str, paths: List[str]) -> Tuple[str, ...]:
    urls = []
    for p in paths:
        url = p.replace(static_dir, "/static").replace("\\", "/")
        urls.append(url)
    return tuple(urls)
