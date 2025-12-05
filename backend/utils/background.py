import os
from typing import Literal, Optional
from PIL import Image, ImageFilter
import numpy as np
import cv2


def apply_background_option(image_path: str, option: Literal["Keep", "Blur", "Plain"], person_mask: Optional[np.ndarray] = None) -> str:
    img = Image.open(image_path).convert("RGB")
    if option == "Keep":
        img.save(image_path)
        return image_path
    if person_mask is None:
        if option == "Blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        elif option == "Plain":
            bg = Image.new("RGB", img.size, (240, 240, 240))
            img = Image.blend(bg, img, alpha=0.8)
        img.save(image_path)
        return image_path
    arr = np.array(img)
    mask = cv2.resize(person_mask.astype(np.uint8), (arr.shape[1], arr.shape[0]))
    if option == "Blur":
        blurred = cv2.GaussianBlur(arr, (9, 9), 0)
        out = np.where(mask[..., None] == 1, arr, blurred)
        Image.fromarray(out.astype(np.uint8)).save(image_path)
        return image_path
    if option == "Plain":
        plain = np.full_like(arr, 240)
        out = np.where(mask[..., None] == 1, arr, plain)
        Image.fromarray(out.astype(np.uint8)).save(image_path)
        return image_path
    img.save(image_path)
    return image_path
