import os
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T


_resnet = None
_tx = T.Compose([T.Resize((224, 224)), T.ToTensor()])


def _init_resnet():
    global _resnet
    if _resnet is None:
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Identity()
        m.eval()
        _resnet = m


def _embed(path: str) -> np.ndarray:
    _init_resnet()
    img = Image.open(path).convert('RGB')
    x = _tx(img).unsqueeze(0)
    with torch.no_grad():
        z = _resnet(x).squeeze(0).numpy()
    z = z / (np.linalg.norm(z) + 1e-8)
    return z


def recommend(catalog_dir: str, cloth_path: Optional[str], k: int = 3) -> List[str]:
    items = []
    for f in os.listdir(catalog_dir):
        fp = os.path.join(catalog_dir, f)
        if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in {'.png','.jpg','.jpeg'}:
            items.append(fp)
    if not items:
        return []
    if cloth_path and os.path.isfile(cloth_path):
        q = _embed(cloth_path)
        scored = []
        for it in items:
            z = _embed(it)
            scored.append((it, float(np.dot(q, z))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p,_ in scored[:k]]
    return items[:k]

