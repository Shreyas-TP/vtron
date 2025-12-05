import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class UNetSmall(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(), nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(), nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(), nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bott(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        y = torch.sigmoid(self.out(d1))
        return y


_net: UNetSmall | None = None


def load_diffusion_model():
    global _net
    if _net is None:
        _net = UNetSmall()


def tensor_from_image(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def run_conditional_diffusion(person_img: Image.Image, cloth_img: Image.Image, mask_img: Image.Image, steps: int = 3) -> Image.Image:
    load_diffusion_model()
    p = tensor_from_image(person_img.resize((512, 512)))  # [1,3,H,W]
    c = tensor_from_image(cloth_img.resize((512, 512)))   # [1,3,H,W]
    m_arr = (np.array(mask_img.resize((512, 512)).convert('L')) > 0).astype(np.float32)
    m = torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    cm = c * m  # mask cloth to preserve 3 channels
    x_in = torch.cat([p, cm], dim=1)  # [1,6,H,W]
    assert x_in.shape[1] == 6, f"Diffusion input channels expected 6, got {x_in.shape[1]}"
    x = x_in.clone()
    for _ in range(steps):
        y = _net(x)  # [1,3,H,W] refined cloth
        weight3 = (m.repeat(1,3,1,1) * 0.5 + 0.5)
        base = x[:, 0:3]
        cloth = x[:, 3:6]
        cloth = (cloth * (1 - 0.3*weight3) + y * (0.3*weight3)).clamp(0,1)
        x = torch.cat([base, cloth], dim=1)
    base = x[:,0:3]
    cloth = x[:,3:6]
    m3 = m.repeat(1,3,1,1)
    comp = (base*(1-m3) + cloth*m3).clamp(0,1)
    out = (comp.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(out)
