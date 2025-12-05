import os
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class SimpleClothDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.files = [f for f in os.listdir(root) if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert("RGB").resize((256, 256))
        arr = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x


class TinyWarpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(data_dir: str, out_ckpt: str, epochs: int = 1, batch_size: int = 4):
    ds = SimpleClothDataset(data_dir)
    if len(ds) == 0:
        os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
        torch.save({"dummy": True}, out_ckpt)
        return
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    net = TinyWarpNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    net.train()
    for ep in range(epochs):
        for x in dl:
            y = x
            pred = net(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(net.state_dict(), out_ckpt)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_ckpt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()
    train(args.data_dir, args.out_ckpt, args.epochs, args.batch_size)
    print("saved:", args.out_ckpt)
