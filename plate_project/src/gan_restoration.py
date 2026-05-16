from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils import get_device


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip), dim=1)


class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


@dataclass
class GanRestorer:
    model: GeneratorUNet
    device: torch.device
    input_size: int = 256

    @classmethod
    def from_weights(cls, weights_path: Path, device_preference: str = "auto", input_size: int = 256) -> "GanRestorer":
        model = GeneratorUNet()
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict)
        device = get_device(device_preference)
        model.to(device)
        model.eval()
        return cls(model=model, device=device, input_size=input_size)

    def restore(self, image_bgr: np.ndarray) -> np.ndarray:
        original_h, original_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)[0].detach().cpu()

        output = ((output + 1.0) / 2.0).clamp(0, 1).numpy()
        output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cv2.resize(output, (original_w, original_h), interpolation=cv2.INTER_CUBIC)


def build_restorer_from_config(root: Path, cfg: dict) -> Optional[GanRestorer]:
    gan_cfg = cfg.get("gan", {})
    if not gan_cfg.get("enabled", False):
        return None

    weights_path = Path(gan_cfg.get("generator_weights", ""))
    if not weights_path.is_absolute():
        weights_path = (root / weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"GAN generator weight missing: {weights_path}")

    return GanRestorer.from_weights(
        weights_path=weights_path,
        device_preference=str(gan_cfg.get("device", "auto")),
        input_size=int(gan_cfg.get("input_size", 256)),
    )
