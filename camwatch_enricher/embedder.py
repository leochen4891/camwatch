"""DINOv2 ViT-S/14 wrapper: image path -> L2-normalized float32 embedding."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = _pick_device(device)
        log.info("loading %s on %s", model_name, self.device)
        # facebookresearch/dinov2 publishes the small/base/large ViTs via torch.hub.
        # First call downloads weights to ~/.cache/torch/hub; subsequent calls are
        # offline.
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, trust_repo=True
        )
        self.model.eval()
        self.model.to(self.device)
        # DINOv2 expects 224x224 ImageNet-normalized RGB. We resize the short side
        # to 256 then center-crop to 224 so the aspect ratio doesn't get squashed.
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            out = self.model(dummy)
        self.embed_dim = int(out.shape[-1])
        log.info("%s ready (dim=%d)", model_name, self.embed_dim)

    @torch.no_grad()
    def encode_path(self, image_path: Path | str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        feat = self.model(x)[0]
        feat = feat / (feat.norm(p=2) + 1e-12)
        return feat.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_paths(self, image_paths: list[Path | str], batch_size: int = 16) -> np.ndarray:
        out = np.zeros((len(image_paths), self.embed_dim), dtype=np.float32)
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            tensors = [self.tf(Image.open(p).convert("RGB")) for p in chunk]
            x = torch.stack(tensors).to(self.device)
            feats = self.model(x)
            feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-12)
            out[i : i + len(chunk)] = feats.detach().cpu().numpy().astype(np.float32)
        return out
