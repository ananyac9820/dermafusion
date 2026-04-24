"""
fusion_model.py
───────────────
DermaFusion multimodal model:
  Visual stream  : EfficientNet-B4  → 1792-dim feature vector
  Clinical stream: MLP              →   32-dim feature vector
  Fusion head    : concat → FC layers → 7-class output

Also exposes a GradCAM helper that wraps the visual stream.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models

# metadata vector size must match dataset.py
METADATA_DIM  = 19
NUM_CLASSES   = 7
CNN_FEAT_DIM  = 1280   # EfficientNet-B4 penultimate feature size
MLP_FEAT_DIM  = 32


# ── Clinical stream (MLP) ────────────────────────────────────────────────────
class MetadataModel(nn.Module):
    """
    Encodes the 19-dim metadata vector into a 32-dim feature.
    Input: (batch, 19)  →  Output: (batch, 32)
    """
    def __init__(self, input_size=METADATA_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, MLP_FEAT_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ── Fusion model ─────────────────────────────────────────────────────────────
class DermaFusionModel(nn.Module):
   

    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()

        # ── Visual stream ──────────────────────────────────────────────────
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mob     = models.mobilenet_v2(weights=weights)
        self.cnn_features = mob.features
        self.cnn_pool     = nn.AdaptiveAvgPool2d((1, 1))  # output: (B, 1792, 1, 1)

        # ── Clinical stream ────────────────────────────────────────────────
        self.mlp = MetadataModel(input_size=METADATA_DIM)

        # ── Fusion head ────────────────────────────────────────────────────
        fused_dim = CNN_FEAT_DIM + MLP_FEAT_DIM   # 1792 + 32 = 1824
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, image, metadata):
        # Visual stream
        cnn_feat = self.cnn_features(image)          # (B, 1792, H, W)
        cnn_feat = self.cnn_pool(cnn_feat)            # (B, 1792, 1, 1)
        cnn_feat = torch.flatten(cnn_feat, 1)         # (B, 1792)

        # Clinical stream
        mlp_feat = self.mlp(metadata)                 # (B, 32)

        # Fuse + classify
        fused  = torch.cat([cnn_feat, mlp_feat], dim=1)   # (B, 1824)
        logits = self.classifier(fused)                    # (B, 7)
        return logits


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for the visual stream of DermaFusionModel.

    Usage:
        cam     = GradCAM(model)
        heatmap = cam(image_tensor, metadata_tensor)   # numpy (H, W) in [0,1]
        cam.remove_hooks()
    """

    def __init__(self, model: DermaFusionModel):
        self.model    = model
        self._feats   = None
        self._grads   = None

        # Hook onto the last convolutional block of EfficientNet-B4
        target_layer  = model.cnn_features[-1]

        self._fwd_hook = target_layer.register_forward_hook(self._save_feats)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, module, input, output):
        self._feats = output.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, metadata: torch.Tensor):
        """
        Returns a numpy heatmap (H, W) normalised to [0, 1].
        image    : (1, 3, 380, 380)  — single image, on same device as model
        metadata : (1, 19)
        """
        import numpy as np
        import torch.nn.functional as F

        self.model.eval()

        image    = image.requires_grad_(True)
        logits   = self.model(image, metadata)          # (1, 7)
        pred_cls = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, pred_cls].backward()

        # Global average pool of gradients → channel weights
        weights   = self._grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam_map   = (weights * self._feats).sum(dim=1).squeeze() # (H, W)
        cam_map   = F.relu(cam_map)

        # Normalise to [0, 1]
        cam_map  -= cam_map.min()
        if cam_map.max() > 0:
            cam_map /= cam_map.max()

        return cam_map.cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = DermaFusionModel(pretrained=False)
    model.eval()

    dummy_img  = torch.randn(2, 3, 380, 380)
    dummy_meta = torch.randn(2, METADATA_DIM)

    out = model(dummy_img, dummy_meta)
    print(f'Output shape : {out.shape}')   # expect (2, 7)

    cam     = GradCAM(model)
    heatmap = cam(dummy_img[:1], dummy_meta[:1])
    print(f'Heatmap shape: {heatmap.shape}')
    cam.remove_hooks()
    print('Fusion model + Grad-CAM OK')