"""
TransNet FLOPs Calculator (CR=1/4 and CR=1/16)
- Encoder FLOPs: TransformerEncoder (2 layers) + fc_encoder
- Total FLOPs:   Encoder + fc_decoder + TransformerDecoder (2 layers) + predict
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from thop import profile, clever_format

# ── Minimal logger stub so TransNet.py doesn't crash ──────────────────────────
import types
fake_utils = types.ModuleType("utils")
class _Logger:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
fake_utils.logger = _Logger()
sys.modules["utils"] = fake_utils

from models.TransNet import Transformer

# ── Encoder-only wrapper ───────────────────────────────────────────────────────
class TransNetEncoder(nn.Module):
    """Runs only the UE-side encoder portion."""
    def __init__(self, base: Transformer):
        super().__init__()
        self.feature_shape = base.feature_shape
        self.encoder    = base.encoder
        self.fc_encoder = base.fc_encoder

    def forward(self, x):
        # x: (B, 2, 32, 32)
        feat = x.view(-1, self.feature_shape[0], self.feature_shape[1])  # (B,32,64)
        mem  = self.encoder(feat)                                          # (B,32,64)
        z    = self.fc_encoder(mem.view(mem.shape[0], -1))                # (B, latent)
        return z


def calc(reduction, label):
    model = Transformer(
        d_model=64, num_encoder_layers=2, num_decoder_layers=2,
        nhead=2, reduction=reduction, dropout=0.
    )
    model.eval()

    dummy = torch.zeros(1, 2, 32, 32)

    # ── Total FLOPs ──────────────────────────────────────────────────────────
    total_flops, total_params = profile(model, inputs=(dummy,), verbose=False)

    # ── Encoder FLOPs ────────────────────────────────────────────────────────
    enc_model = TransNetEncoder(model)
    enc_flops, enc_params = profile(enc_model, inputs=(dummy,), verbose=False)

    print(f"\n{'='*50}")
    print(f"  {label}  (reduction={reduction})")
    print(f"{'='*50}")
    print(f"  Total   FLOPs : {total_flops/1e6:.2f} M")
    print(f"  Encoder FLOPs : {enc_flops/1e6:.2f} M")
    print(f"  Decoder FLOPs : {(total_flops-enc_flops)/1e6:.2f} M")
    print(f"  Total  Params : {total_params/1e6:.3f} M")
    print(f"  Enc    Params : {enc_params/1e6:.3f} M")

    return total_flops, enc_flops


if __name__ == "__main__":
    print("\nPaper reference values (outdoor, FP32):")
    print("  CR=1/4 : Total=35.72M,  NMSE=-14.86 dB")
    print("  CR=1/16: Total=34.14M,  NMSE= -7.82 dB")

    calc(reduction=4,  label="CR = 1/4")
    calc(reduction=16, label="CR = 1/16")
