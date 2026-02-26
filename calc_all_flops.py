"""
Unified FLOPs calculator for all baseline models (paper-consistent convention).
BN and activation FLOPs are zeroed out to match paper reporting style.

Models: CsiNet, CsiNet+, CLNet, TransNet
CRs:    1/4 (reduction=4), 1/16 (reduction=16)
"""
import sys, os, types
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from thop import profile

ROOT = os.path.dirname(__file__)

# -- Stub logger so CLNet/TransNet don't crash on import -----------------------
fake_utils = types.ModuleType("utils")
class _Logger:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
fake_utils.logger = _Logger()
sys.modules["utils"] = fake_utils

# -- thop: zero out BN + activations (paper convention) ------------------------
def zero_ops(m, x, y):
    m.total_ops += torch.zeros(1, dtype=torch.float64)

ZERO_OPS = {
    nn.BatchNorm2d:  zero_ops,
    nn.BatchNorm1d:  zero_ops,
    nn.ReLU:         zero_ops,
    nn.LeakyReLU:    zero_ops,
    nn.Sigmoid:      zero_ops,
    nn.Tanh:         zero_ops,
    nn.ReLU6:        zero_ops,
    nn.Identity:     zero_ops,
}

def flops(model, dummy):
    model.eval()
    f, p = profile(model, inputs=(dummy,), custom_ops=ZERO_OPS, verbose=False)
    return f, p

def report(label, total_f, enc_f, total_p, enc_p, ref_total=None):
    print(f"\n  {'-'*46}")
    print(f"  {label}")
    print(f"  {'-'*46}")
    print(f"  Total   FLOPs : {total_f/1e6:.2f} M"
          + (f"  (paper: {ref_total} M)" if ref_total else ""))
    print(f"  Encoder FLOPs : {enc_f/1e6:.2f} M")
    print(f"  Decoder FLOPs : {(total_f-enc_f)/1e6:.2f} M")
    print(f"  Total  Params : {total_p/1e6:.3f} M")
    print(f"  Enc    Params : {enc_p/1e6:.3f} M")
    return total_f, enc_f


# ==============================================================================
# 1. CsiNet  (Python_CsiNet-master/Improvement/CsiNet_Train.py)
# ==============================================================================
class RefineNet(nn.Module):
    def __init__(self, img_channels=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 8,  3, padding=1, bias=False), nn.BatchNorm2d(8),  nn.LeakyReLU(0.3),
            nn.Conv2d(8, 16, 3, padding=1,            bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 2, 3, padding=1,            bias=False), nn.BatchNorm2d(2))
        self.relu = nn.LeakyReLU(0.3)
    def forward(self, x):
        return self.relu(self.conv(x) + x)

class CsiNet(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.conv1  = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1, bias=False), nn.BatchNorm2d(2), nn.LeakyReLU(0.3))
        self.dense  = nn.Sequential(nn.Linear(2048, encoded_dim), nn.Tanh(), nn.Linear(encoded_dim, 2048))
        self.decode = nn.ModuleList([RefineNet() for _ in range(2)])
        self.conv2  = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1, bias=False), nn.Sigmoid())
    def forward(self, x):
        b = x.shape[0]
        x = self.conv1(x).view(b, -1)
        x = self.dense(x).view(b, 2, 32, 32)
        for l in self.decode: x = l(x)
        return self.conv2(x)

class CsiNetEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv1  = base.conv1
        self.fc_enc = base.dense[0]   # Linear(2048, dim)
    def forward(self, x):
        b = x.shape[0]
        return self.fc_enc(self.conv1(x).view(b, -1))


# ==============================================================================
# 2. CsiNet+  (CsiNetPlus/network.py)
# ==============================================================================
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_csinet_plus_mod = load_module("csinet_plus_network",
    os.path.join(ROOT, "CsiNetPlus", "network.py"))
CsiNetPlus = _csinet_plus_mod.CsiNetPlus

class CsiNetPlusEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder_conv = base.encoder_conv
        self.encoder_fc   = base.encoder_fc
    def forward(self, x):
        n = x.shape[0]
        return self.encoder_fc(self.encoder_conv(x).view(n, -1))


# ==============================================================================
# 3. CLNet  (CLNet-master/models/clnet.py)
# ==============================================================================
_clnet_mod = load_module("clnet_models_clnet",
    os.path.join(ROOT, "CLNet-master", "models", "clnet.py"))
CLNet = _clnet_mod.CLNet

class CLNetEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder = base.encoder
    def forward(self, x):
        return self.encoder(x)


# ==============================================================================
# 4. TransNet  (TransNet-master/models/TransNet.py)
# ==============================================================================
_transnet_mod = load_module("transnet_models_TransNet",
    os.path.join(ROOT, "TransNet-master", "models", "TransNet.py"))
Transformer = _transnet_mod.Transformer

class TransNetEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.feature_shape = base.feature_shape
        self.encoder    = base.encoder
        self.fc_encoder = base.fc_encoder
    def forward(self, x):
        feat = x.view(-1, self.feature_shape[0], self.feature_shape[1])
        mem  = self.encoder(feat)
        return self.fc_encoder(mem.view(mem.shape[0], -1))


# ==============================================================================
# Main
# ==============================================================================
results = {}   # {(model, cr): (total_f, enc_f)}

dummy = torch.zeros(1, 2, 32, 32)

for reduction, cr_label in [(4, "CR=1/4"), (16, "CR=1/16")]:
    print(f"\n{'='*50}")
    print(f"  {cr_label}")
    print(f"{'='*50}")

    encoded_dim = 2048 // reduction

    # -- CsiNet --------------------------------------------------------------
    m   = CsiNet(encoded_dim=encoded_dim)
    enc = CsiNetEncoder(m)
    tf, tp = flops(m,   dummy)
    ef, ep = flops(enc, dummy)
    results[("CsiNet", cr_label)] = (tf, ef)
    report(f"CsiNet  (dim={encoded_dim})", tf, ef, tp, ep,
           ref_total={4: "10.89", 16: "7.74"}[reduction])

    # -- CsiNet+ -------------------------------------------------------------
    m   = CsiNetPlus(reduction=reduction)
    enc = CsiNetPlusEncoder(m)
    tf, tp = flops(m,   dummy)
    ef, ep = flops(enc, dummy)
    results[("CsiNet+", cr_label)] = (tf, ef)
    report(f"CsiNet+ (reduction={reduction})", tf, ef, tp, ep,
           ref_total={4: "24.57", 16: "23.00"}[reduction])

    # -- CLNet ----------------------------------------------------------------
    m   = CLNet(reduction=reduction)
    enc = CLNetEncoder(m)
    tf, tp = flops(m,   dummy)
    ef, ep = flops(enc, dummy)
    results[("CLNet", cr_label)] = (tf, ef)
    report(f"CLNet   (reduction={reduction})", tf, ef, tp, ep,
           ref_total={4: "4.54", 16: "2.97"}[reduction])

    # -- TransNet -------------------------------------------------------------
    m   = Transformer(d_model=64, num_encoder_layers=2, num_decoder_layers=2,
                      nhead=2, reduction=reduction, dropout=0.)
    enc = TransNetEncoder(m)
    tf, tp = flops(m,   dummy)
    ef, ep = flops(enc, dummy)
    results[("TransNet", cr_label)] = (tf, ef)
    report(f"TransNet (reduction={reduction})", tf, ef, tp, ep,
           ref_total={4: "35.78", 16: "34.21"}[reduction])

# -- Summary table -------------------------------------------------------------
print(f"\n\n{'='*66}")
print(f"  SUMMARY  (BN/activation excluded)")
print(f"{'='*66}")
print(f"  {'Model':<14} {'CR':<8} {'Enc FLOPs (M)':>14} {'Total FLOPs (M)':>16}")
print(f"  {'-'*60}")
for cr in ["CR=1/4", "CR=1/16"]:
    for mdl in ["CsiNet", "CsiNet+", "CLNet", "TransNet"]:
        tf, ef = results[(mdl, cr)]
        print(f"  {mdl:<14} {cr:<8} {ef/1e6:>14.2f} {tf/1e6:>16.2f}")
    print()
