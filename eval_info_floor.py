"""
Experimental validation of Eq. (10): Encoder-induced information floor.

Fix encoder, vary decoder capacity → MSE converges to a floor.
  floor = encoder-induced information floor
  gap above floor = decoder refinement error
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

PROJECT  = os.path.dirname(os.path.abspath(__file__))
MAMBA_IC = os.path.join(PROJECT, "MambaIC")
DATA     = os.path.join(MAMBA_IC, "data", "DATA_Htestout.mat")
SAVE_DIR = os.path.join(MAMBA_IC, "saved_models")
OUT_DIR  = os.path.join(PROJECT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Compatible model definitions (matching saved checkpoint key structure)
# ═══════════════════════════════════════════════════════════════════════

# --- CsiNet ---
class CsiNetEncoderCompat(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, 1, 1)
        self.bn = nn.BatchNorm2d(2)
        self.act = nn.LeakyReLU(0.3)
        self.fc = nn.Linear(2*32*32, encoded_dim)
    def forward(self, x):
        return self.fc(self.act(self.bn(self.conv(x))).view(-1, 2*32*32))

class CsiNetRefineBlockCompat(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(2, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.LeakyReLU(0.3),
            nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 2, 3, 1, 1), nn.BatchNorm2d(2))
        self.act = nn.LeakyReLU(0.3)
    def forward(self, x):
        return self.act(x + self.seq(x))

class CsiNetDecoderCompat(nn.Module):
    def __init__(self, encoded_dim=512, num_layers=2):
        super().__init__()
        self.fc_dec = nn.Linear(encoded_dim, 2*32*32)
        self.refine = nn.Sequential(*[CsiNetRefineBlockCompat() for _ in range(num_layers)])
        self.final_conv = nn.Conv2d(2, 2, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, z):
        x = self.fc_dec(z).view(-1, 2, 32, 32)
        x = self.refine(x)
        return self.sigmoid(self.final_conv(x))

# --- TransNet (compatible with saved checkpoints) ---
# Checkpoints use: separate Q/K/V projections, single shared layer

class CompatMultiheadAttention(nn.Module):
    """MHA with separate Q/K/V projections (matching checkpoint keys)."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights=False):
        E = self.embed_dim
        q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:E])
        k = F.linear(key, self.k_proj_weight, self.in_proj_bias[E:2*E])
        v = F.linear(value, self.v_proj_weight, self.in_proj_bias[2*E:])

        # reshape for multi-head: (L, B, E) -> (B*H, L, D)
        L, B, _ = q.shape
        q = q.view(L, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, B * self.num_heads, self.head_dim).transpose(0, 1)

        attn = torch.bmm(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.bmm(attn, v)
        out = out.transpose(0, 1).contiguous().view(L, B, E)
        out = self.out_proj(out)
        return out, None

class CompatTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CompatMultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = CompatMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class CompatTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CompatMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src

class WeightSharedDecoder(nn.Module):
    """Single layer looped num_layers times (key: decoder.layer.*)"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
    def forward(self, tgt, memory):
        for _ in range(self.num_layers):
            tgt = self.layer(tgt, memory)
        return tgt

class WeightSharedEncoder(nn.Module):
    """Single layer looped num_layers times (key: encoder.layer.*)"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
    def forward(self, src):
        for _ in range(self.num_layers):
            src = self.layer(src)
        return src

class TransNetEncoderCompat(nn.Module):
    def __init__(self, encoded_dim=512, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.feature_shape = (2048 // d_model, d_model)
        layer = CompatTransformerEncoderLayer(d_model, nhead=2, dim_feedforward=2048)
        self.encoder = WeightSharedEncoder(layer, num_layers)
        self.fc_encoder = nn.Linear(2048, encoded_dim)
    def forward(self, x):
        src = x.view(-1, self.feature_shape[0], self.feature_shape[1])
        memory = self.encoder(src)
        return self.fc_encoder(memory.reshape(memory.shape[0], -1))

class TransNetDecoderCompat(nn.Module):
    def __init__(self, encoded_dim=512, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.feature_shape = (2048 // d_model, d_model)
        self.fc_decoder = nn.Linear(encoded_dim, 2048)
        layer = CompatTransformerDecoderLayer(d_model, nhead=2, dim_feedforward=2048)
        self.decoder = WeightSharedDecoder(layer, num_layers)
    def forward(self, z):
        mem = self.fc_decoder(z).view(-1, self.feature_shape[0], self.feature_shape[1])
        out = self.decoder(mem, mem)
        return out.view(-1, 2, 32, 32)

# --- MobileNet ---
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(True),
            nn.Linear(ch//r, ch, bias=False), nn.Hardsigmoid(True))
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)

class MBBlock(nn.Module):
    def __init__(self, ic, oc, ks, s, ef):
        super().__init__()
        self.use_res = (s == 1 and ic == oc)
        hd = int(ic * ef); layers = []
        if ef != 1:
            layers += [nn.Conv2d(ic, hd, 1, bias=False), nn.BatchNorm2d(hd), nn.Hardswish(True)]
        layers += [nn.Conv2d(hd, hd, ks, s, ks//2, groups=hd, bias=False),
                   nn.BatchNorm2d(hd), nn.Hardswish(True), SEBlock(hd),
                   nn.Conv2d(hd, oc, 1, bias=False), nn.BatchNorm2d(oc)]
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_res else self.conv(x)

class MobileNetEncoderCompat(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.Hardswish(True))
        self.blocks = nn.Sequential(
            MBBlock(16, 16, 3, 1, 1), MBBlock(16, 24, 3, 2, 4),
            MBBlock(24, 24, 3, 1, 3), MBBlock(24, 40, 5, 2, 3),
            MBBlock(40, 40, 5, 1, 3), MBBlock(40, 80, 5, 2, 3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 128, 1, bias=False), nn.BatchNorm2d(128), nn.Hardswish(True))
        self.fc = nn.Linear(128*4*4, encoded_dim)
    def forward(self, x):
        x = self.conv1(x); x = self.blocks(x); x = self.conv2(x)
        return self.fc(x.flatten(1))

class CompatAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

# ═══════════════════════════════════════════════════════════════════════
# Flexible state dict loading
# ═══════════════════════════════════════════════════════════════════════
def clean_and_load(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # 1. Strip _orig_mod. prefix
    cleaned = OrderedDict()
    for k, v in state.items():
        k2 = k.replace("_orig_mod.", "")
        cleaned[k2] = v

    # 2. Remove thop artifacts
    cleaned = OrderedDict((k, v) for k, v in cleaned.items()
                          if "total_ops" not in k and "total_params" not in k)

    # 3. Load (strict=False to handle minor mismatches)
    info = model.load_state_dict(cleaned, strict=False)
    n_missing = len(info.missing_keys)
    n_unexpected = len(info.unexpected_keys)
    if n_missing > 0:
        print(f"  [WARN] {n_missing} missing keys")
    return model, n_missing

# ═══════════════════════════════════════════════════════════════════════
# Data & NMSE
# ═══════════════════════════════════════════════════════════════════════
print(f"Loading test data...")
mat = sio.loadmat(DATA)
csi = mat["HT"].astype(np.float32)
if csi.ndim == 2: csi = csi.reshape(-1, 2, 32, 32)
min_val = csi.min(); range_val = csi.max() - min_val + 1e-9
X = torch.from_numpy((csi - min_val) / range_val)
print(f"  {X.shape[0]} samples, outdoor")

@torch.no_grad()
def compute_nmse_db(model, X, bs=200):
    model.eval()
    mses, pows = [], []
    for i in range(0, len(X), bs):
        xb = X[i:i+bs]
        xr = model(xb)
        oc, rc = xb - 0.5, xr - 0.5
        pows.append(torch.sum(oc**2, dim=[1,2,3]))
        mses.append(torch.sum((oc - rc)**2, dim=[1,2,3]))
    pows, mses = torch.cat(pows), torch.cat(mses)
    v = pows > 1e-8
    nmse = torch.mean(mses[v] / pows[v])
    return 10 * torch.log10(nmse).item()

# ═══════════════════════════════════════════════════════════════════════
# Configs & Evaluation
# ═══════════════════════════════════════════════════════════════════════
def build(enc_t, dec_t, dec_L):
    enc = {"csinet": CsiNetEncoderCompat, "transnet": TransNetEncoderCompat,
           "mobilenet": MobileNetEncoderCompat}[enc_t](512)
    dec = {"csinet": lambda: CsiNetDecoderCompat(512, 2),
           "transnet": lambda: TransNetDecoderCompat(512, 64, dec_L)}[dec_t]()
    return CompatAE(enc, dec)

configs = [
    # (save_name, enc, dec, dec_layers, label_enc, label_dec)
    ("csinet_csinet_dim512",          "csinet",    "csinet",    2, "CsiNet",    "CsiNet"),
    ("csinet_transnet_L2_dim512",     "csinet",    "transnet",  2, "CsiNet",    "TransNet-L2"),
    ("transnet_transnet_L2_dim512",   "transnet",  "transnet",  2, "TransNet",  "TransNet-L2"),
    ("mobilenet_transnet_L1_dim512",  "mobilenet", "transnet",  1, "MobileNet", "TransNet-L1"),
    ("mobilenet_transnet_L2_dim512",  "mobilenet", "transnet",  2, "MobileNet", "TransNet-L2"),
    ("mobilenet_transnet_L3_dim512",  "mobilenet", "transnet",  3, "MobileNet", "TransNet-L3"),
    ("mobilenet_transnet_L4_dim512",  "mobilenet", "transnet",  4, "MobileNet", "TransNet-L4"),
]

results = []
for save_name, enc_t, dec_t, dec_L, lbl_e, lbl_d in configs:
    ckpt_path = os.path.join(SAVE_DIR, save_name, "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {save_name}")
        continue

    # Check epoch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ep = ckpt.get("epoch", -1)

    print(f"{lbl_e:10s}+{lbl_d:14s} (ep={ep:3d}) ...", end=" ", flush=True)
    try:
        model = build(enc_t, dec_t, dec_L)
        model, n_miss = clean_and_load(model, ckpt_path)
        if n_miss > 0:
            print(f"SKIP (missing keys)")
            continue
        nmse = compute_nmse_db(model, X)
        enc_p = sum(p.numel() for p in model.encoder.parameters())
        dec_p = sum(p.numel() for p in model.decoder.parameters())
        print(f"NMSE = {nmse:.2f} dB  (enc={enc_p:,}  dec={dec_p:,})")
        results.append(dict(name=save_name, enc=lbl_e, dec=lbl_d,
                           dec_layers=dec_L, nmse_db=nmse, epoch=ep,
                           enc_params=enc_p, dec_params=dec_p))
    except Exception as e:
        print(f"ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*85}")
print(f"{'Encoder':12s} {'Decoder':14s} {'Epoch':>6s} {'NMSE(dB)':>9s} {'Enc#':>10s} {'Dec#':>10s}")
print(f"{'-'*85}")
for r in results:
    print(f"{r['enc']:12s} {r['dec']:14s} {r['epoch']:6d} {r['nmse_db']:9.2f} {r['enc_params']:10,} {r['dec_params']:10,}")
print(f"{'='*85}")

if len(results) < 2:
    print("\n[INFO] Not enough successfully loaded models for a meaningful plot.")
    print("       The checkpoints use a different architecture than expected.")
    print("       → Run evaluation on Colab where CUDA + original code are available.")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'mathtext.fontset': 'stix',
    'axes.labelsize': 13, 'axes.titlesize': 14,
})

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
groups = defaultdict(list)
for r in results: groups[r["enc"]].append(r)

colors = {"CsiNet": "#D84315", "MobileNet": "#1565C0", "TransNet": "#2E7D32"}
markers = {"CsiNet": "s", "MobileNet": "o", "TransNet": "^"}

for enc_name, items in groups.items():
    items_s = sorted(items, key=lambda x: x["dec_layers"])
    xs = [r["dec_layers"] for r in items_s]
    ys = [r["nmse_db"] for r in items_s]
    c, m = colors.get(enc_name, "#333"), markers.get(enc_name, "o")
    ax.plot(xs, ys, f'-{m}', color=c, ms=9, lw=2.2, label=f'{enc_name} enc', alpha=0.85)
    for x, y, r in zip(xs, ys, items_s):
        ax.annotate(f'{r["dec"]}\n{y:.1f} dB', (x, y),
                    textcoords="offset points", xytext=(8, -5), fontsize=8.5, color=c)
    floor = min(ys)
    ax.axhline(y=floor, color=c, ls=':', lw=1.2, alpha=0.35)

ax.set_xlabel('Decoder depth (num layers)')
ax.set_ylabel('NMSE (dB)  $\\downarrow$ better')
ax.set_title('Encoder-Induced Information Floor (Eq. 10)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, ls='--', alpha=0.3)
fig.tight_layout()

for ext in ['png', 'pdf']:
    p = os.path.join(OUT_DIR, f'fig_info_floor.{ext}')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"Saved: {p}")
plt.close(fig)
