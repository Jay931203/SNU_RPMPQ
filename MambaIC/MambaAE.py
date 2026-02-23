# MambaAE.py
# Asymmetric CSI Feedback Autoencoder (Section III)
# UE-side SSM encoder f_theta + BS-side decoder g_phi

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial, lru_cache
from einops import repeat, rearrange

# --- [Import Check] ---
# VMamba SS2D CUDA kernel and mamba-ssm for 1D SSM operations
try:
    from models.VSS_module import SS2D as VSSBlock
except ImportError:
    print("Warning: models.VSS_module not found. 'chunked_ss2d' disabled.")
    VSSBlock = None

try:
    from mamba_ssm import Mamba as Mamba1D
except ImportError:
    print("Warning: mamba_ssm not found. 1D Scan modes disabled.")
    Mamba1D = None

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except ImportError:
    # print("Warning: 'hilbertcurve' not found.")
    HilbertCurve = None


# --- [Quantization: Feedback & Activation (Section II-B)] ---

class UniformQuantizer_0_1(torch.autograd.Function):
    """Latent feedback quantization Q_fb (Eq.6): [0,1] uniform"""
    @staticmethod
    def forward(ctx, x, bits=8):
        n_levels = 2 ** bits
        x = torch.clamp(x, 0.0, 1.0)
        x_q = torch.round(x * (n_levels - 1))
        denom = n_levels - 1
        if denom == 0: denom = 1
        return x_q / denom

    @staticmethod
    def backward(ctx, grad_output): return grad_output, None

class DynamicActivationQuantizer(nn.Module):
    """Dynamic activation quantizer for UE-side inference (asymmetric min-max)"""
    def __init__(self, bits=32):
        super().__init__()
        self.bits = bits
        self.active = (bits < 32)
    
    def forward(self, x):
        if not self.active: return x
        
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val: return x

        # Asymmetric scale & zero-point
        q_max = (2 ** self.bits) - 1
        scale = (max_val - min_val) / q_max
        zero_point = torch.round(-min_val / scale)
        
        # Fake quantization with STE (Straight-Through Estimator)
        x_q = torch.clamp(torch.round(x / scale + zero_point), 0, q_max)
        x_deq = (x_q - zero_point) * scale
        return (x_deq - x).detach() + x


# --- [Helper Classes & Functions] ---

@lru_cache(maxsize=4)
def _get_hilbert_indices(chunk_size):
    if HilbertCurve is None:
        raise ImportError("scan_mode='chunked_hilbert' requires 'hilbertcurve' library.")
    if chunk_size == 1: return torch.tensor([0]), torch.tensor([0])
    if not (chunk_size & (chunk_size - 1) == 0) and chunk_size != 0:
        raise ValueError(f"Hilbert chunk_size must be a power of 2, but got {chunk_size}")
    p = int(math.log2(chunk_size))
    ndim = 2
    hilbert_curve = HilbertCurve(p, ndim)
    L = chunk_size * chunk_size
    coords = hilbert_curve.points_from_distances(list(range(L)))
    coords = torch.tensor(coords, dtype=torch.long)
    indices_1d = coords[:, 0] * chunk_size + coords[:, 1]
    inv_indices = torch.empty_like(indices_1d)
    inv_indices[indices_1d] = torch.arange(L)
    return inv_indices, indices_1d

def conv(in_channels, out_channels, kernel_size=3, stride=2, bias=True):
    if isinstance(kernel_size, tuple): padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else: padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def deconv(in_channels, out_channels, kernel_size=3, stride=2, bias=True):
    if isinstance(kernel_size, tuple): padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else: padding = kernel_size // 2
    if isinstance(stride, tuple): out_pad = (stride[0] - 1, stride[1] - 1)
    else: out_pad = stride - 1
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              output_padding=out_pad, padding=padding, bias=bias)

class PermuteBCHWtoBHWC(nn.Module):
    def forward(self, x): return x.permute(0, 2, 3, 1).contiguous()

class PermuteBHWCtoBCHW(nn.Module):
    def forward(self, x): return x.permute(0, 3, 1, 2).contiguous()

class StateNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

class BeamMaskModule(nn.Module):
    def __init__(self, channels, init_bias=3.0):
        super().__init__()
        hidden_dim = max(16, channels // 4)
        self.mlp_fc1 = nn.Linear(channels, hidden_dim)
        self.mlp_act = nn.SiLU()
        self.mlp_fc2 = nn.Linear(hidden_dim, 1)
        if init_bias is not None:
            torch.nn.init.constant_(self.mlp_fc2.bias, init_bias)
    def forward(self, x):
        x_mean_delay = F.adaptive_avg_pool2d(x, (1, x.shape[3])).squeeze(2)
        x_permuted = x_mean_delay.permute(0, 2, 1).contiguous()
        x_hidden = self.mlp_act(self.mlp_fc1(x_permuted))
        beam_logits = self.mlp_fc2(x_hidden)
        beam_score = torch.sigmoid(beam_logits)
        mask = beam_score.permute(0, 2, 1).unsqueeze(2)
        return mask


# --- [Mamba Blocks with Quantization & Chunking] ---

class ChunkedResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dpr, chunk_size=8, use_beam_mask=False, use_chunking=True, act_bits=32):
        super().__init__()
        if VSSBlock is None: raise ImportError("Requires models.VSS_module.SS2D")
        self.chunk_size = chunk_size
        self.use_chunking = use_chunking
        self.norm = StateNorm(d_model)
        self.act = nn.SiLU()
        self.act_quant = DynamicActivationQuantizer(act_bits)
        
        self.vss = nn.Sequential(
            PermuteBCHWtoBHWC(),
            VSSBlock(d_model=d_model, drop_path=dpr),
            PermuteBHWCtoBCHW(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_beam_mask = use_beam_mask
        self.beam_mask_module = BeamMaskModule(d_model) if use_beam_mask else None

    def forward(self, x):
        B, C, H, W = x.shape
        cs = self.chunk_size
        x_norm = self.norm(x)
        
        x_act = self.act(x_norm)
        x_act = self.act_quant(x_act)

        if self.use_beam_mask and self.beam_mask_module:
            x_act = x_act * self.beam_mask_module(x_act)
            x_act = self.act_quant(x_act)

        # Global Scan Logic
        if not self.use_chunking:
            return x + self.alpha * self.vss(x_act)

        # Chunked Scan Logic
        try:
            x_chunked = rearrange(x_act, 'b c (h_num cs_h) (w_num cs_w) -> (b h_num w_num) c cs_h cs_w', cs_h=cs, cs_w=cs)
        except Exception as e:
            return x + self.alpha * self.vss(x_act)
            
        x_vss_chunked = self.vss(x_chunked)
        x_vss = rearrange(x_vss_chunked, '(b h_num w_num) c cs_h cs_w -> b c (h_num cs_h) (w_num cs_w)', h_num=H//cs, w_num=W//cs, cs_h=cs, cs_w=cs)
        return x + self.alpha * x_vss

class HilbertResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dpr, chunk_size=8, use_beam_mask=False, use_chunking=True, act_bits=32):
        super().__init__()
        if Mamba1D is None: raise ImportError("Requires mamba_ssm")
        self.chunk_size = chunk_size
        self.norm = StateNorm(d_model); self.act = nn.SiLU()
        self.act_quant = DynamicActivationQuantizer(act_bits)
        
        self.mamba_fwd = Mamba1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.permute_idx, self.unpermute_idx = _get_hilbert_indices(chunk_size)
        self.use_beam_mask = use_beam_mask
        self.beam_mask_module = BeamMaskModule(d_model) if use_beam_mask else None

    def forward(self, x):
        B, C, H, W = x.shape; cs = self.chunk_size
        x_norm = self.norm(x)
        x_act = self.act_quant(self.act(x_norm))
        
        if self.use_beam_mask and self.beam_mask_module: 
            x_act = self.act_quant(x_act * self.beam_mask_module(x_act))

        try:
            x_chunked = rearrange(x_act, 'b c (h_num cs_h) (w_num cs_w) -> (b h_num w_num) c cs_h cs_w', cs_h=cs, cs_w=cs)
        except: return x

        B_chunk = x_chunked.shape[0]
        x_flat = rearrange(x_chunked, 'b c n m -> b c (n m)')
        idx = self.permute_idx.to(x.device)
        x_hilbert = rearrange(x_flat, 'b c l -> (b c) l').index_select(dim=-1, index=idx)
        x_in = rearrange(x_hilbert, '(b c) l -> b c l', b=B_chunk).transpose(-1, -2).contiguous()
        x_ssm = self.mamba_fwd(x_in) + self.mamba_bwd(x_in.flip(dims=[1])).flip(dims=[1])
        x_out_flat = rearrange(x_ssm.transpose(-1, -2), 'b c l -> (b c) l').index_select(dim=-1, index=self.unpermute_idx.to(x.device))
        x_out_chunked = rearrange(x_out_flat, '(b c) (n m) -> b c n m', b=B_chunk, n=cs, m=cs)
        x_vss = rearrange(x_out_chunked, '(b h_num w_num) c cs_h cs_w -> b c (h_num cs_h) (w_num cs_w)', h_num=H//cs, w_num=W//cs)
        return x + self.alpha * x_vss

class RasterResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dpr, chunk_size=8, use_beam_mask=False, use_chunking=True, act_bits=32):
        super().__init__()
        if Mamba1D is None: raise ImportError("Requires mamba_ssm")
        self.chunk_size = chunk_size; self.norm = StateNorm(d_model); self.act = nn.SiLU()
        self.act_quant = DynamicActivationQuantizer(act_bits)
        
        self.mamba_fwd = Mamba1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_beam_mask = use_beam_mask
        self.beam_mask_module = BeamMaskModule(d_model) if use_beam_mask else None

    def forward(self, x):
        B, C, H, W = x.shape; cs = self.chunk_size
        x_norm = self.norm(x)
        x_act = self.act_quant(self.act(x_norm))
        if self.use_beam_mask and self.beam_mask_module: x_act = self.act_quant(x_act * self.beam_mask_module(x_act))
        
        try:
            x_chunked = rearrange(x_act, 'b c (h_num cs_h) (w_num cs_w) -> (b h_num w_num) c cs_h cs_w', cs_h=cs, cs_w=cs)
        except: return x
        B_chunk, C_chunk, H_c, W_c = x_chunked.shape
        x_flat = x_chunked.permute(0, 1, 3, 2).reshape(B_chunk, C_chunk, -1).transpose(-1, -2).contiguous()
        x_ssm = self.mamba_fwd(x_flat) + self.mamba_bwd(x_flat.flip(dims=[1])).flip(dims=[1])
        x_out_chunked = x_ssm.transpose(-1, -2).reshape(B_chunk, C_chunk, W_c, H_c).permute(0, 1, 3, 2)
        x_vss = rearrange(x_out_chunked, '(b h_num w_num) c cs_h cs_w -> b c (h_num cs_h) (w_num cs_w)', h_num=H//cs, w_num=W//cs)
        return x + self.alpha * x_vss

class BiaxialResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dpr, chunk_size=8, use_beam_mask=False, use_chunking=True, act_bits=32):
        super().__init__()
        if Mamba1D is None: raise ImportError("Requires mamba_ssm")
        self.chunk_size = chunk_size; self.norm = StateNorm(d_model); self.act = nn.SiLU()
        self.act_quant = DynamicActivationQuantizer(act_bits)
        
        self.row_fwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.row_bwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.col_fwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.col_bwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_beam_mask = use_beam_mask
        self.beam_mask_module = BeamMaskModule(d_model) if use_beam_mask else None

    def forward(self, x):
        B, C, H, W = x.shape; cs = self.chunk_size
        x_norm = self.norm(x)
        x_act = self.act_quant(self.act(x_norm))
        if self.use_beam_mask and self.beam_mask_module: x_act = self.act_quant(x_act * self.beam_mask_module(x_act))

        try:
            x_chunked = rearrange(x_act, 'b c (h_num cs_h) (w_num cs_w) -> (b h_num w_num) c cs_h cs_w', cs_h=cs, cs_w=cs)
        except: return x
        H_c, W_c = cs, cs
        x_row = rearrange(x_chunked, 'b c h w -> (b h) w c')
        x_row_ssm = self.row_fwd(x_row) + self.row_bwd(x_row.flip(dims=[1])).flip(dims=[1])
        x_row_2d = rearrange(x_row_ssm, '(b h) w c -> b c h w', h=H_c)
        x_col = rearrange(x_chunked, 'b c h w -> (b w) h c')
        x_col_ssm = self.col_fwd(x_col) + self.col_bwd(x_col.flip(dims=[1])).flip(dims=[1])
        x_col_2d = rearrange(x_col_ssm, '(b w) h c -> b c h w', w=W_c)
        x_vss = rearrange(x_row_2d + x_col_2d, '(b h_num w_num) c cs_h cs_w -> b c (h_num cs_h) (w_num cs_w)', h_num=H//cs, w_num=W//cs)
        return x + self.alpha * x_vss

class UniaxialHResidualMambaBlock(nn.Module):
    def __init__(self, d_model, dpr, chunk_size=8, use_beam_mask=False, use_chunking=True, act_bits=32):
        super().__init__()
        if Mamba1D is None: raise ImportError("Requires mamba_ssm")
        self.chunk_size = chunk_size; self.norm = StateNorm(d_model); self.act = nn.SiLU()
        self.act_quant = DynamicActivationQuantizer(act_bits)

        self.col_fwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.col_bwd = Mamba1D(d_model, d_state=16, d_conv=4, expand=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_beam_mask = use_beam_mask
        self.beam_mask_module = BeamMaskModule(d_model) if use_beam_mask else None

    def forward(self, x):
        B, C, H, W = x.shape; cs = self.chunk_size
        x_norm = self.norm(x)
        x_act = self.act_quant(self.act(x_norm))
        if self.use_beam_mask and self.beam_mask_module: x_act = self.act_quant(x_act * self.beam_mask_module(x_act))

        try:
            x_chunked = rearrange(x_act, 'b c (h_num cs_h) (w_num cs_w) -> (b h_num w_num) c cs_h cs_w', cs_h=cs, cs_w=cs)
        except: return x
        W_c = cs
        x_col = rearrange(x_chunked, 'b c h w -> (b w) h c')
        x_col_ssm = self.col_fwd(x_col) + self.col_bwd(x_col.flip(dims=[1])).flip(dims=[1])
        x_out_chunked = rearrange(x_col_ssm, '(b w) h c -> b c h w', w=W_c)
        x_vss = rearrange(x_out_chunked, '(b h_num w_num) c cs_h cs_w -> b c (h_num cs_h) (w_num cs_w)', h_num=H//cs, w_num=W//cs)
        return x + self.alpha * x_vss


# --- [Asymmetric CSI Feedback Autoencoder (Section III)] ---

class MambaAE(nn.Module):
    def __init__(self, depths=[2, 2, 4, 2], drop_path_rate=0.1, N=128, M=32, bits=8,
                 chunk_sizes=[8, 8, 4],
                 scan_mode="chunked_ss2d",
                 use_beam_mask=False,
                 compression_mode="default",
                 use_chunking=True,
                 act_quant_bits=32,
                 **kwargs):
        super().__init__()

        self.encoder_type = "mamba_ae"
        self.N = N; self.M = M; self.bits = bits
        self.depths_enc = depths[:3]; self.depths_dec = depths[:3]
        self.chunk_sizes = chunk_sizes
        self.scan_mode = scan_mode
        self.use_beam_mask = use_beam_mask
        self.compression_mode = compression_mode
        self.use_chunking = use_chunking
        self.act_quant_bits = act_quant_bits 
        
        self.input_quant = DynamicActivationQuantizer(act_quant_bits)

        print(f"[INFO] Asymmetric CSI AE Initialized (UE: SSM Encoder, BS: Decoder) | Scan: {self.scan_mode}, Mask: {self.use_beam_mask}, Comp: {self.compression_mode}, Chunking: {self.use_chunking}, ActQuant: {self.act_quant_bits}-bit")

        if "ss2d" in scan_mode and VSSBlock is None: raise ImportError("SS2D missing")
        if ("hilbert" in scan_mode or "biaxial" in scan_mode or "raster" in scan_mode) and Mamba1D is None:
            raise ImportError("Mamba1D missing")

        total_blocks = sum(self.depths_enc) + sum(self.depths_dec)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dpr_idx = 0

        self.latent_tanh = nn.Tanh()
        self.final_sigmoid = nn.Sigmoid()

        if self.compression_mode == "hybrid":
            s1, s2, s3 = (2, 4), (2,2), (2, 1)
            k1 = (5, 5)
            print(f"[INFO] Hybrid Strides: {s1}(K{k1}) -> {s2} -> {s3}")
        else:
            s1, s2, s3 = 2, 2, 2
            k1 = 5
        
        # --- UE Encoder f_theta (SSM-based, Section III-D) ---
        self.g_a_conv1 = conv(2, N, kernel_size=k1, stride=s1)
        self.g_a_stage1 = self.make_stage(self.depths_enc[0], N, dpr, dpr_idx, self.chunk_sizes[0])
        dpr_idx += self.depths_enc[0]
        
        self.g_a_conv2 = conv(N, N, kernel_size=3, stride=s2)
        self.g_a_stage2 = self.make_stage(self.depths_enc[1], N, dpr, dpr_idx, self.chunk_sizes[1])
        dpr_idx += self.depths_enc[1]
        
        self.g_a_conv3 = conv(N, M, kernel_size=3, stride=s3)
        self.g_a_stage3 = self.make_stage(self.depths_enc[2], M, dpr, dpr_idx, self.chunk_sizes[2])
        dpr_idx += self.depths_enc[2]

        # --- BS Decoder g_phi (Section III-E) ---
        self.g_s_stage3 = self.make_stage(self.depths_dec[2], M, dpr, dpr_idx, self.chunk_sizes[2])
        dpr_idx += self.depths_dec[2]
        self.g_s_deconv3 = deconv(M, N, kernel_size=3, stride=s3)
        
        self.g_s_stage2 = self.make_stage(self.depths_dec[1], N, dpr, dpr_idx, self.chunk_sizes[1])
        dpr_idx += self.depths_dec[1]
        self.g_s_deconv2 = deconv(N, N, kernel_size=3, stride=s2)
        
        self.g_s_stage1 = self.make_stage(self.depths_dec[0], N, dpr, dpr_idx, self.chunk_sizes[0])
        dpr_idx += self.depths_dec[0]
        self.g_s_deconv1 = deconv(N, 2, kernel_size=k1, stride=s1)

        self.apply(self._init_weights)

    def make_stage(self, n_blocks, d_model, dpr_list, dpr_idx_start, chunk_size):
        if n_blocks == 0: return nn.Identity()
        blocks = []
        for i in range(n_blocks):
            dpr_val = dpr_list[dpr_idx_start + i]
            kwargs = {
                "d_model": d_model, "dpr": dpr_val, 
                "chunk_size": chunk_size, "use_beam_mask": self.use_beam_mask,
                "use_chunking": self.use_chunking,
                "act_bits": self.act_quant_bits
            }
            if self.scan_mode == "chunked_hilbert": blocks.append(HilbertResidualMambaBlock(**kwargs))
            elif self.scan_mode == "chunked_biaxial": blocks.append(BiaxialResidualMambaBlock(**kwargs))
            elif self.scan_mode == "chunked_raster": blocks.append(RasterResidualMambaBlock(**kwargs))
            elif self.scan_mode == "chunked_uniaxial_h": blocks.append(UniaxialHResidualMambaBlock(**kwargs))
            elif self.scan_mode == "chunked_ss2d": blocks.append(ChunkedResidualMambaBlock(**kwargs))
            else: raise ValueError(f"Unknown scan_mode: {self.scan_mode}")
        return nn.Sequential(*blocks)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, StateNorm):
            nn.init.constant_(m.gamma, 1.0)
            nn.init.constant_(m.beta, 0.0)

    def f_theta(self, x):
        """UE encoder f_theta: X_a -> z (Eq.4)"""
        x = self.input_quant(x)
        x = self.g_a_stage1(self.g_a_conv1(x))
        x = self.g_a_stage2(self.g_a_conv2(x))
        x = self.g_a_stage3(self.g_a_conv3(x))
        return self.latent_tanh(x)

    def g_phi(self, y_hat):
        """BS decoder g_phi: z_hat -> X_hat (Eq.7)"""
        x = self.g_s_deconv3(self.g_s_stage3(y_hat))
        x = self.g_s_deconv2(self.g_s_stage2(x))
        x = self.g_s_deconv1(self.g_s_stage1(x))
        return self.final_sigmoid(x)

    def forward(self, x):
        y_norm = (self.f_theta(x) + 1) / 2
        if self.bits < 1: y_hat = y_norm
        else:
            is_warmup = getattr(self, "disable_quant", False)
            if self.bits == 1: is_warmup = False; n_levels = 2
            else: n_levels = 2 ** self.bits
            if is_warmup:
                denom = (n_levels - 1) * 0.5
                noise = (torch.rand_like(y_norm) - 0.5) / (denom if denom != 0 else 1.0)
                y_hat = y_norm + noise
            else:
                y_hat = UniformQuantizer_0_1.apply(y_norm, self.bits)
        return self.g_phi(y_hat)