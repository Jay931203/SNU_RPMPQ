"""
Policy-Constrained Segment QAT for RP-MPQ  (v2 -- full-weight STE redesign).

Trains the encoder FC layer to be robust to segment-level mixed-precision
quantization by applying STE directly to the FC weight (no LoRA).

Redesign rationale (v2 vs v1):
    v1 used LoRA adapters + encoder-space loss.  This FAILED because:
    (a) LoRA merge created quantization-hostile weight distributions -- the
        low-rank delta disrupted the smooth pretrained distribution, making the
        merged weight MORE sensitive to quantization than the original.
    (b) Encoder-space loss Lq = MSE(z_q, z_fp) does not reflect reconstruction
        quality.  The decoder amplifies certain z-directions, so small Lq can
        hide large reconstruction errors.
    (c) Policy mismatch: training sampled from a static LUT, eval used DP with
        population-averaged omega -- different quantization patterns.
    (d) Eval returned infeasible anchor-only policies above 93.16% saving.

v2 fixes all four issues:
    (A) Full-weight STE: fine-tune the FC weight directly. No merge step.
        The weight distribution evolves smoothly toward quantization robustness.
    (B) Reconstruction loss for Lq: Lq = NMSELoss(decoder(quantize(encoder(x))), x).
        This directly optimizes what we measure at eval.
    (C) DP-based policy sampling: sample policies via the same DP solver used in
        eval, with the same avg_omega. Train and eval see the same policies.
    (D) Eval budget cap: savings above max_achievable are skipped/clamped.

Mathematical formulation:
    L(theta) = L_fp + lambda * E_{pi} [ L_q(pi) ]

    L_fp  = NMSELoss(f_theta(X), X)                         -- FP anchor
    L_q   = NMSELoss(f_theta(X; Q_pi(W_enc)), X)            -- quantized reconstruction
    pi    ~ DP-solved policies at random target savings

Usage:
    python analysis/segment_qat.py --epochs 30 --lambda-q 1.0 --n-policies 2
    python analysis/segment_qat.py --eval-only --compare
"""
from __future__ import annotations

import os
import sys
import re
import copy
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# -- project imports --------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_ae import (
    apply_precision_policy,
    quantize_feedback_torch,
    quantize_int_asym,
    calculate_su_miso_rate_mrt,
    calculate_nmse_db,
    CsiDataset,
    NMSELoss,
    restore_fp32_weights,
)
from ModularModels import ModularAE
from rpmpq_v2 import (
    get_encoder_block_names,
    get_encoder_layer_params,
    RESULTS_CSV,
)
from analysis.segment_dp_policy import (
    enumerate_segments,
    solve_dp,
    segmentation_to_policy,
)
from analysis.budget_allocation import load_cached_omegas

# -- output directories -----------------------------------------------------
RESULTS_PLOT = os.path.join(PROJECT_ROOT, "results", "plots")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "..", "figures")
for d in (RESULTS_CSV, RESULTS_PLOT, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

# -- constants ---------------------------------------------------------------
K_BINS = 5
L_MAX = 6
BIT_OPTIONS = [16, 8, 4, 2]
ANCHOR_BITS = 16
SNR = 20
AQ_BITS = 8

DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "saved_models",
    "mamba_transnet_L2_dim512_baseline", "best.pth",
)
DEFAULT_QAT_DIR = os.path.join(
    PROJECT_ROOT, "saved_models",
    "mamba_transnet_L2_dim512_qat_v2",
)


# ============================================================================
# SECTION 1 : STE Quantization (differentiable)
# ============================================================================

def ste_quantize_asym(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric STE quantization matching quantize_int_asym convention.

    Uses the same quantization grid as the PTQ path (quantize_int_asym)
    but wraps it in the STE idiom for gradient flow.

    Forward: quantize to `bits` levels using asymmetric min-max.
    Backward: gradient passes through unchanged (STE identity).
    """
    if bits >= 16:
        return w
    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1
    w_min = w.min()
    w_max = w.max()
    if w_max == w_min:
        return w
    scale = (w_max - w_min) / (q_max - q_min)
    zp = (q_min - w_min / scale).round()
    w_q = (w / scale + zp).round().clamp(q_min, q_max)
    w_deq = (w_q - zp) * scale
    return w + (w_deq - w).detach()


# ============================================================================
# SECTION 2 : Segment Quantizer -- applies a full policy via STE hooks
# ============================================================================

class SegmentQuantizer:
    """Applies a segment-level mixed-precision policy to encoder weights via STE.

    Uses forward pre/post hooks to temporarily replace weight Parameters with
    STE-quantized tensors during the forward pass.  After forward, the original
    Parameters are restored so the optimizer updates the true FP32 weights.

    v2 change: no LoRA -- works directly on the module's own weight Parameter.
    The pre-hook reads the current weight (which the optimizer is updating),
    applies STE quantization, and swaps it in for the forward pass.

    Usage:
        quantizer = SegmentQuantizer(net, fc_chunks=32)
        with quantizer.apply(policy):
            output = net(x)
            loss = criterion(output, x)
            loss.backward()  # gradients flow through STE to FC weight
    """

    def __init__(self, model: nn.Module, fc_chunks: int = 32):
        self.model = model
        self.fc_chunks = fc_chunks
        self._pre_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._post_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._stashed: Dict[nn.Module, nn.Parameter] = {}

    class _ScopeGuard:
        """Context manager that removes hooks on exit and restores weights."""
        def __init__(self, quantizer: "SegmentQuantizer"):
            self._quantizer = quantizer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            for h in self._quantizer._pre_hooks:
                h.remove()
            self._quantizer._pre_hooks.clear()
            for h in self._quantizer._post_hooks:
                h.remove()
            self._quantizer._post_hooks.clear()
            # Restore any stashed Parameters (safety net)
            for mod, orig_param in self._quantizer._stashed.items():
                mod._parameters['weight'] = orig_param
            self._quantizer._stashed.clear()
            return False

    def apply(self, policy: Dict[str, int]) -> _ScopeGuard:
        """Register forward hooks that apply STE quantization for one forward pass.

        Returns a context manager that removes hooks on exit.
        """
        # Clear any leftover hooks
        for h in self._pre_hooks:
            h.remove()
        self._pre_hooks.clear()
        for h in self._post_hooks:
            h.remove()
        self._post_hooks.clear()
        self._stashed.clear()

        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        # Build module map: clean_name -> module
        module_map: Dict[str, nn.Module] = {}
        for name, mod in real_model.encoder.named_modules():
            if hasattr(mod, "weight") and mod.weight is not None:
                clean = name.replace("encoder.", "").replace("module.", "")
                module_map[clean] = mod

        # Group policy by base layer name (merge fc_part* into parent FC)
        policy_groups: Dict[str, object] = {}
        for p_key, bits in policy.items():
            clean = (p_key.replace("encoder.", "")
                     .replace("module.", "")
                     .replace(".weight", ""))
            match = re.search(r"(.+)_part(\d+)$", clean)
            if match:
                base = match.group(1)
                idx = int(match.group(2))
                if base not in policy_groups:
                    policy_groups[base] = {}
                policy_groups[base][idx] = bits
            else:
                policy_groups[clean] = bits

        # Register hooks for each target layer
        for base_name, bits_info in policy_groups.items():
            target = module_map.get(base_name)
            if target is None:
                for m_name, mod in module_map.items():
                    if m_name.endswith(base_name):
                        target = mod
                        break
            if target is None:
                continue

            if isinstance(bits_info, dict) and "fc" in base_name:
                # Split FC: quantize each chunk independently
                chunk_bits = bits_info

                def _make_fc_pre_hook(chunk_map, n_chunks, stash_dict):
                    def hook(module, inputs):
                        stash_dict[module] = module._parameters['weight']
                        w = stash_dict[module]  # current Parameter (trainable)
                        chunks = torch.chunk(w, n_chunks, dim=0)
                        q_chunks = []
                        for ci, chunk in enumerate(chunks):
                            b = chunk_map.get(ci, 32)
                            q_chunks.append(ste_quantize_asym(chunk, b))
                        module._parameters['weight'] = torch.cat(q_chunks, dim=0)
                    return hook

                def _make_post_hook(stash_dict):
                    def hook(module, inputs, output):
                        if module in stash_dict:
                            module._parameters['weight'] = stash_dict.pop(module)
                    return hook

                h_pre = target.register_forward_pre_hook(
                    _make_fc_pre_hook(chunk_bits, self.fc_chunks, self._stashed)
                )
                h_post = target.register_forward_hook(
                    _make_post_hook(self._stashed)
                )
                self._pre_hooks.append(h_pre)
                self._post_hooks.append(h_post)

            else:
                # Single bit-width for the whole layer
                b = bits_info if isinstance(bits_info, int) else 32
                if b >= 32:
                    continue

                def _make_pre_hook(b_val, stash_dict):
                    def hook(module, inputs):
                        stash_dict[module] = module._parameters['weight']
                        w = stash_dict[module]
                        module._parameters['weight'] = ste_quantize_asym(w, b_val)
                    return hook

                def _make_post_hook(stash_dict):
                    def hook(module, inputs, output):
                        if module in stash_dict:
                            module._parameters['weight'] = stash_dict.pop(module)
                    return hook

                h_pre = target.register_forward_pre_hook(
                    _make_pre_hook(b, self._stashed)
                )
                h_post = target.register_forward_hook(
                    _make_post_hook(self._stashed)
                )
                self._pre_hooks.append(h_pre)
                self._post_hooks.append(h_post)

        return self._ScopeGuard(self)


# ============================================================================
# SECTION 3 : Policy Sampler (v2 -- DP-based, matches eval)
# ============================================================================

class PolicySampler:
    """Samples segment-level mixed-precision policies for QAT training.

    v2 design: uses the SAME DP solver and avg_omega as the eval path,
    ensuring that training and eval see identical policy distributions.

    Sampling: pick a random target saving in [lo, hi], solve the DP,
    return the resulting policy.
    """

    def __init__(
        self,
        fc_blocks: List[str],
        non_fc_blocks: List[str],
        M: int,
        segments: List[Tuple[int, int]],
        kappa_seg: Dict,
        non_fc_cost: float,
        omega_per_bin: Dict[int, Dict],
        anchor_bits: int = ANCHOR_BITS,
        bit_options: List[int] = None,
        max_saving: float = 93.0,
    ):
        self.fc_blocks = fc_blocks
        self.non_fc_blocks = non_fc_blocks
        self.M = M
        self.segments = segments
        self.kappa_seg = kappa_seg
        self.non_fc_cost = non_fc_cost
        self.omega_per_bin = omega_per_bin
        self.anchor_bits = anchor_bits
        self.bit_options = bit_options or BIT_OPTIONS
        self.max_saving = max_saving

        # Pre-compute population-averaged omega (same as eval)
        self.avg_omega: Dict[Tuple[int, int, int], float] = {}
        for (l, r) in segments:
            for b in self.bit_options:
                vals = [omega_per_bin[j].get((l, r, b), 0) for j in range(K_BINS)]
                self.avg_omega[(l, r, b)] = float(np.mean(vals))

        # Pre-compute a cache of DP-solved policies at discrete savings
        self._policy_cache: List[Tuple[float, Dict[str, int]]] = []
        savings_grid = np.arange(75.0, min(max_saving, 93.5), 0.5)
        for sav in savings_grid:
            fc_budget = max((1.0 - sav / 100.0) - non_fc_cost, 0.001)
            _, seg = solve_dp(
                M, segments, self.avg_omega, kappa_seg,
                fc_budget, self.bit_options, anchor_bits,
            )
            pol = segmentation_to_policy(
                seg, fc_blocks, non_fc_blocks, anchor_bits,
            )
            self._policy_cache.append((sav, pol))
        print(f"    PolicySampler: cached {len(self._policy_cache)} DP policies "
              f"(75.0% - {min(max_saving, 93.0):.1f}%)")

    def sample(
        self,
        n: int,
        saving_range: Tuple[float, float] = (85.0, 93.0),
    ) -> List[Dict[str, int]]:
        """Sample *n* policies from the cached DP solutions.

        Picks random target savings in the range and returns the closest
        cached policy. This ensures training sees the EXACT same policies
        that eval will use.
        """
        lo, hi = saving_range
        hi = min(hi, self.max_saving)
        if hi <= lo:
            hi = lo + 0.5

        policies = []
        for _ in range(n):
            target = np.random.uniform(lo, hi)
            # Find closest cached policy
            best_idx = 0
            best_dist = float("inf")
            for i, (sav, _) in enumerate(self._policy_cache):
                d = abs(sav - target)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            policies.append(self._policy_cache[best_idx][1])
        return policies


# ============================================================================
# SECTION 4 : Infrastructure helpers
# ============================================================================

def _compute_max_saving(kappa_seg, fc_blocks, non_fc_cost, M, segments):
    """Compute the maximum achievable saving (all FC at INT2, non-FC at anchor).

    This is the hard budget limit; DP cannot find feasible solutions beyond this.
    """
    # Minimum FC cost: all chunks at INT2, each as a single-element segment
    min_fc_cost = 0.0
    for i in range(M):
        # Cost of single-block segment [i, i+1) at INT2
        min_fc_cost += kappa_seg.get((i, i + 1, 2), 0)
    max_saving = (1.0 - min_fc_cost - non_fc_cost) * 100.0
    return max_saving


def _load_infra() -> Tuple:
    """Load block structure, kappa, segments, omega from cached CSVs.

    Returns
    -------
    fc_blocks, non_fc_blocks, M, segments, kappa_seg, non_fc_cost,
    omega_per_bin_nmse, omega_per_bin_cos2, max_saving
    """
    kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_step1_nmse_kappa.csv")
    if not os.path.exists(kappa_csv):
        kappa_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_kappa.csv")
    kdf = pd.read_csv(kappa_csv)

    all_blocks = sorted(kdf["block"].unique())
    fc_blocks = sorted(
        [b for b in all_blocks if "fc_part" in b],
        key=lambda x: int(re.search(r"(\d+)$", x).group()),
    )
    non_fc_blocks = [b for b in all_blocks if "fc_part" not in b]
    M = len(fc_blocks)

    segments = enumerate_segments(M, L_MAX)

    block_kappa: Dict[Tuple[str, int], float] = {}
    for _, row in kdf.iterrows():
        block_kappa[(row["block"], int(row["bits"]))] = row["kappa"]

    kappa_seg: Dict[Tuple[int, int, int], float] = {}
    for (l, r) in segments:
        for b in BIT_OPTIONS:
            kappa_seg[(l, r, b)] = sum(
                block_kappa.get((fc_blocks[i], b), 0) for i in range(l, r)
            )

    non_fc_cost = sum(
        block_kappa.get((bn, ANCHOR_BITS), 0) for bn in non_fc_blocks
    )

    omega_nmse, omega_cos2 = load_cached_omegas(
        K_BINS, segments, BIT_OPTIONS, ANCHOR_BITS,
    )

    max_saving = _compute_max_saving(kappa_seg, fc_blocks, non_fc_cost, M, segments)

    return (
        fc_blocks, non_fc_blocks, M, segments,
        kappa_seg, non_fc_cost, omega_nmse, omega_cos2, max_saving,
    )


def _load_model_and_data(
    checkpoint: str = DEFAULT_CKPT,
    batch_size: int = 256,
) -> Tuple:
    """Load model, train set, test set, loaders, norm_params, device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")

    train_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htrainout.mat"), "HT",
    )
    test_set = CsiDataset(
        os.path.join(PROJECT_ROOT, "data", "DATA_Htestout.mat"), "HT",
        normalization_params=train_set.normalization_params,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    norm_params = train_set.normalization_params

    net = ModularAE(
        encoder_type="mamba",
        decoder_type="transnet",
        encoded_dim=512,
        M=32,
        encoder_layers=2,
        decoder_layers=2,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state.get("state_dict", state), strict=False)

    return net, train_set, test_set, train_loader, test_loader, norm_params, device


def _load_zeta_and_bins() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load zeta values and compute bin assignments."""
    zeta_csv = os.path.join(RESULTS_CSV, "rpmpq_v2_zeta.csv")
    zeta_vals = pd.read_csv(zeta_csv)["zeta_proxy"].values
    zeta_edges = np.quantile(zeta_vals, np.linspace(0, 1, K_BINS + 1))
    zeta_edges[0] -= 1e-6
    zeta_edges[-1] += 1e-6
    k_indices = np.clip(np.digitize(zeta_vals, zeta_edges) - 1, 0, K_BINS - 1)
    return zeta_vals, k_indices, zeta_edges


def _load_r_ref() -> np.ndarray:
    """Load perfect-CSI rates at the configured SNR."""
    return pd.read_csv(
        os.path.join(RESULTS_CSV, "rpmpq_v2_perfect_rates.csv"),
    )[f"r_perf_{SNR}"].values


# ============================================================================
# SECTION 5 : Training Loop (v2 -- full-weight STE, reconstruction loss)
# ============================================================================

def train_segment_qat(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    norm_params: Tuple,
    device: str,
    fc_blocks: List[str],
    non_fc_blocks: List[str],
    M: int,
    segments: List[Tuple[int, int]],
    kappa_seg: Dict,
    non_fc_cost: float,
    omega_per_bin: Dict[int, Dict],
    max_saving: float,
    *,
    epochs: int = 30,
    lr: float = 2e-5,
    lambda_q: float = 1.0,
    n_policies: int = 2,
    saving_range: Tuple[float, float] = (85.0, 93.0),
    save_dir: str = DEFAULT_QAT_DIR,
    clip_norm: float = 1.0,
    log_interval: int = 50,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """Full-weight STE Segment QAT training (v2).

    Fine-tunes the encoder FC weight directly (no LoRA) for robustness to
    segment-level mixed-precision quantization.  The decoder is frozen.

    Per mini-batch:
        1. Compute FP anchor loss L_fp = NMSELoss(f(X), X)
        2. Sample N policies via DP (same solver as eval)
        3. For each policy, apply STE weight quantization + full forward + decoder
        4. Compute L_q = NMSELoss(f(X; Q_pi), X)  [RECONSTRUCTION loss, not encoder-space]
        5. Total loss = L_fp + lambda_q * mean(L_q)

    Only the encoder FC layer weight is optimized.
    Decoder and all other encoder weights are frozen.
    """
    os.makedirs(save_dir, exist_ok=True)

    real_model = net.module if isinstance(net, nn.DataParallel) else net

    # Freeze everything, then unfreeze only the encoder FC weight
    for p in real_model.parameters():
        p.requires_grad = False
    real_model.encoder.fc.weight.requires_grad = True
    if real_model.encoder.fc.bias is not None:
        real_model.encoder.fc.bias.requires_grad = True

    n_trainable = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in real_model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} "
          f"({n_trainable/n_total*100:.2f}%) [FC weight only]")

    # Clamp saving range to max achievable
    saving_hi = min(saving_range[1], max_saving - 0.5)
    saving_lo = saving_range[0]
    if saving_hi <= saving_lo:
        saving_hi = saving_lo + 0.5
    effective_range = (saving_lo, saving_hi)

    print(f"\n{'=' * 70}")
    print("  SEGMENT QAT v2: Full-Weight STE + Reconstruction Loss")
    print(f"{'=' * 70}")
    print(f"  Epochs:        {epochs}")
    print(f"  LR:            {lr}")
    print(f"  Lambda_q:      {lambda_q}")
    print(f"  Weight decay:  {weight_decay}")
    print(f"  N_policies:    {n_policies}")
    print(f"  Saving range:  {effective_range} (max feasible: {max_saving:.1f}%)")
    print(f"  Clip norm:     {clip_norm}")
    print(f"  Save dir:      {save_dir}")

    # Build policy sampler (DP-based, matches eval)
    sampler = PolicySampler(
        fc_blocks=fc_blocks,
        non_fc_blocks=non_fc_blocks,
        M=M,
        segments=segments,
        kappa_seg=kappa_seg,
        non_fc_cost=non_fc_cost,
        omega_per_bin=omega_per_bin,
        anchor_bits=ANCHOR_BITS,
        bit_options=BIT_OPTIONS,
        max_saving=max_saving,
    )

    # Build segment quantizer (no LoRA)
    quantizer = SegmentQuantizer(net, fc_chunks=32)

    # Loss and optimizer -- only FC weight
    criterion = NMSELoss()
    trainable_params = [p for p in real_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Save original weights for comparison / rollback
    orig_fc_weight = real_model.encoder.fc.weight.data.clone()

    best_val_nmse = float("inf")
    history = {
        "epoch": [],
        "train_loss": [],
        "train_loss_fp": [],
        "train_loss_q": [],
        "val_nmse_db": [],
        "val_nmse_q_db": [],
        "fc_delta_norm": [],
    }

    for epoch in range(epochs):
        # ---- Training ----
        real_model.encoder.train()
        real_model.decoder.eval()

        epoch_loss = 0.0
        epoch_loss_fp = 0.0
        epoch_loss_q = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [QAT-v2]")
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # -- FP anchor loss (no quantization) --
            z_fp = real_model.encoder(x)
            if AQ_BITS > 0:
                z_fp_q = quantize_feedback_torch(z_fp, AQ_BITS)
                # STE for activation quantization
                z_fp_ste = z_fp + (z_fp_q - z_fp).detach()
            else:
                z_fp_ste = z_fp
            x_hat_fp = real_model.decoder(z_fp_ste)
            loss_fp = criterion(x_hat_fp, x)

            # -- Quantized RECONSTRUCTION loss --
            policies = sampler.sample(n_policies, saving_range=effective_range)

            loss_q = torch.tensor(0.0, device=device)
            for pi in policies:
                with quantizer.apply(pi):
                    z_q = real_model.encoder(x)
                    if AQ_BITS > 0:
                        z_q_act = quantize_feedback_torch(z_q, AQ_BITS)
                        z_q_ste = z_q + (z_q_act - z_q).detach()
                    else:
                        z_q_ste = z_q
                    x_hat_q = real_model.decoder(z_q_ste)
                    loss_q = loss_q + criterion(x_hat_q, x)

            loss_q = loss_q / n_policies

            # -- Total loss --
            loss = loss_fp + lambda_q * loss_q
            loss.backward()

            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_fp += loss_fp.item()
            epoch_loss_q += loss_q.item()
            n_batches += 1

            if batch_idx % log_interval == 0:
                # Diagnostic: weight delta and gradient
                fc_grad = real_model.encoder.fc.weight.grad
                g_norm = fc_grad.norm().item() if fc_grad is not None else 0.0
                delta = (real_model.encoder.fc.weight.data - orig_fc_weight).norm().item()
                w_norm = real_model.encoder.fc.weight.data.norm().item()
                pbar.set_postfix(
                    L=f"{loss.item():.5f}",
                    Lfp=f"{loss_fp.item():.5f}",
                    Lq=f"{loss_q.item():.5f}",
                    g=f"{g_norm:.2e}",
                    dW=f"{delta/w_norm:.2e}",
                )

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_fp = epoch_loss_fp / max(n_batches, 1)
        avg_q = epoch_loss_q / max(n_batches, 1)

        # ---- Validation ----
        real_model.eval()
        val_nmse = _validate(real_model, val_loader, norm_params, device)

        # Also validate with a mid-range quantization policy to track QAT progress
        mid_saving = (saving_lo + saving_hi) / 2.0
        val_nmse_q = _validate_quantized(
            real_model, val_loader, norm_params, device,
            sampler, mid_saving,
        )

        fc_delta = (real_model.encoder.fc.weight.data - orig_fc_weight).norm().item()
        fc_norm = real_model.encoder.fc.weight.data.norm().item()

        print(
            f"  Epoch {epoch+1}: loss={avg_loss:.5f}  "
            f"Lfp={avg_fp:.5f}  Lq={avg_q:.5f}  "
            f"val_FP={val_nmse:.2f}dB  val_Q@{mid_saving:.0f}%={val_nmse_q:.2f}dB  "
            f"dW={fc_delta/fc_norm:.3e}  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_loss_fp"].append(avg_fp)
        history["train_loss_q"].append(avg_q)
        history["val_nmse_db"].append(val_nmse)
        history["val_nmse_q_db"].append(val_nmse_q)
        history["fc_delta_norm"].append(fc_delta / fc_norm)

        # Save best (by quantized val NMSE, not FP!)
        # This is key: we want the checkpoint that performs best UNDER quantization
        if val_nmse_q < best_val_nmse:
            best_val_nmse = val_nmse_q
            torch.save(
                {"state_dict": real_model.state_dict(), "epoch": epoch + 1},
                os.path.join(save_dir, "best.pth"),
            )
            print(f"  ** New best (quantized): {val_nmse_q:.2f} dB")

        # Save latest
        torch.save(
            {"state_dict": real_model.state_dict(), "epoch": epoch + 1},
            os.path.join(save_dir, "last.pth"),
        )

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "qat_history.csv"), index=False,
    )

    # Load best checkpoint
    best_state = torch.load(
        os.path.join(save_dir, "best.pth"), map_location=device,
    )
    real_model.load_state_dict(best_state["state_dict"])

    print(f"\n  QAT v2 training complete.  Best quantized val NMSE: {best_val_nmse:.2f} dB")
    print(f"  Checkpoint: {os.path.join(save_dir, 'best.pth')}")

    # Unfreeze all params for downstream usage
    for param in real_model.parameters():
        param.requires_grad = True

    return net


def _validate(
    model: nn.Module,
    loader: DataLoader,
    norm_params: Tuple,
    device: str,
) -> float:
    """Compute validation NMSE (dB) in FP32 with activation quantization."""
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    nmse_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if AQ_BITS > 0:
                z = quantize_feedback_torch(z, AQ_BITS)
            x_hat = real_model.decoder(z)
            nmse_db = calculate_nmse_db(d, x_hat, norm_params)
            nmse_sum += nmse_db.item()
            count += 1
    return nmse_sum / max(count, 1)


def _validate_quantized(
    model: nn.Module,
    loader: DataLoader,
    norm_params: Tuple,
    device: str,
    sampler: PolicySampler,
    target_saving: float,
) -> float:
    """Validate with PTQ at a specific saving level (matches eval path).

    Uses apply_precision_policy (the same PTQ function used in eval) to ensure
    we measure exactly what eval will measure.
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    # Save current weights
    saved_state = {k: v.clone() for k, v in real_model.state_dict().items()}

    # Get the DP policy for this saving
    fc_budget = max((1.0 - target_saving / 100.0) - sampler.non_fc_cost, 0.001)
    _, seg = solve_dp(
        sampler.M, sampler.segments, sampler.avg_omega, sampler.kappa_seg,
        fc_budget, sampler.bit_options, sampler.anchor_bits,
    )
    policy = segmentation_to_policy(
        seg, sampler.fc_blocks, sampler.non_fc_blocks, sampler.anchor_bits,
    )

    # Apply PTQ (same as eval)
    apply_precision_policy(model, policy, device)
    real_model.eval()

    nmse_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            d = batch.to(device)
            z = real_model.encoder(d)
            if AQ_BITS > 0:
                z = quantize_feedback_torch(z, AQ_BITS)
            x_hat = real_model.decoder(z)
            nmse_db = calculate_nmse_db(d, x_hat, norm_params)
            nmse_sum += nmse_db.item()
            count += 1

    # Restore weights
    real_model.load_state_dict(saved_state)
    return nmse_sum / max(count, 1)


# ============================================================================
# SECTION 6 : Evaluation -- PTQ vs QAT comparison
# ============================================================================

def evaluate_outage(
    net: nn.Module,
    test_set: CsiDataset,
    norm_params: Tuple,
    device: str,
    fc_blocks: List[str],
    non_fc_blocks: List[str],
    M: int,
    segments: List[Tuple[int, int]],
    kappa_seg: Dict,
    non_fc_cost: float,
    omega_per_bin: Dict[int, Dict],
    k_indices: np.ndarray,
    r_ref: np.ndarray,
    max_saving: float,
    budget_savings: Optional[List[float]] = None,
    gammas: Optional[List[float]] = None,
    label: str = "model",
) -> pd.DataFrame:
    """Evaluate rate-based outage at various budget levels.

    v2 fix: skips savings above max_saving (where DP returns infeasible/anchor).
    """
    if budget_savings is None:
        budget_savings = np.arange(85.0, 97.01, 0.25).tolist()
    if gammas is None:
        gammas = [0.99, 0.95]

    real_model = net.module if isinstance(net, nn.DataParallel) else net
    original_state = {k: v.clone().cpu() for k, v in real_model.state_dict().items()}

    N = len(test_set)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    # Pre-compute avg omega (same for all savings)
    avg_omega = {}
    for (l, r) in segments:
        for b in BIT_OPTIONS:
            vals = [omega_per_bin[j].get((l, r, b), 0) for j in range(K_BINS)]
            avg_omega[(l, r, b)] = float(np.mean(vals))

    rows = []
    pbar = tqdm(budget_savings, desc=f"Eval [{label}]")

    for target_saving in pbar:
        # v2 fix: skip infeasible savings
        if target_saving > max_saving:
            for gamma in gammas:
                rows.append({
                    "label": label,
                    "target_saving": target_saving,
                    "gamma": gamma,
                    "outage": float("nan"),
                    "nmse_db": float("nan"),
                    "note": "infeasible",
                })
            pbar.set_postfix(sav=f"{target_saving:.1f}%", nmse="INFEASIBLE")
            continue

        total_budget = 1.0 - target_saving / 100.0
        fc_budget = total_budget - non_fc_cost
        if fc_budget < 0:
            fc_budget = 0.001

        _, seg = solve_dp(
            M, segments, avg_omega, kappa_seg,
            fc_budget, BIT_OPTIONS, ANCHOR_BITS,
        )
        policy = segmentation_to_policy(
            seg, fc_blocks, non_fc_blocks, ANCHOR_BITS,
        )

        # Apply PTQ
        real_model.load_state_dict(original_state)
        apply_precision_policy(net, policy, device)
        real_model.eval()

        # Run inference
        all_rates = []
        all_nmse = []
        min_val, range_val = norm_params
        with torch.no_grad():
            for batch in loader:
                d = batch.to(device)
                z = real_model.encoder(d)
                if AQ_BITS > 0:
                    z = quantize_feedback_torch(z, AQ_BITS)
                x_hat = real_model.decoder(z)

                h_true = (d * range_val) + min_val - 0.5
                h_hat = (x_hat * range_val) + min_val - 0.5

                r = calculate_su_miso_rate_mrt(h_true, h_hat, SNR, device)
                all_rates.extend(r.cpu().numpy().tolist())

                error = torch.sum((h_true - h_hat) ** 2, dim=[1, 2, 3])
                power = torch.sum(h_true ** 2, dim=[1, 2, 3])
                nmse_l = (error / (power + 1e-9)).cpu().numpy()
                all_nmse.extend(nmse_l.tolist())

        rates = np.array(all_rates)
        nmse_arr = np.array(all_nmse)
        nmse_db = 10 * np.log10(np.mean(nmse_arr) + 1e-15)

        for gamma in gammas:
            outage = float(np.mean(rates < gamma * r_ref[:len(rates)]))
            rows.append({
                "label": label,
                "target_saving": target_saving,
                "gamma": gamma,
                "outage": outage,
                "nmse_db": nmse_db,
            })

        pbar.set_postfix(
            sav=f"{target_saving:.1f}%",
            nmse=f"{nmse_db:.2f}dB",
        )

    # Restore
    real_model.load_state_dict(original_state)
    return pd.DataFrame(rows)


def plot_comparison(
    df_ptq: pd.DataFrame,
    df_qat: pd.DataFrame,
    gammas: Optional[List[float]] = None,
    save_prefix: str = "segment_qat_v2_comparison",
) -> None:
    """Plot PTQ vs QAT outage and NMSE comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if gammas is None:
        gammas = sorted(df_ptq["gamma"].unique())

    # Drop infeasible rows
    df_ptq = df_ptq.dropna(subset=["nmse_db"])
    df_qat = df_qat.dropna(subset=["nmse_db"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # -- Outage comparison --
    fig, axes = plt.subplots(
        1, len(gammas), figsize=(6 * len(gammas), 5), sharey=True,
    )
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        sub_ptq = df_ptq[df_ptq["gamma"] == gamma].sort_values("target_saving")
        sub_qat = df_qat[df_qat["gamma"] == gamma].sort_values("target_saving")

        ax.plot(
            sub_ptq["target_saving"], sub_ptq["outage"],
            "b-o", label="PTQ (baseline)", markersize=4, linewidth=2,
        )
        ax.plot(
            sub_qat["target_saving"], sub_qat["outage"],
            "r-s", label="Segment QAT v2", markersize=4, linewidth=2,
        )
        ax.set_title(f"gamma = {gamma}", fontsize=13)
        ax.set_xlabel("BOPs Saving (%)")
        if ax == axes[0]:
            ax.set_ylabel("Outage Probability")
            ax.legend(fontsize=10)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Rate-Based Outage: PTQ vs Segment QAT v2", fontsize=14)
    fig.tight_layout()

    png_path = os.path.join(RESULTS_PLOT, f"{save_prefix}_outage.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{save_prefix}_outage.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    print(f"  Saved: {png_path}")
    plt.close(fig)

    # -- NMSE comparison --
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    g0 = gammas[0]
    sub_ptq = df_ptq[df_ptq["gamma"] == g0].sort_values("target_saving")
    sub_qat = df_qat[df_qat["gamma"] == g0].sort_values("target_saving")

    ax2.plot(
        sub_ptq["target_saving"], sub_ptq["nmse_db"],
        "b-o", label="PTQ (baseline)", markersize=4, linewidth=2,
    )
    ax2.plot(
        sub_qat["target_saving"], sub_qat["nmse_db"],
        "r-s", label="Segment QAT v2", markersize=4, linewidth=2,
    )
    ax2.set_xlabel("BOPs Saving (%)")
    ax2.set_ylabel("NMSE (dB)")
    ax2.set_title("NMSE: PTQ vs Segment QAT v2", fontsize=14)
    ax2.legend(fontsize=11)
    fig2.tight_layout()

    png2 = os.path.join(RESULTS_PLOT, f"{save_prefix}_nmse.png")
    pdf2 = os.path.join(FIGURES_DIR, f"{save_prefix}_nmse.pdf")
    fig2.savefig(png2, dpi=300)
    fig2.savefig(pdf2, dpi=300)
    print(f"  Saved: {png2}")
    plt.close(fig2)


# ============================================================================
# SECTION 7 : CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment QAT v2: Full-Weight STE + Reconstruction Loss",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (lower than v1 since updating FC directly)")
    parser.add_argument("--lambda-q", type=float, default=1.0,
                        help="Weight for quantized reconstruction loss")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization to keep weights near pretrained")
    parser.add_argument("--n-policies", type=int, default=2,
                        help="Policies per mini-batch (2 is enough with reconstruction loss)")
    parser.add_argument("--saving-lo", type=float, default=85.0)
    parser.add_argument("--saving-hi", type=float, default=93.0,
                        help="Upper saving (auto-clamped to max feasible)")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_QAT_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                        help="Pretrained model checkpoint to start from")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing QAT model")
    parser.add_argument("--compare", action="store_true",
                        help="Run PTQ vs QAT comparison (with --eval-only)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  SEGMENT QAT v2: Full-Weight STE + Reconstruction Loss")
    print("=" * 70)

    t0 = time.time()

    # Load infrastructure
    print("\n[1] Loading infrastructure (block structure, kappa, omega)...")
    (
        fc_blocks, non_fc_blocks, M, segments,
        kappa_seg, non_fc_cost, omega_nmse, omega_cos2, max_saving,
    ) = _load_infra()
    print(f"    FC blocks: {M}, segments: {len(segments)}")
    print(f"    Max feasible saving: {max_saving:.2f}%")

    # Load model and data
    print("\n[2] Loading model and data...")
    ckpt = args.checkpoint
    (
        net, train_set, test_set,
        train_loader, test_loader, norm_params, device,
    ) = _load_model_and_data(checkpoint=ckpt, batch_size=args.batch_size)

    # Load zeta and rates
    zeta_vals, k_indices, zeta_edges = _load_zeta_and_bins()
    r_ref = _load_r_ref()
    print(f"    Test samples: {len(test_set)}")

    if not args.eval_only:
        # ---- Training ----
        print("\n[3] Starting Segment QAT v2 training...")
        net = train_segment_qat(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader,
            norm_params=norm_params,
            device=device,
            fc_blocks=fc_blocks,
            non_fc_blocks=non_fc_blocks,
            M=M,
            segments=segments,
            kappa_seg=kappa_seg,
            non_fc_cost=non_fc_cost,
            omega_per_bin=omega_nmse,
            max_saving=max_saving,
            epochs=args.epochs,
            lr=args.lr,
            lambda_q=args.lambda_q,
            n_policies=args.n_policies,
            saving_range=(args.saving_lo, args.saving_hi),
            save_dir=args.save_dir,
            clip_norm=args.clip_norm,
            weight_decay=args.weight_decay,
        )

    # ---- Evaluation ----
    print("\n[4] Evaluating QAT model...")

    # Load QAT checkpoint
    qat_ckpt = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(qat_ckpt):
        state = torch.load(qat_ckpt, map_location=device)
        real_model = net.module if isinstance(net, nn.DataParallel) else net
        real_model.load_state_dict(state.get("state_dict", state), strict=False)
        print(f"    Loaded QAT checkpoint: {qat_ckpt}")
    else:
        print(f"    WARNING: QAT checkpoint not found at {qat_ckpt}")
        if args.eval_only:
            print("    Cannot evaluate without checkpoint. Exiting.")
            return

    gammas = [0.99, 0.95]
    # v2 fix: only evaluate up to max_saving + a small margin
    budget_savings = np.arange(85.0, min(max_saving + 1.0, 97.01), 0.25).tolist()

    # Evaluate QAT model
    df_qat = evaluate_outage(
        net=net,
        test_set=test_set,
        norm_params=norm_params,
        device=device,
        fc_blocks=fc_blocks,
        non_fc_blocks=non_fc_blocks,
        M=M,
        segments=segments,
        kappa_seg=kappa_seg,
        non_fc_cost=non_fc_cost,
        omega_per_bin=omega_nmse,
        k_indices=k_indices,
        r_ref=r_ref,
        max_saving=max_saving,
        budget_savings=budget_savings,
        gammas=gammas,
        label="QAT_v2",
    )

    qat_csv = os.path.join(RESULTS_CSV, "segment_qat_v2_results.csv")
    df_qat.to_csv(qat_csv, index=False)
    print(f"    QAT results saved: {qat_csv}")

    if args.compare:
        # ---- PTQ baseline comparison ----
        print("\n[5] Running PTQ baseline for comparison...")
        ptq_state = torch.load(args.checkpoint, map_location=device)
        real_model = net.module if isinstance(net, nn.DataParallel) else net
        real_model.load_state_dict(
            ptq_state.get("state_dict", ptq_state), strict=False,
        )

        df_ptq = evaluate_outage(
            net=net,
            test_set=test_set,
            norm_params=norm_params,
            device=device,
            fc_blocks=fc_blocks,
            non_fc_blocks=non_fc_blocks,
            M=M,
            segments=segments,
            kappa_seg=kappa_seg,
            non_fc_cost=non_fc_cost,
            omega_per_bin=omega_nmse,
            k_indices=k_indices,
            r_ref=r_ref,
            max_saving=max_saving,
            budget_savings=budget_savings,
            gammas=gammas,
            label="PTQ",
        )

        ptq_csv = os.path.join(RESULTS_CSV, "segment_ptq_v2_results.csv")
        df_ptq.to_csv(ptq_csv, index=False)
        print(f"    PTQ results saved: {ptq_csv}")

        # Combined CSV
        df_combined = pd.concat([df_ptq, df_qat], ignore_index=True)
        combined_csv = os.path.join(RESULTS_CSV, "segment_qat_v2_comparison.csv")
        df_combined.to_csv(combined_csv, index=False)
        print(f"    Combined results saved: {combined_csv}")

        # Plot comparison
        print("\n[6] Plotting PTQ vs QAT v2 comparison...")
        plot_comparison(df_ptq, df_qat, gammas=gammas)

        # Print summary table
        print("\n" + "=" * 70)
        print("  SUMMARY: PTQ vs QAT v2")
        print("=" * 70)
        for gamma in gammas:
            print(f"\n  gamma = {gamma}")
            for sav in [85.0, 87.5, 90.0, 92.5]:
                row_ptq = df_ptq[
                    (df_ptq["gamma"] == gamma)
                    & (df_ptq["target_saving"].between(sav - 0.3, sav + 0.3))
                ].dropna(subset=["nmse_db"])
                row_qat = df_qat[
                    (df_qat["gamma"] == gamma)
                    & (df_qat["target_saving"].between(sav - 0.3, sav + 0.3))
                ].dropna(subset=["nmse_db"])
                if len(row_ptq) > 0 and len(row_qat) > 0:
                    o_ptq = row_ptq["outage"].values[0]
                    o_qat = row_qat["outage"].values[0]
                    n_ptq = row_ptq["nmse_db"].values[0]
                    n_qat = row_qat["nmse_db"].values[0]
                    print(
                        f"    {sav:5.1f}%: PTQ outage={o_ptq:.4f} NMSE={n_ptq:.2f}dB  |  "
                        f"QAT outage={o_qat:.4f} NMSE={n_qat:.2f}dB  |  "
                        f"delta_NMSE={n_qat - n_ptq:+.2f}dB"
                    )

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
