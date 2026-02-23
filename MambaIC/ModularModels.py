import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
from typing import Optional, Tuple, Any, List
import math
import copy

__all__ = ["ModularAE"]

Tensor = torch.Tensor

# =========================================================================
# [Part 0] Helper Functions & Quantization
# =========================================================================
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# ✅ 양자화 함수 (Straight-Through Estimator)
class QuantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        # 입력의 Min/Max를 이용해 Scale 계산
        min_val, max_val = input.min(), input.max()
        step = (max_val - min_val) / (2 ** bits - 1)
        if step == 0: return input # 값 변화가 없으면 그대로 반환

        # 양자화 (Round)
        input_clamped = torch.clamp(input, min_val, max_val)
        input_normalized = (input_clamped - min_val) / step
        input_rounded = torch.round(input_normalized)
        
        # 역양자화 (De-quantization)하여 값 복원 (Forward는 양자화된 값 사용)
        output = input_rounded * step + min_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파 시에는 기울기를 그대로 통과 (STE)
        return grad_output, None

def quantize_tensor(x, bits):
    """텐서(Weight or Activation)를 지정된 비트로 양자화"""
    return QuantizationFunction.apply(x, bits)

# =========================================================================
# [Part 1] Transformer Components (Attention)
# =========================================================================
def scale_dot_attention(q, k, v, dropout_p=0.0, attn_mask=None):
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None: attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p: attn = F.dropout(attn, p=dropout_p)
    return torch.bmm(attn, v), attn

def multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None):
    tgt_len, bsz, embed_dim = query.shape; head_dim = embed_dim // num_heads
    if use_separate_proj_weight:
        q = F.linear(query, q_proj_weight, in_proj_bias[:embed_dim] if in_proj_bias is not None else None)
        k = F.linear(key, k_proj_weight, in_proj_bias[embed_dim:2*embed_dim] if in_proj_bias is not None else None)
        v = F.linear(value, v_proj_weight, in_proj_bias[2*embed_dim:] if in_proj_bias is not None else None)
    else: q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if not training: dropout_p = 0.0
    attn_output, attn_output_weights = scale_dot_attention(q, k, v, dropout_p, attn_mask)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return attn_output, None

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.dropout = dropout; self.batch_first = batch_first; self.head_dim = embed_dim // num_heads
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias: self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else: self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None: constant_(self.in_proj_bias, 0.); constant_(self.out_proj.bias, 0.)
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first: query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        attn_output, _ = multi_head_attention_forward(query, key, value, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)
        if self.batch_first: return attn_output.transpose(1, 0), None
        return attn_output, None

# =========================================================================
# [Part 2] Transformer Layers (Encoder/Decoder)
# =========================================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward); self.dropout = nn.Dropout(dropout); self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps); self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.activation = activation
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2); src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))); src = src + self.dropout(src2); src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward); self.dropout = nn.Dropout(dropout); self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps); self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps); self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout); self.activation = activation
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2); tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2); tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))); tgt = tgt + self.dropout3(tgt2); tgt = self.norm3(tgt)
        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(); self.layers = _get_clones(encoder_layer, num_layers); self.num_layers = num_layers; self.norm = norm
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers: output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None: output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(); self.layers = _get_clones(decoder_layer, num_layers); self.num_layers = num_layers; self.norm = norm
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers: output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None: output = self.norm(output)
        return output

# =========================================================================
# [Part 3] MobileNet Components
# =========================================================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels, bias=False), nn.Hardsigmoid(inplace=True))
    def forward(self, x):
        b, c, _, _ = x.size(); y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor):
        super().__init__(); self.use_res_connect = (stride == 1 and in_ch == out_ch); hidden_dim = int(in_ch * expansion_factor); layers = []
        if expansion_factor != 1: layers.extend([nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.Hardswish(inplace=True)])
        layers.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.Hardswish(inplace=True), SEBlock(hidden_dim)])
        layers.extend([nn.Conv2d(hidden_dim, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]); self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_res_connect else self.conv(x)

# =========================================================================
# [Part 4] Encoders
# =========================================================================

# 1. MobileNet Encoder (Standard)
class MobileNetEncoder(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.Hardswish(inplace=True))
        self.blocks = nn.Sequential(MobileNetBlock(16, 16, 3, 1, 1), MobileNetBlock(16, 24, 3, 2, 4), MobileNetBlock(24, 24, 3, 1, 3), MobileNetBlock(24, 40, 5, 2, 3), MobileNetBlock(40, 40, 5, 1, 3), MobileNetBlock(40, 80, 5, 2, 3))
        self.conv2 = nn.Sequential(nn.Conv2d(80, 128, 1, bias=False), nn.BatchNorm2d(128), nn.Hardswish(inplace=True))
        self.fc = nn.Linear(128 * 4 * 4, encoded_dim)
    def forward(self, x): x = self.conv1(x); x = self.blocks(x); x = self.conv2(x); x = x.flatten(1); return self.fc(x)

# 2. CsiNet Encoder
class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, 1, 1); self.bn = nn.BatchNorm2d(2); self.act = nn.LeakyReLU(0.3); self.flatten_dim = 2 * 32 * 32; self.fc = nn.Linear(self.flatten_dim, encoded_dim)
    def forward(self, x): return self.fc(self.act(self.bn(self.conv(x))).view(-1, self.flatten_dim))

# 3. TransNet Encoder
class TransNetEncoder(nn.Module):
    def __init__(self, encoded_dim=512, d_model=64, num_layers=2):
        super().__init__(); self.d_model = d_model; self.feature_shape = (2048//d_model, d_model) 
        encoder_layer = TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=2048, dropout=0.1)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_encoder = nn.Linear(2048, encoded_dim)
    def forward(self, x): src = x.view(-1, self.feature_shape[0], self.feature_shape[1]); memory = self.encoder(src); return self.fc_encoder(memory.view(memory.shape[0], -1))

# 4. ✅ Mamba Encoder (양자화 지원 수정됨)
class MambaEncoder(nn.Module):
    def __init__(self, encoded_dim=512, M=32, num_layers=2, quant_act_bits=0, quant_param_bits=0):
        super(MambaEncoder, self).__init__()
        
        # Mamba Load
        try: from models.MambaAE import ChunkedResidualMambaBlock
        except ImportError:
            try: from MambaAE import ChunkedResidualMambaBlock
            except ImportError: raise ImportError("MambaAE.py not found.")
            
        self.M = M
        self.H_proc = 16
        self.W_proc = 16
        
        # 양자화 설정
        self.q_act_bits = quant_act_bits
        self.q_param_bits = quant_param_bits
        
        if self.q_act_bits > 0: print(f"🔹 [Mamba] Activation Quantization: {self.q_act_bits} bits")
        if self.q_param_bits > 0: print(f"🔹 [Mamba] Parameter Quantization: {self.q_param_bits} bits")
        
        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(2, M, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(M),
            nn.SiLU()
        )
        
        # 2. Layers
        self.layers = nn.Sequential(*[
            ChunkedResidualMambaBlock(M, 0.0) for _ in range(num_layers)
        ])
        
        # 3. Projection & FC
        self.proj_conv = nn.Conv2d(M, 8, kernel_size=1, bias=False) 
        self.fc = nn.Linear(8 * self.H_proc * self.W_proc, encoded_dim)

    def _apply_weight_quant(self, x, layer):
        """레이어 가중치를 즉석에서 양자화하여 연산 수행"""
        if self.q_param_bits > 0 and hasattr(layer, 'weight'):
            w_q = quantize_tensor(layer.weight, self.q_param_bits)
            if isinstance(layer, nn.Conv2d):
                return F.conv2d(x, w_q, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
            elif isinstance(layer, nn.Linear):
                return F.linear(x, w_q, layer.bias)
        else:
            return layer(x)

    def forward(self, x):
        # 1. Stem (Conv Parameter Quantization)
        if self.q_param_bits > 0:
            x = self._apply_weight_quant(x, self.stem[0]) # Conv
            x = self.stem[1](x) # BN
            x = self.stem[2](x) # SiLU
        else:
            x = self.stem(x)

        # 2. Mamba Layers (내부 Weight는 그대로 사용)
        x = self.layers(x)
        
        # 3. Proj & FC (Parameter Quantization)
        if self.q_param_bits > 0:
            x = self._apply_weight_quant(x, self.proj_conv)
            x = x.flatten(1)
            x = self._apply_weight_quant(x, self.fc)
        else:
            x = self.proj_conv(x)
            x = x.flatten(1)
            x = self.fc(x)

        # 4. Activation Quantization (Latent Vector)
        if self.q_act_bits > 0:
            x = torch.sigmoid(x)  # 0~1 범위로 매핑
            x = quantize_tensor(x, self.q_act_bits)
            
        return x

# =========================================================================
# [Part 5] Decoders
# =========================================================================

# 1. CsiNet Decoder
class CsiNetRefineBlock(nn.Module):
    def __init__(self): super().__init__(); self.seq = nn.Sequential(nn.Conv2d(2, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.LeakyReLU(0.3), nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3), nn.Conv2d(16, 2, 3, 1, 1), nn.BatchNorm2d(2)); self.act = nn.LeakyReLU(0.3)
    def forward(self, x): return self.act(x + self.seq(x))

class CsiNetDecoder(nn.Module):
    def __init__(self, encoded_dim=512, num_layers=2): 
        super().__init__()
        self.flatten_dim = 2 * 32 * 32; self.fc_dec = nn.Linear(encoded_dim, self.flatten_dim)
        self.refine = nn.Sequential(*[CsiNetRefineBlock() for _ in range(num_layers)])
        self.final_conv = nn.Conv2d(2, 2, 3, 1, 1); self.sigmoid = nn.Sigmoid()
    def forward(self, z): x = self.fc_dec(z).view(-1, 2, 32, 32); x = self.refine(x); x = self.final_conv(x); return self.sigmoid(x)

# 2. TransNet Decoder
class TransNetDecoder(nn.Module):
    def __init__(self, encoded_dim=512, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model; self.feature_shape = (2048//d_model, d_model); self.fc_decoder = nn.Linear(encoded_dim, 2048)
        decoder_layer = TransformerDecoderLayer(d_model, nhead=2, dim_feedforward=2048, dropout=0.1)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
    def forward(self, z): memory_decoder = self.fc_decoder(z).view(-1, self.feature_shape[0], self.feature_shape[1]); output = self.decoder(memory_decoder, memory_decoder); return output.view(-1, 2, 32, 32)

# 3. Mamba Decoder
class MambaTransDecoder(nn.Module):
    def __init__(self, encoded_dim=512, M=32, num_layers=2):
        super(MambaTransDecoder, self).__init__()
        try: from models.MambaAE import ChunkedResidualMambaBlock
        except ImportError:
            try: from MambaAE import ChunkedResidualMambaBlock
            except ImportError: raise ImportError("MambaAE.py not found.")
        
        self.M = M
        self.H_proc = 16; self.W_proc = 16
        
        self.fc_expand = nn.Linear(encoded_dim, 2 * self.H_proc * self.W_proc)
        self.input_conv = nn.Sequential(nn.Conv2d(2, M, kernel_size=3, padding=1), nn.BatchNorm2d(M), nn.SiLU())
        self.layers = nn.Sequential(*[ChunkedResidualMambaBlock(M, 0.0) for _ in range(num_layers)])
        self.up_conv = nn.Conv2d(M, 8, kernel_size=3, padding=1) 
        self.pixel_shuffle = nn.PixelShuffle(2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc_expand(z).view(-1, 2, self.H_proc, self.W_proc)
        x = self.input_conv(x)
        x = self.layers(x)
        x = self.up_conv(x)
        x = self.pixel_shuffle(x)
        return self.sigmoid(x)

# 4. MobileNet Decoder
class MobileNetDecoder(nn.Module):
    def __init__(self, encoded_dim=512):
        super().__init__()
        self.fc_bridge = nn.Linear(encoded_dim, 128 * 4 * 4)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), MobileNetBlock(128, 64, 3, 1, 3)) 
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), MobileNetBlock(64, 32, 3, 1, 3))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), MobileNetBlock(32, 16, 3, 1, 3))
        self.final_conv = nn.Conv2d(16, 2, 3, 1, 1); self.sigmoid = nn.Sigmoid()
    def forward(self, z): x = self.fc_bridge(z).view(-1, 128, 4, 4); x = self.up1(x); x = self.up2(x); x = self.up3(x); return self.sigmoid(self.final_conv(x))

# =========================================================================
# [Part 6] Main Assembler
# =========================================================================
class ModularAE(nn.Module):
    def __init__(self, encoder_type="csinet", decoder_type="csinet", encoded_dim=512, M=32, decoder_layers=2, encoder_layers=2, quant_act_bits=0, quant_param_bits=0, **kwargs):
        super().__init__()
        print(f"🏗️ Building: Enc[{encoder_type}-L{encoder_layers}] + Dec[{decoder_type}-L{decoder_layers}]")
        
        # [Encoder]
        if encoder_type == "mamba": 
            # Mamba는 내부에서 직접 처리
            self.encoder = MambaEncoder(encoded_dim, M=M, num_layers=encoder_layers, quant_act_bits=quant_act_bits, quant_param_bits=quant_param_bits)
        elif encoder_type == "csinet": self.encoder = CsiNetEncoder(encoded_dim)
        elif encoder_type == "transnet": self.encoder = TransNetEncoder(encoded_dim, d_model=64, num_layers=encoder_layers)
        elif encoder_type == "mobilenet": self.encoder = MobileNetEncoder(encoded_dim)
        else: raise ValueError(f"Unknown encoder: {encoder_type}")
            
        # [Decoder]
        if decoder_type == "csinet": self.decoder = CsiNetDecoder(encoded_dim, num_layers=decoder_layers)
        elif decoder_type == "transnet": self.decoder = TransNetDecoder(encoded_dim, d_model=64, num_layers=decoder_layers)
        elif decoder_type == "mamba": self.decoder = MambaTransDecoder(encoded_dim, M=M, num_layers=decoder_layers)
        elif decoder_type == "mobilenet": self.decoder = MobileNetDecoder(encoded_dim)
        else: raise ValueError(f"Unknown decoder: {decoder_type}")

    def forward(self, x): return self.decoder(self.encoder(x))