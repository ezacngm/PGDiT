
"""
Originally inspired by VDT at https://github.com/RERV/VDT

inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# 
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint
class AngularAwareTemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5

        # qkv projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.angular_bias_scale = nn.Parameter(torch.ones(1))
    def forward(self, x, bvecs):
        """
        x:      (B*N, T, D)  — after rearrange: batch×patches, time, channels
        bvecs:  (B,   T, 3)  — original per-frame b-vectors
        """
        BN, T, D = x.shape
        # 1) compute Q,K,V
        qkv = self.qkv(x) \
            .reshape(BN, T, 3, self.num_heads, D // self.num_heads) \
            .permute(2, 0, 3, 1, 4)  # (3, BN, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bvecs = bvecs / (torch.norm(bvecs, dim=-1, keepdim=True) + 1e-6)
        # 2) attention logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (BN, heads, T, T)

        # 3) angular bias
        #    compute once per batch B, then repeat for each of the N patches
        B = bvecs.shape[0]
        # cosine similarity matrix, shape (B, T, T)
        cos_sim = torch.einsum('b i d, b j d -> b i j', bvecs, bvecs).clamp(-1, 1)
        # broadcast to (B, heads, T, T)
        angular_bias = cos_sim.unsqueeze(1)  # (B, 1, T, T)
        # repeat for each of the N patches
        N = BN // B
        angular_bias = angular_bias.repeat_interleave(N, dim=0)  # (BN, 1, T, T)

        # 4) add bias into logits
        attn = (attn_logits + self.angular_bias_scale * angular_bias).softmax(dim=-1)

        # 5) weighted sum
        out = (attn @ v)  # (BN, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(BN, T, D)
        return self.proj(out)
def modulate_framewise(x, shift, scale):
    """
    x     : (B*T, N, D)
    shift : (B*T, D)   # 逐帧 γ/β
    scale : (B*T, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
class DGEFiLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # Main angular network
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden)
        )


        self.layer_scales = nn.Parameter(torch.ones(2))


    def forward(self, bvec):
        out = self.mlp(bvec)
        gamma, beta= out.chunk(2, dim=-1)

        # Apply layer-specific learned scales
        scale = self.layer_scales
        gamma = gamma * scale[0]
        beta = beta * scale[1]

        return gamma, beta
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


import math
import torch
import torch.nn as nn


import math
import torch
import torch.nn as nn

class BvecEmbedder2DSincos(nn.Module):
    """
    用 2D sin–cos 把每个 b-vector (x,y,z) 编码成 (theta,phi) 嵌入，
    并输出形状 (B, T, hidden_size)。

    总体策略：
      - hidden_size 必须能被 4 整除；
      - hidden_size/4 给 theta sin，hidden_size/4 给 theta cos，
        hidden_size/4 给 phi sin，hidden_size/4 给 phi cos。
    """
    def __init__(self, hidden_size):
        super().__init__()
        assert hidden_size % 4 == 0, "hidden_size must be divisible by 4"
        self.hidden_size = hidden_size
        # 每个角度的 sin/cos 各分 hidden_size/4
        self.quarter_dim = hidden_size // 4

    def forward(self, bvecs: torch.Tensor):
        # bvecs: (B, T, 3)
        B, T, _ = bvecs.shape
        x, y, z = bvecs.unbind(-1)  # each: (B, T)

        # 1) 计算 (theta, phi)
        theta = torch.acos(z.clamp(-1+1e-6, 1-1e-6))      # [0, π]
        phi   = torch.atan2(y, x) % (2*math.pi)          # [0, 2π)

        # 2) 准备频率 ω：shape = (quarter_dim,)
        di = torch.arange(self.quarter_dim, device=bvecs.device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (di / self.quarter_dim))

        # 3) 分别做 sin/cos 编码
        #    theta_args: (B, T, quarter_dim)
        theta_args = theta.unsqueeze(-1) * omega
        emb_theta = torch.cat([theta_args.sin(), theta_args.cos()], dim=-1)
        #    emb_theta.shape = (B, T, 2*quarter_dim) == (B, T, hidden_size/2)

        phi_args = phi.unsqueeze(-1) * omega
        emb_phi   = torch.cat([phi_args.sin(), phi_args.cos()], dim=-1)
        #    emb_phi.shape = (B, T, hidden_size/2)

        # 4) 最终拼接： (B, T, hidden_size/2 + hidden_size/2) = (B, T, hidden_size)
        pos_embed = torch.cat([emb_theta, emb_phi], dim=-1)
        return pos_embed

class BvecEmbedder_angle(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size


    def cartesian_to_spherical(self, b_vectors):
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
          b_vectors: numpy array of shape (N, 3) representing (x, y, z).

        Returns:
          theta: numpy array of polar angles (0 to pi), shape (N,).
          phi: numpy array of azimuth angles (0 to 2pi), shape (N,).
        """
        x = b_vectors[:, 0]
        y = b_vectors[:, 1]
        z = b_vectors[:, 2]

        theta = np.arccos(z)  # polar angle: 0 <= theta <= pi
        phi = np.arctan2(y, x)  # azimuth angle: -pi to pi
        phi[phi < 0] += 2 * np.pi  # adjust to [0, 2pi]

        return theta, phi
    def get_1d_sincos_pos_embed_from_values(self,embed_dim, pos):
        """
        Compute 1D sin–cos positional embeddings for a given set of positions.

        Args:
          embed_dim: int, dimension for the embedding (should be even).
          pos: numpy array of positions, shape (N,).

        Returns:
          pos_embed: numpy array of shape (N, embed_dim).
        """
        assert embed_dim % 2 == 0, "Embed dimension must be even."
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # shape: (embed_dim/2,)

        pos = pos.reshape(-1)  # ensure it's a 1D array of shape (N,)
        out = np.outer(pos, omega)  # shape: (N, embed_dim/2)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # shape: (N, embed_dim)
        return pos_embed

    def get_2d_sincos_pos_embed_for_bvec(self,embed_dim, theta, phi):
        """
        Generate a 2D sin–cos embedding for b vectors using their θ and φ values.

        Args:
          embed_dim: int, total embedding dimension (should be even).
          theta: numpy array of θ values, shape (N,).
          phi: numpy array of φ values, shape (N,).

        Returns:
          pos_embed: numpy array of shape (N, embed_dim).
        """
        # Compute embedding for each angle with half of the embed dimension
        emb_theta = self.get_1d_sincos_pos_embed_from_values(embed_dim // 2, theta)
        emb_phi = self.get_1d_sincos_pos_embed_from_values(embed_dim // 2, phi)

        # Concatenate to get the final embedding per b vector
        pos_embed = np.concatenate([emb_theta, emb_phi], axis=1)
        return pos_embed
    def forward(self, bvecs):
        theta,phi = self.cartesian_to_spherical(bvecs.squeeze().cpu().numpy())
        emb = self.get_2d_sincos_pos_embed_for_bvec(self.hidden_size,theta,phi)
        return emb



class BvecEmbedder_sh(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def cart2sph(self,b_vectors):
        """
        Converts Cartesian coordinates to spherical coordinates.
        b_vectors: array of shape (N, 3) where each row is (x, y, z)
        Returns: theta (polar angle), phi (azimuth angle)
        """
        x = b_vectors[:, 0]
        y = b_vectors[:, 1]
        z = b_vectors[:, 2]
        theta = np.arccos(z)  # polar angle: 0 <= theta <= pi
        phi = np.arctan2(y, x)  # azimuth angle: -pi to pi
        phi[phi < 0] += 2 * np.pi  # adjust phi to be in [0, 2*pi]
        return theta, phi

    def compute_spherical_harmonics(self,b_vectors, L_max):
        """
        Computes spherical harmonics for each b vector up to degree L_max.
        b_vectors: array of shape (N, 3), each row is a unit vector.
        L_max: maximum degree of the spherical harmonic expansion.
        Returns: Array of shape (N, total_coeffs) with the spherical harmonic coefficients.
        """
        theta, phi = self.cart2sph(b_vectors)
        N = b_vectors.shape[0]
        # Compute the total number of coefficients: sum(2l+1) for l=0 to L_max
        total_coeffs = sum(2 * l + 1 for l in range(L_max + 1))

        # Initialize an array to hold the coefficients
        Y = np.zeros((N, total_coeffs), dtype=complex)
        idx = 0
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                # sph_harm expects arguments: m, l, phi, theta
                Y[:, idx] = sph_harm_y(m, l, phi, theta)
                idx += 1
        return Y

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16,num_heads_angular=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        
        self.mode = mode
        self.dge_film = DGEFiLM(hidden_size)

        if self.mode == 'video':
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = AngularAwareTemporalAttention(
              hidden_size, num_heads=num_heads_angular, qkv_bias=True)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c,bvec):
        """
        x    : (B*T, N, D)
        c    : (B,   D)  – timestep+label
        bvec : (B,T, 3)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T
        gamma_b, beta_b = self.dge_film(bvec.view(-1, 3))
        rep = lambda t: t.repeat_interleave(T, dim=0)
        shift_msa = rep(shift_msa) + beta_b
        scale_msa = rep(scale_msa) + gamma_b
        shift_mlp = rep(shift_mlp) + beta_b
        scale_mlp = rep(scale_mlp) + gamma_b


        if self.mode == 'video':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_attn(self.temporal_norm1(x),bvecs=bvec)
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + res_temporal

        attn = self.attn(modulate_framewise(self.norm1(x), shift_msa, scale_msa))
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + attn

        mlp = self.mlp(modulate_framewise(self.norm2(x), shift_mlp, scale_mlp))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size,
                                    patch_size * patch_size * out_channels,
                                    bias=True)
        self.adaLN_modulation   = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

        self.dge_film   = DGEFiLM(hidden_size)

    # ------------------------------------------------------------
    def forward(self, x, c, bvec):

        B,T, _ = bvec.shape
        _, _, D = x.shape

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)   # (B, D)
        gamma_b, beta_b = self.dge_film(bvec.view(-1, 3))  # (B*T, D)
        rep = lambda t: t.repeat_interleave(T, dim=0)
        shift = rep(shift) + beta_b  # (B*T, D)
        scale = rep(scale) + gamma_b # (B*T, D)
        x = modulate_framewise(self.norm_final(x), shift, scale)  # (B*T, N, D)
        return self.linear(x)                                     # (B*T, N, p²·C)



class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        learn_sigma=True,
        mode='video',
        num_frames=16,
        num_heads_angular=16,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Use frame-specific label embedding without classifier-free guidance
        self.bvec_embedder = BvecEmbedder2DSincos(hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.mode = mode
        if self.mode == 'video':
            self.num_frames = num_frames
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
            self.time_drop = nn.Dropout(p=0)
        else:
            self.num_frames = 1
        self.checkpoint_every_status = True
        self.checkpoint_every = 2
        self.blocks = nn.ModuleList([
            VDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, mode=mode, num_frames=self.num_frames,num_heads_angular = num_heads_angular) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for blk in self.blocks:
            nn.init.zeros_(blk.dge_film.mlp[-1].weight)
            nn.init.zeros_(blk.dge_film.mlp[-1].bias)
        nn.init.zeros_(self.final_layer.dge_film.mlp[-1].weight)
        nn.init.zeros_(self.final_layer.dge_film.mlp[-1].bias)
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mode == 'video':
            grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
            time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, bvec):
        B, T, C, W, H = x.shape # 1,90,4,40,40
        x = x.contiguous().view(-1, C, W, H)
        y = torch.zeros(B).long().to(x.device)
        x = self.x_embedder(x) + self.pos_embed  # (T, patch_num, D), where patch_num = H * W / patch_size ** 2 (90,100,1152) + (1,100,1152)


        if self.mode == 'video':
            # Angular embed
            bvec_embed = self.bvec_embedder(bvec)  # Shape: (B, T, hidden_size): (1, 90, 512)
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)  # (patch_num, T, D) (400, 90, 512)
            ## Resizing bvec embeddings in case they don't match
            x = x + bvec_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        for idx, block in enumerate(self.blocks):
            if self.checkpoint_every_status and idx % self.checkpoint_every == 0:
                x = checkpoint(block,x, c,bvec)                      # (N, T, D)
            else:
                x = block(x,c,bvec)
        x = self.final_layer(x, c,bvec)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                    Configs                                  #
#################################################################################

def VDT_XL_2(**kwargs):
    return VDT(depth=34, hidden_size=1152, patch_size=2, num_heads=16,num_heads_angular=32, **kwargs)

def VDT_XL_4(**kwargs):
    return VDT(depth=34, hidden_size=1152, patch_size=4, num_heads=16,num_heads_angular=32, **kwargs)

def VDT_XL_8(**kwargs):
    return VDT(depth=34, hidden_size=1152, patch_size=8, num_heads=16,num_heads_angular=32, **kwargs)

def VDT_L_2(**kwargs):
    return VDT(depth=24, hidden_size=1152, patch_size=2, num_heads=16,num_heads_angular=32, **kwargs)
def VDT_L_4(**kwargs):
    return VDT(depth=24, hidden_size=1024, patch_size=3, num_heads=16,num_heads_angular=32, **kwargs)
def VDT_L_8(**kwargs):
    return VDT(depth=24, hidden_size=1024, patch_size=8, num_heads=16,num_heads_angular=32, **kwargs)

def VDT_B_2(**kwargs):
    return VDT(depth=12, hidden_size=768, patch_size=2, num_heads=12,num_heads_angular=16, **kwargs)

def VDT_B_4(**kwargs):
    return VDT(depth=12, hidden_size=768, patch_size=4, num_heads=12,num_heads_angular=16, **kwargs)

def VDT_B_8(**kwargs):
    return VDT(depth=12, hidden_size=768, patch_size=8, num_heads=12,num_heads_angular=16, **kwargs)
def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=512, patch_size=2, num_heads=8,num_heads_angular=8, **kwargs)

def VDT_S_4(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=4, num_heads=6,num_heads_angular=8, **kwargs)

def VDT_S_8(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=8, num_heads=6,num_heads_angular=8, **kwargs)


VDT_models = {
    'VDT-XL/2': VDT_XL_2,  'VDT-XL/4': VDT_XL_4,  'VDT-XL/8': VDT_XL_8,
    'VDT-L/2':  VDT_L_2,   'VDT-L/4':  VDT_L_4,   'VDT-L/8':  VDT_L_8,
    'VDT-B/2':  VDT_B_2,   'VDT-B/4':  VDT_B_4,   'VDT-B/8':  VDT_B_8,
    'VDT-S/2':  VDT_S_2,   'VDT-S/4':  VDT_S_4,   'VDT-S/8':  VDT_S_8,
}