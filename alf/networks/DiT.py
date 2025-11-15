import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention, Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor with shift and scale factors.
    
    :param x: Input tensor.
    :param shift: Shift tensor.
    :param scale: Scale tensor.
    :return: Modulated tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Initialize the TimestepEmbedder module.
        
        :param hidden_size: Size of the hidden layer.
        :param frequency_embedding_size: Size of the frequency embeddings.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        :param t: A 1-D tensor of N indices, one per batch element. These may be fractional.
        :param dim: The dimension of the output.
        :param max_period: Controls the minimum frequency of the embeddings.
        :return: An (N, D) tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the timestep embeddings.
        
        :param t: A tensor of timesteps.
        :return: A tensor of timestep embeddings.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden, act_layer=approx_gelu, drop=0)

        # One small head that maps conditioning vector c (B,H) -> 6*H mods
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # DiT-Zero style: zero init so block starts as identity
        nn.init.constant_(self.adaLN_mod[-1].weight, 0)
        nn.init.constant_(self.adaLN_mod[-1].bias,   0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, H), c: (B, H)
        """
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_mod(c).chunk(6, dim=1)

        # MSA branch
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MLP branch
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, out_channels: int):
        """
        Initialize the FinalLayer module.
        
        :param hidden_size: Size of the hidden layer.
        :param out_channels: Number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the final layer.
        
        :param x: Input tensor.
        :param c: Conditioning tensor.
        :return: Output tensor.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x
    

class DiT(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_size: int = 512,
                 depth: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 num_frames: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames  = num_frames

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = Mlp(
            in_features=in_channels * num_frames,
            hidden_features=512,
            out_features=hidden_size,
            act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, in_channels * num_frames)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Keep final layer zeroed like you had
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.proj[-1].weight,             0)
        nn.init.constant_(self.final_layer.proj[-1].bias,               0)

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, 1, in_channels * num_frames)
        cond_feat: (B, D)   # arbitrary D
        t:         (B,)
        """
        assert x.dim() == 3 and x.shape[1] == 1
        x = self.x_embedder(x.float())                     # (B, 1, H)

        t_emb  = self.t_embedder(t)                        # (B, H)
        c = t_emb + cond_feat                              # (B, H)

        for block in self.blocks:
            x = block(x, c)                                # (B, 1, H)

        x = self.final_layer(x, c)                         # (B, 1, out_channels)
        return x