import torch
import torch.nn as nn
import torch.nn.functional as F

from mixed_res.patch_scorers.feature_based_patch_scorer import FeatureBasedPatchScorer
from mixed_res.patch_scorers.pixel_blur_patch_scorer import PixelBlurPatchScorer
from mixed_res.patch_scorers.random_patch_scorer import RandomPatchScorer

from mixed_res.quadtree_impl.quadtree_z_curve import ZCurveQuadtreeRunner
from mixed_res.tokenization.patch_embed import FlatPatchEmbed, PatchEmbed
from mixed_res.tokenization.tokenizers import QuadtreeTokenizer, VanillaTokenizer

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      # TODO

        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(self.in_channels, 
                            self.embed_dim, 
                            kernel_size=self.patch_size, 
                            stride=self.patch_size
                           )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, useFlash):
        super().__init__()
        
        self.useFlash = useFlash
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        
        self.qk_norm = False
        self.use_activation = False
        
        self.activation = nn.ReLU() if self.use_activation else nn.Identity()
        
        self.q_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        
        self.attn_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        batch_si, seq_len, emb_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_si, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        
        if self.useFlash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
                z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        else:
            attention = q @ k.transpose(-2, -1)
            attention = attention.softmax(dim=-1)
            attention = self.attn_dropout(attention)
            z = attention @ v
            
        z = z.transpose(1, 2).reshape(batch_si, seq_len, emb_dim)
        z = self.proj(z)
        return z


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, useFlash):

        super().__init__()
        
        self.attention = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadSelfAttention(embed_dim, num_heads, useFlash),
        )
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        res = x
        x = self.attention(x)
        x = x + res # residual connection
        res = x
        x = self.mlp(x)
        x = x + res
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, 
        num_classes, dropout=0.1, useMRT=False, max_patch_size=64, quadtree_num_patches=64, useFlash=False
    ):
        super().__init__()
        self.useMRT = useMRT
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if useMRT:
            self.patch_embed = FlatPatchEmbed(img_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
            self.quadtree_runner = ZCurveQuadtreeRunner(quadtree_num_patches, patch_size, max_patch_size)
            self.patch_scorer = FeatureBasedPatchScorer()
            self.quadtree_tokenizer = QuadtreeTokenizer(
                self.patch_embed, self.cls_token, self.quadtree_runner, self.patch_scorer
            )
        
        self.embed_len = self.patch_embed.num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer1 = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, useFlash) for i in range(3)
        ])
        
        self.transformer2 = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, useFlash) for i in range(3, num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
            # nn.Dropout(dropout)
        )                           

    def forward(self, x):
        
        if self.useMRT:
            x = self.quadtree_tokenizer.tokenize(x)
        else:
            x = self.patch_embed(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        
        # x = self.dropout(x)
        # x = self.norm(x)
        A = self.transformer1(x)
        x = self.transformer2(A)
        x = self.cls_head(x[:, 0])

        return x, A
