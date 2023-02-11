import math
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn
from copy import deepcopy


class InputEmbedding(nn.Module):

    def __init__(self, dim, channels, patch_size, image_h, image_w, num_frames=1):
        super(InputEmbedding, self).__init__()
        self.num_patches = (image_h // patch_size) * (image_w // patch_size)
        num_positions = num_frames * self.num_patches
        patch_dim = channels * (patch_size ** 2)

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions + 1, dim)

    def forward(self, frame, device="cuda"):
        p = self.patch_size
        frame = rearrange(frame, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        frame = self.patch_embedding(frame)
        pos_embedding = self.pos_emb(torch.arange(frame.shape[1], device=device))
        tokens = frame + pos_embedding
        return tokens


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult * 2)
        self.activate_f = nn.functional.glu
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * mult, dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate_f(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, head_dim=64, num_heads=8, dropout=0.):
        """
        :param dim: model input size
        :param head_dim: attention head size
        :param num_heads: num of head
        :param dropout: dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.head_dim * self.num_heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads, self.head_dim
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        # print(q.size(), k.size(), v.size())
        # scaled dot-product
        attention_score = torch.matmul(q, k.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_dim)
        # print(attention_score.size())
        attention_prob = self.attention_dropout(self.softmax(attention_score))
        context_layer = torch.matmul(attention_prob, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.inner_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # print(context_layer.size())
        out = self.to_out(context_layer)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, head_dim=64, num_head=8):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.ffn = FeedForward(hidden_dim)
        self.attn = Attention(hidden_dim, head_dim=head_dim, num_heads=num_head)

    def forward(self, x):
        residual = self.attention_norm(x)
        residual = self.attn(residual)
        x = x + residual

        residual = self.ffn_norm(x)
        residual = self.ffn(residual)
        x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_dim, head_dim=64, num_head=8, N=6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        for i in range(N):
            layer = deepcopy(TransformerBlock(hidden_dim, head_dim, num_head))
            self.layers.append(layer)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        x = self.encoder_norm(x)
        return x


class Transformer(nn.Module):

    def __init__(self, channels, image_h, image_w, hidden_dim, head_dim, num_head):
        super(Transformer, self).__init__()
        self.patch_size = 16
        self.embedding = InputEmbedding(hidden_dim, channels, self.patch_size, image_h, image_w)
        self.encoder = Encoder(hidden_dim, head_dim, num_head)
        self.image_h = image_h
        self.image_w = image_w

    def forward(self, x, device):
        embedding_feat = self.embedding(x, device)
        feat = self.encoder(embedding_feat)
        return feat
