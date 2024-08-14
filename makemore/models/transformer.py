import math

import torch
from torch import nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function (identical to OpenAI GPT).

    The method is described in the paper:
    `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class MaskedMultiHeadAttention(nn.Module):
    r"""Allows the model to jointly attend to information from different
    representations only at preceding positions in an input sequence.

    The method is described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        block_size (int): Maximum length of the sequence.
        d_model (int): Dimensionality of the vectors used throughout the
            transformer.
        n_head (int): Number of self-attention heads.
        bias (bool, optional): If ``True``, adds bias to the input / output
            projection layers. Default: ``False``.
        dropout (float, optional): Dropout probability to be applied on the
            attention weights and also on the output projection layer
            (default=0.2).

    Inputs: x
        * **x**: tensor of shape :math:`(B, T, D)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length} \\
                D ={} & \text{model dimension}
            \end{aligned}

    Outputs: out
        * **out**: tensor of shape :math:`(B, T, T)`
    """

    def __init__(
        self,
        block_size: int,
        d_model: int,
        n_head: int,
        bias: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.h_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.resid_dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            name="mask",
            tensor=torch.tril(
                torch.ones(size=(block_size, block_size)).view(
                    1, 1, block_size, block_size
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and move head dim
        # forward to be the batch dim
        q, k, v = self.qkv_proj(x).split(split_size=self.d_model, dim=-1)
        q = q.view(B, T, self.n_head, self.h_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.h_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.h_dim).transpose(1, 2)  # (B, nh, T, hs)

        # implement scaled dot-product masked self-attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )
        masked_attn_scores = attn_scores.masked_fill(
            mask=(self.mask[:, :, :T, :T] == 0), value=float("-inf")
        )
        masked_attn_weights = F.softmax(masked_attn_scores, dim=-1)
        masked_attn_weights = self.attn_dropout(masked_attn_weights)
        out = torch.matmul(masked_attn_weights, v)

        # re-assemble all head outputs
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        out = self.resid_dropout(self.o_proj(out))
        return out


class FeedForward(nn.Module):
    r"""A point-wise feed forward neural network.

    Args:
        d_model (int): Dimensionality of the vectors used throughout the
            transformer.
        bias (bool, optional): If ``True``, ``Linear`` layers will learn an
            additive bias. Default: ``False``.
        dropout (float, optional): Dropout probability to be applied at the
            end of the feed-forward network (default=0.2).

    Inputs: x
        * **x**: tensor of shape :math:`(B, T, D)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length} \\
                D ={} & \text{model dimension}
            \end{aligned}

    Outputs: out
        * **out**: tensor of shape :math:`(B, T, D)`
    """

    def __init__(
        self, d_model: int, bias: bool = False, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=4 * d_model, bias=bias),
            NewGELU(),
            nn.Linear(in_features=4 * d_model, out_features=d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    r"""Stacks a masked multihead attention layer and a point-wise feed
    forward network together.

    Incorporates normalization layers and residual connections.

    Args:
        block_size (int): Maximum length of the sequence.
        d_model (int): Dimensionality of the vectors used throughout the
            transformer.
        n_head (int): Number of self-attention heads.
        bias (bool, optional): If ``True``, ``Linear`` and ``LayerNorm`` layers
            will learn an additive bias. Default: ``False``.
        dropout (float, optional): Dropout probability to be applied throughout
            the transformer layers (default=0.2).

    Inputs: x
        * **x**: tensor of shape :math:`(B, T, D)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length} \\
                D ={} & \text{model dimension}
            \end{aligned}

    Outputs: out
        * **out**: tensor of shape :math:`(B, T, D)`
    """

    def __init__(
        self,
        block_size: int,
        d_model: int,
        n_head: int,
        bias: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sa = MaskedMultiHeadAttention(
            block_size=block_size,
            d_model=d_model,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
        )
        self.ffd = FeedForward(d_model=d_model, bias=bias, dropout=dropout)
        self.ln1 = nn.LayerNorm(normalized_shape=d_model, bias=bias)
        self.ln2 = nn.LayerNorm(normalized_shape=d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    r"""A transformer decoder as an autoregressive language model.

    The architecture is based on the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Number of characters to consider for predicting
            the next one.
        d_model (int): Dimensionality of the vectors used throughout the
            transformer.
        n_head (int): Number of self-attention heads.
        n_layer (int, optional): The number of sub-decoder layers (default=1).
        bias (bool, optional): If ``True``, ``Linear`` and ``LayerNorm`` layers
            will learn an additive bias. Default: ``False``.
        dropout (float, optional): Dropout probability to be applied throughout
            the transformer layers (default=0.2).

    Inputs: x
        * **x**: tensor of shape :math:`(B, T)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length}
            \end{aligned}

    Outputs: logits
        * **logits**: tensor of shape :math:`(B, T, V)`

        where:

        .. math::
            \begin{aligned}
                V ={} & \text{vocabulary size} \\
            \end{aligned}
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_head: int,
        n_layer: int = 1,
        bias: bool = False,
        dropout: float = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model),
                wpe=nn.Embedding(num_embeddings=block_size, embedding_dim=d_model),
                blocks=nn.ModuleList(
                    DecoderBlock(block_size, d_model, n_head, bias, dropout)
                    for _ in range(n_layer)
                ),
                ln_f=nn.LayerNorm(normalized_shape=d_model),
            )
        )
        self.lm_head = nn.Linear(in_features=d_model, out_features=vocab_size)

    def get_block_size(self) -> int:
        """Returns the context length, which is the number of
        characters considered for predicting the next one."""
        return self.block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.block_size}"
        )
        pos = torch.arange(
            start=0, end=T, dtype=torch.long, device=self.device
        ).unsqueeze(dim=0)

        # forward to the GPT model
        tok_emb = self.transformer.wte(x)  # (B, T, d_model)
        pos_emb = self.transformer.wpe(pos)  # (1, T, d_model)
        x = tok_emb + pos_emb
        for block in self.transformer.blocks:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        return logits
