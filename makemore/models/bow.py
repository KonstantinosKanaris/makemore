import torch
import torch.nn as nn
from torch.nn import functional as F


class CasualBoW(nn.Module):
    r"""Casual bag of words.

    Averages the vectors of the preceding elements.

    Args:
        block_size (int): Number of characters to consider for predicting
            the next one.

    Inputs: x
        * **x**: tensor of shape :math:`(B, T, C)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length} \\
                C ={} & \text{embedding dimension}
            \end{aligned}

    Outputs: out
        * **out**: tensor of shape :math:`(B, T, C)`
    """

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.register_buffer(
            name="bias",
            tensor=torch.tril(torch.ones(size=(block_size, block_size))).view(
                1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        att = torch.zeros(size=(B, T, T))
        att = att.masked_fill(mask=self.bias[:, :T, :T] == 0, value=float("-inf"))
        att = F.softmax(input=att, dim=-1)
        out = att @ x  # (B, T, T) x (B, T, C) -> (B, T, C)
        return out


class BoWBlock(nn.Module):
    r"""Collects the BoW features and passes them through an MLP.

    Args:
        block_size (int): Number of characters to consider for predicting
            the next one.
        emb_dim (int): Size of each embedding vector.
        hidden_dim (int): Size of the hidden layers.

    Inputs: x
        * **x**: tensor of shape :math:`(B, T, C)`

        where:

        .. math::
            \begin{aligned}
                B ={} & \text{batch size} \\
                T ={} & \text{max sequence length} \\
                C ={} & \text{embedding dimension}
            \end{aligned}

    Outputs: out
        * **out**: tensor of shape :math:`(B, T, C)`
    """

    def __init__(self, block_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()

        # Casual BoW module
        self.cbow = CasualBoW(block_size=block_size)

        # MLP assembler
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(emb_dim, hidden_dim),
                c_proj=nn.Linear(hidden_dim, emb_dim),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cbow(x)
        out = x + self.mlpf(x)
        return out


class BoW(nn.Module):
    r"""A character-level bag of words model.

    Takes the previous `:attr:block_size` characters, encodes them
    and their position using lookup tables, and then uses the average
    of these encoded characters to predict the next one.

    Args:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Number of characters to consider for predicting
            the next one.
        emb_dim (int): Size of each embedding vector.
        hidden_dim (int): Size of the hidden layers.

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
        emb_dim: int,
        hidden_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device = device

        # token embedding table
        self.wte = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        # position embedding table
        self.wpe = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=emb_dim
        )
        # context block
        self.context_block = BoWBlock(
            block_size=block_size, emb_dim=emb_dim, hidden_dim=hidden_dim
        )
        # language model head decoder layer
        self.lm_head = nn.Linear(in_features=emb_dim, out_features=vocab_size)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.block_size}"
        )

        pos = torch.arange(
            start=0, end=T, dtype=torch.long, device=self.device
        ).unsqueeze(dim=0)
        tok_emb = self.wte(x)  # (B, T, emb_dim)
        pos_emb = self.wpe(pos)  # (1, T, emb_dim)
        emb_x = tok_emb + pos_emb
        emb_x = self.context_block(emb_x)
        logits = self.lm_head(emb_x)
        return logits
