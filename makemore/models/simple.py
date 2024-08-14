import torch
import torch.nn as nn


class Bigram(nn.Module):
    r"""A character-level bigram language model.

    A simple lookup table of logits for the next character
    given the previous character.

    Args:
        vocab_size (int): Size of the vocabulary.

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

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            data=torch.zeros(size=(vocab_size, vocab_size)), requires_grad=True
        )

    @staticmethod
    def get_block_size() -> int:
        """Returns the context length, which is the number of
        characters to consider for predicting the next one."""
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits[x]
        return logits


class MLP(nn.Module):
    r"""A character-level neural probabilistic language model.

    Concatenates the character embeddings of the previous three
    characters and passes the result through an MLP.

    The architecture is based on the paper `A Neural Probabilistic
    Language Model <https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf>`__
    but it operates at the character level instead of the word level.

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
        self, vocab_size: int, block_size: int, emb_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=emb_dim * self.block_size,
                out_features=hidden_dim,
                bias=True,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=vocab_size,
                bias=True,
            ),
        )

    def get_block_size(self) -> int:
        """Returns the context length, which is the number of
        characters to consider for predicting the next one."""
        return self.block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = []
        for k in range(self.block_size):
            tok_emb = self.emb(x)
            embs.append(tok_emb)

            # shift the seq of indices to the right
            x = torch.roll(x, 1, 1)
            x[:, 0] = 0  # special 0 token

        embs.reverse()
        emb_x = torch.cat(embs, -1)
        logits = self.mlp(emb_x)
        return logits
