from typing import List, Tuple

import torch

from makemore.vocabulary import CharVocabulary


class CharTokenizer:
    def __init__(self, vocab: CharVocabulary, max_word_length: int) -> None:
        self.vocab = vocab
        self.max_word_length = max_word_length

        self.char2idx = vocab.char2idx
        self.idx2char = vocab.idx2char

    def encode(self, word: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorizes the input word into two 1-D tensors of integers.

        The first tensor, `x`, is the input, zero-padded tensor to the
        model. The second tensor, `y`, is the target tensor, padded with
        `-1` to mask the loss at inactive locations.

        Args:
            word (str): The input word.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input and target tensors
                to the model.

        Example::

            If the input word is `dio` then:

            x = tensor([0,4,9,15,0,0,0,0,0,0,0,0,0,0,0,0])
            y = tensor([4,9,15,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        """
        indices = torch.tensor(
            data=[self.char2idx[char] for char in word], dtype=torch.long
        )
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1 : 1 + len(indices)] = indices
        y[: len(indices)] = indices
        y[len(indices) + 1 :] = -1
        return x, y

    def decode(self, indices: List[int]) -> str:
        """Converts a list of indices to a human-readable format."""
        return "".join(self.idx2char[idx] for idx in indices)
