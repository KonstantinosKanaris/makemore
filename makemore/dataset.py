from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from makemore.vectorizer import CharTokenizer


class CharDataset(Dataset):
    def __init__(
        self,
        words: List[str],
        vectorizer: CharTokenizer,
    ) -> None:
        self.words = words
        self.vectorizer = vectorizer
        self.max_word_length = vectorizer.max_word_length

    def __len__(self) -> int:
        return len(self.words)

    def get_output_length(self) -> int:
        # plus 1 for the <START> token
        return self.max_word_length + 1

    def contains(self, word):
        return word in self.words

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        x, y = self.vectorizer.encode(word=word)
        return x, y
