from typing import List


class CharVocabulary:
    """A simple mapping of each character in the vocabulary
    to a corresponding integer, and vice versa."""

    def __init__(self, chars: List[str]) -> None:
        self.chars = chars
        self.char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def __len__(self) -> int:
        # all the possible characters plus the special 0 token
        return len(self.chars) + 1

    def __str__(self) -> str:
        return f"<Vocabulary(size={len(self)})>"
