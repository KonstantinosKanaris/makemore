import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from makemore import logger
from makemore.vectorizer import CharTokenizer


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    indices: torch.Tensor,
    max_seq_length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generates a number of sequences of indices.

    Each sequence of indices is completed by feeding
    the previous sequence of indices into the model to
    predict the next index until the sequence reaches
    the `max_seq_length`.

    Args:
        model (torch.nn.Module): The neural network model.
        indices (torch.tensor): The initial sequence of indices
            where each sequence is consisting of only one index.
        max_seq_length (int): The maximum sequence length.
        temperature (float, optional): Controls the randomness of
            the model's output.

    Returns:
        torch.tensor:
            The completed sequences of indices.
    """
    block_size = model.get_block_size()
    for _ in range(max_seq_length):
        # if the sequence context is growing too long we must
        # crop it at block_size
        idx_cond = (
            indices if indices.size(1) <= block_size else indices[:, -block_size:]
        )
        # Forward pass (returns logits)
        y_pred = model(idx_cond)
        # Focus only on the last time step because those are the predictions
        # of what coming next (last element in the time dimension)
        y_pred = y_pred[:, -1, :] / temperature
        probs = F.softmax(y_pred, dim=-1)
        # sample from the distribution
        next_idx = torch.multinomial(probs, num_samples=1)
        # concatenate the indices
        indices = torch.cat((indices, next_idx), dim=1)

    return indices


def print_samples(
    model: torch.nn.Module,
    vectorizer: CharTokenizer,
    device: str = "cpu",
    num: int = 10,
) -> None:
    """Samples from the model and pretty prints the decoded samples."""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(device)
    steps = vectorizer.max_word_length
    X_samp = generate(model, X_init, steps).to(device)
    word_samples = []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        # note: we need to crop out the first <START> token
        row = X_samp[i, 1:].tolist()
        # token 0 is the <STOP> token, so we crop the output
        # sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_sample = vectorizer.decode(row)
        word_samples.append(word_sample)
    print("-" * 80)
    word_samples = [word for word in word_samples if word]
    for word in word_samples:
        print(word)
    print("-" * 80)


def load_from_txt(input_file: str) -> List[str]:
    with open(file=input_file, mode="r", encoding="utf-8") as f:
        txt_data = f.read().splitlines()
    lines = [line.strip() for line in txt_data if line]
    return lines


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for torch operations.

    Args:
      seed (int, optional): Random seed to set (default=42).
    """
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)


def load_general_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> Dict[str, Any]:
    """Loads a general checkpoint.

    Args:
        model (torch.nn.Module):
            The model to be updated with its saved `state_dict`.
        optimizer (torch.optim.Optimizer):
            The optimizer to be updated with its saved `state_dict`.
        filepath (str): The file path of the general checkpoint.

    Returns:
        A dictionary containing the following keys:
            - 'model': The updated model with its saved `state_dict`.
            - 'optimizer': The updated optimizer with its saved `state_dict`.
            - 'epoch': The epoch value from the last checkpoint.
            - 'loss': The loss value from the last checkpoint.
    """
    try:
        checkpoint = torch.load(f=filepath, weights_only=True)
    except RuntimeError:
        checkpoint = torch.load(
            f=filepath,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return {
        "epoch": checkpoint["epoch"],
        "model": model,
        "optimizer": optimizer,
        "val_loss": checkpoint["val_loss"],
    }


def write_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    filepath: str = "./checkpoint.pth",
) -> None:
    """Saves the states of the training components and also the
    last validation loss and epoch.

    Args:
        epoch (int): Current epoch.
        model (torch.nn.Module): The language model.
        optimizer (torch.optim.Optimizer): The optimizer.
        val_loss (float): Loss value during validation.
        filepath (str, optional): Path to write the checkpoint.
            Should include either `.pth` or `.pt` as the file
            extension. Defaults to ``'./checkpoint.pth'``.
    """
    if not os.path.isdir(s=os.path.dirname(filepath)):
        os.makedirs(name=os.path.dirname(filepath), exist_ok=True)

    torch.save(
        obj={
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        f=filepath,
    )


class EarlyStopping:
    """Stops the training process and checkpoints the states of
    the training components of a seq-to-seq model.

    The stopping occurs if the loss value during the validation
    step stops decreasing for a number of epochs specified by
    the :attr:`patience`.

    Args:
        patience (int, optional): Number of epochs to wait
            before early stopping. (default=5).
        delta (float, optional): Minimum change in monitored
            quantity to qualify as an improvement (default=0).
        verbose (bool, optional): If ``True``, logs a message
            for each improvement. Defaults to `False`.
        filepath (str, optional): Path to write the checkpoint.
            Should include either `.pth` or `.pt` as the file
            extension. Defaults to ``'./checkpoint.pth'``.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        val_loss_min: float = np.Inf,
        filepath: str = "./checkpoint.pt",
    ) -> None:
        assert os.path.basename(filepath).endswith(
            (".pth", ".pt")
        ), "model_name should end with '.pt' or '.pth'"

        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.filepath: str = filepath

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = val_loss_min

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        score = -val_loss

        if not self.best_score:
            self.best_score = score

            if self.verbose:
                logger.info(
                    f"Validation loss decreased "
                    f"({self.val_loss_min:.4f} --> {val_loss:.4f})."
                )

            write_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                filepath=self.filepath,
            )

            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            if self.verbose:
                logger.info(
                    f"Validation loss decreased "
                    f"({self.val_loss_min:.4f} --> {val_loss:.4f})."
                )

            write_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                filepath=self.filepath,
            )

            self.counter = 0
            self.val_loss_min = val_loss
