"""The project's orchestrator.

Defines the execution flow, including loading the data, initializing
the training components (model, optimizer, learning rate scheduler),
and executing the training process.
"""

import argparse
import os.path
import random
import sys
from typing import List, Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from makemore import logger
from makemore.dataset import CharDataset
from makemore.engine.train import train
from makemore.models.bow import BoW
from makemore.models.simple import MLP, Bigram
from makemore.models.transformer import GPTLanguageModel
from makemore.utils.aux import (
    EarlyStopping,
    load_from_txt,
    load_general_checkpoint,
    print_samples,
    set_seed,
)
from makemore.utils.constants import SEED
from makemore.vectorizer import CharTokenizer
from makemore.vocabulary import CharVocabulary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command Line Interface for makemore."
    )

    # system input-output
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="The input file with things one per line.",
    )
    parser.add_argument(
        "--workdir",
        "-o",
        type=str,
        default="out",
        required=False,
        help="Output working directory.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        required=False,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="When this flag is used, the training process will "
        "resume optimization from existing model in the workdir.",
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Print generated samples from the model (no training).",
    )

    # model
    parser.add_argument(
        "--type",
        type=str,
        default="transformer",
        required=False,
        help="Model class type to use, bigram|mlp|bow|transformer.",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=32,
        required=False,
        help="Size of each embedding vector (for mlp and bow models).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        required=False,
        help="Size of the hidden layers (for mlp and bow models).",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        required=False,
        help="Dimensionality of the vectors used throughout the transformer.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="When this flag is used, bias will be added to the input / output"
        "projection layers of self-attention (in a transformer).",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=4,
        required=False,
        help="Number of self-attention heads (in a transformer).",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=4,
        required=False,
        help="Number of sub-decoder layers (in a transformer).",
    )

    # optimization
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        required=False,
        default=64,
        help="The batch size during optimization",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        type=float,
        required=False,
        default=5e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        "-w",
        type=float,
        required=False,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="When this flag is used, early stopping will be enabled.",
    )
    parser.add_argument(
        "--patience",
        "-p",
        type=int,
        required=False,
        default=7,
        help="Number of epochs to wait before early stopping.",
    )
    parser.add_argument(
        "--delta",
        "-d",
        type=float,
        required=False,
        default=0.0001,
        help="Minimum change in monitored quantity to qualify as an improvement.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="When this flag is used, verbose during "
        "early stopping will be enabled.",
    )
    return parser.parse_args()


def build_vocabulary(words: List[str]) -> CharVocabulary:
    chars = sorted(list(set("".join(words))))
    return CharVocabulary(chars=chars)


def split_data(words: List[str]) -> Tuple[List[str], List[str]]:
    random.seed(SEED)
    random.shuffle(words)
    n1 = int(0.9 * (len(words)))
    train_split = words[:n1]
    val_split = words[n1:]
    logger.info(
        f"Split up the dataset into {len(train_split)} training examples, "
        f"{len(val_split)} validation examples, "
    )
    return train_split, val_split


if __name__ == "__main__":
    args = parse_arguments()

    # system initializations
    set_seed(seed=SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    data = load_from_txt(input_file=args.input_file)

    # build vocabulary
    vocab = build_vocabulary(words=data)
    vocab_size = len(vocab)

    # split the input data into a training and test set
    train_words, val_words = split_data(words=data)

    # initialize vectorizer
    max_word_length = max(len(word) for word in data)
    vectorizer = CharTokenizer(vocab=vocab, max_word_length=max_word_length)

    # initialize datasets
    train_dataset = CharDataset(words=train_words, vectorizer=vectorizer)
    val_dataset = CharDataset(words=val_words, vectorizer=vectorizer)
    block_size = train_dataset.get_output_length()
    logger.info(f"{vocab_size=}, {block_size=}")

    # initialize dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # initialize model
    models = {
        "bigram": Bigram(vocab_size=vocab_size),
        "mlp": MLP(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
        ),
        "bow": BoW(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            device=device,
        ),
        "transformer": GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            d_model=args.d_model,
            n_head=args.n_head,
            n_layer=args.n_layer,
            bias=args.bias,
            device=device,
        ),
    }
    if args.type not in models.keys():
        logger.error(f"model type {args.type} is not recognized.")
        raise ValueError(f"model type {args.type} is not recognized.")
    else:
        model = models[args.type]
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Selected model: {model.__class__.__name__} | "
        f"Model #params: {num_params}"
    )

    # create checkpoint path
    checkpoint_path = os.path.join(
        args.workdir,
        f"{args.type}_params{num_params}.pth",
    )
    logger.info(f"Checkpoint filepath: {checkpoint_path}")

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    # initialize lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=2
    )

    # initialize model and optimizer from checkpoint
    start_epoch = 1
    last_best_score = np.Inf
    if args.resume:
        logger.info("Resuming from last saved checkpoint.")
        checkpoint = load_general_checkpoint(
            model=model,
            optimizer=optimizer,
            filepath=checkpoint_path,
        )
        model = checkpoint["model"].to(device)
        optimizer = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"] + 1
        last_best_score = checkpoint["val_loss"]
        logger.info(
            f"last epoch: {start_epoch - 1} | "
            f"last val loss: {last_best_score:.4f}"
        )
    # Generate samples (no training)
    if args.sample_only:
        checkpoint = load_general_checkpoint(
            model=model,
            optimizer=optimizer,
            filepath=checkpoint_path,
        )
        model = checkpoint["model"].to(device)
        print_samples(model=model, vectorizer=vectorizer, device=device, num=30)
        sys.exit()

    # initialize early stopping
    early_stopper = None
    if args.early_stop:
        logger.info(
            f"Early stopping is enabled with: "
            f"patience={args.patience}, delta={args.delta}"
        )
        early_stopper = EarlyStopping(
            patience=args.patience,
            delta=args.delta,
            verbose=args.verbose,
            val_loss_min=last_best_score,
            filepath=checkpoint_path,
        )

    # initialize tqdm bar
    bar_fmt = "{desc} {percentage:3.0f}%|{bar}{postfix} " "[{elapsed}<{remaining}]"
    tqdm_bar = tqdm(
        iterable=range(args.num_epochs + 1),
        total=args.num_epochs + 1,
        bar_format=bar_fmt,
        initial=start_epoch,
        position=0,
        leave=False,
    )

    # start training loop
    with mlflow.start_run(run_name=f"{args.type}_" f"params{num_params}"):
        train(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            num_epochs=args.num_epochs,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_path,
            vectorizer=vectorizer,
            early_stopper=early_stopper,
            tqdm_bar=tqdm_bar,
        )
