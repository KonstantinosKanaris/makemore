from typing import Optional

import mlflow
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from makemore import logger
from makemore.utils.aux import (
    EarlyStopping,
    print_samples,
    set_seed,
    write_checkpoint,
)
from makemore.utils.constants import SEED
from makemore.vectorizer import CharTokenizer

set_seed(seed=SEED)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    vectorizer: CharTokenizer,
    num_epochs: int,
    checkpoint_path: str,
    tqdm_bar: tqdm,
    start_epoch: int = 0,
    device: str = "cpu",
    early_stopper: Optional[EarlyStopping] = None,
) -> None:
    for epoch_idx in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        for batch_idx, (X, y) in enumerate(train_dataloader):
            desc = (
                f"Training: [{epoch_idx}/{num_epochs+1}] | "
                f"[{batch_idx}/{len(train_dataloader)}]"
            )
            tqdm_bar.set_description(desc=desc)
            tqdm_bar.set_postfix(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": f"{train_loss:.4f}",
                }
            )

            X, y = X.to(device), y.to(device)

            # 1. forward pass (returns logits)
            # calculate and accumulate the loss
            y_pred = model(X)
            B, T, C = y_pred.size()
            loss = F.cross_entropy(
                input=y_pred.view(B * T, C),
                target=y.view(B * T),
                ignore_index=-1,
            )
            train_loss += (loss.item() - train_loss) / (batch_idx + 1)

            # 3. zeroing gradients
            optimizer.zero_grad(set_to_none=True)

            # 4. calculate gradients during backward pass
            loss.backward()

            # 5. update the parameters
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(val_dataloader):
                desc = (
                    f"Validation: [{epoch_idx}/{num_epochs+1}] | "
                    f"[{batch_idx}/{len(val_dataloader)}]"
                )
                tqdm_bar.set_description(desc=desc)
                tqdm_bar.set_postfix(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": f"{val_loss:.4f}",
                    }
                )

                X, y = X.to(device), y.to(device)

                # 1. forward pass (returns logits)
                # calculate and accumulate the loss
                y_pred = model(X)
                B, T, C = y_pred.size()
                loss = F.cross_entropy(
                    input=y_pred.view(B * T, C),
                    target=y.view(B * T),
                    ignore_index=-1,
                )
                val_loss += (loss.item() - val_loss) / (batch_idx + 1)

        mlflow.log_metrics(
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            step=epoch_idx,
        )

        if epoch_idx > 0 and epoch_idx % 10 == 0:
            print_samples(
                model=model,
                num=20,
                vectorizer=vectorizer,
                device=device,
            )

        lr_scheduler.step(metrics=val_loss)

        if early_stopper:
            early_stopper(
                epoch=epoch_idx,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
            )
        else:
            write_checkpoint(
                epoch=epoch_idx,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                filepath=checkpoint_path,
            )

        if early_stopper and early_stopper.early_stop:
            logger.info("Stopping training process due to early stopping.")
            break
        else:
            tqdm_bar.update()
            continue
