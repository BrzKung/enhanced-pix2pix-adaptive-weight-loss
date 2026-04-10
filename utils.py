import torch
import numpy as np
import pandas as pd
from config import DEVICE


def save_checkpoint(model, optimizer, filepath="my_checkpoint.pth.tar"):
    """Save model and optimizer state to checkpoint file"""
    print(f"=> Saving checkpoint : {filepath}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Load model and optimizer state from checkpoint file"""
    print(f"=> Loading checkpoint : {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_lambda(history_path):
    """Load lambda values from history CSV file

    Args:
        history_path: Path to the history CSV file

    Returns:
        Tuple of (L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA)
    """
    print("=> Loading lambda from history")
    lambda_df = pd.read_csv(history_path)
    last_lambda = lambda_df.iloc[-1]

    L1_LAMBDA = last_lambda["l1_lambda"] if "l1_lambda" in last_lambda.index else 0.1
    PERC_LAMBDA = (
        last_lambda["perc_lambda"] if "perc_lambda" in last_lambda.index else 0.1
    )
    STYLE_LAMBDA = (
        last_lambda["style_lambda"] if "style_lambda" in last_lambda.index else 0.1
    )

    print(f"l1_lambda: {L1_LAMBDA}")
    print(f"perc_lambda: {PERC_LAMBDA}")
    print(f"style_lambda: {STYLE_LAMBDA}")

    return L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA
