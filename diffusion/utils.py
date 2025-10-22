import gc
import random

import numpy as np
import seaborn as sns
import torch
import torchvision
from rich import print
from rich.pretty import pprint


def print_rich_dict(data: dict) -> None:
    """Pretty print dictionary with colors using rich."""
    pprint(data, expand_all=True)


def print_color(text: str, color: str = "red"):
    print(f"[{color}]{text}[/{color}]")


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def tensor_to_device(*tensors, device=torch.device("cpu"), non_blocking=True):
    moved = tuple(t.to(device, non_blocking=non_blocking) for t in tensors)
    return moved if len(moved) > 1 else moved[0]


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return


def expand_tensor(t: torch.Tensor, dim: int = 2, head: bool = False) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)

    cur_dim = t.dim()
    if cur_dim > dim:
        raise ValueError(f"Current tensor dimension {cur_dim} is greater than target dimension {dim}.")

    n_expand = dim - cur_dim
    if n_expand == 0:
        return t
    if head:
        new_shape = (1,) * n_expand + t.shape
    else:
        new_shape = t.shape + (1,) * n_expand
    return t.view(new_shape)


def save_samples_grid(samples: torch.Tensor, filename: str, nrow: int = 8, padding: int = 1):
    if samples.min() < 0.0:
        samples = (samples + 1.0) / 2.0  # Scale from [-1, 1] to [0, 1]
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=padding)
    torchvision.utils.save_image(grid, filename)


def save_loss_curve(losses: list[float], filename: str):
    import matplotlib.pyplot as plt

    sns.set_theme()

    sns.set_context("talk")  # larger labels
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Step Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Per Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # high-res PNG
    plt.show()
