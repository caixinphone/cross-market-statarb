"""1D CNN that predicts a trade's net-of-cost margin from its feature window.

Architecture (deliberately small — a few thousand trades, heavy noise):

    Conv1d(C→H,3) → ReLU → Conv1d(H→H,3) → ReLU → AdaptiveAvgPool1d(1) → Linear(H→1)

The convolution slides along the *time* axis of the (channels, L) window, so the
net learns short temporal motifs in the residual/z-score/vol path. Features are
standardised with **training-set** statistics only (no leakage). Determinism is
pinned via ``torch.manual_seed`` + CPU so results reproduce.
"""

from __future__ import annotations

import numpy as np

# torch is an optional dependency (the `ml` extra). Imported lazily so the base
# install and the 16 baseline tests never require it.
import torch
from torch import nn


class Conv1DSizer(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Standardizer:
    """Per-channel mean/std from training data only."""

    def __init__(self, X: np.ndarray):
        # X: (n, C, L) -> stats per channel over (n, L)
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(X: np.ndarray, y: np.ndarray, *, hidden: int, epochs: int,
                lr: float, batch_size: int, seed: int, dropout: float = 0.1,
                weight_decay: float = 1e-4, val_frac: float = 0.2,
                patience: int = 5) -> Conv1DSizer:
    """Fit the CNN to predict net margin (MSE). CPU, deterministic.

    A time-ordered tail (``val_frac``) is held out as validation for early
    stopping (keep the best-val weights), with dropout + weight decay — light
    regularisation to improve out-of-sample generalisation without enlarging the
    net or searching hyper-parameters.
    """
    set_seed(seed)
    model = Conv1DSizer(in_channels=X.shape[1], hidden=hidden, dropout=dropout)
    y_scaled = y * 100.0                      # scale tiny margin for conditioning
    n = len(X)
    n_val = int(n * val_frac)
    tr = slice(0, n - n_val) if n_val > 0 else slice(0, n)   # time-ordered split
    Xtr = torch.tensor(X[tr], dtype=torch.float32)
    ytr = torch.tensor(y_scaled[tr], dtype=torch.float32)
    Xva = torch.tensor(X[n - n_val:], dtype=torch.float32) if n_val > 0 else None
    yva = torch.tensor(y_scaled[n - n_val:], dtype=torch.float32) if n_val > 0 else None

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best_val, best_state, wait = float("inf"), None, 0
    ntr = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(ntr, generator=g)
        for s in range(0, ntr, batch_size):
            b = perm[s: s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(Xva), yva))
            if v < best_val - 1e-9:
                best_val, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: Conv1DSizer, X: np.ndarray) -> np.ndarray:
    """Predicted net margin (un-scaled back to per-$ units)."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return out / 100.0
