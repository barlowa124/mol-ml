"""Fusion MLP and single-modality ablation heads for pKd regression.

Fusion: drug fingerprint -> linear, protein embedding -> linear, concat ->
MLP -> scalar pKd. The ablations share the same head shape so the only
difference is which modality enters, the controlled test of whether combining
modalities beats either alone.
"""

import numpy as np
import torch
import torch.nn as nn


class BranchMLP(nn.Module):
    def __init__(self, d_dim: int, p_dim: int, hidden: int, dropout: float,
                 mode: str = "fusion"):
        super().__init__()
        if mode not in ("fusion", "drug", "protein"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.drug_branch = nn.Sequential(
            nn.Linear(d_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.prot_branch = nn.Sequential(
            nn.Linear(p_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        head_in = {"fusion": 2 * hidden, "drug": hidden,
                   "protein": hidden}[mode]
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, xd, xp):
        if self.mode == "drug":
            z = self.drug_branch(xd)
        elif self.mode == "protein":
            z = self.prot_branch(xp)
        else:
            z = torch.cat([self.drug_branch(xd), self.prot_branch(xp)], dim=1)
        return self.head(z).squeeze(-1)


def train_model(mode, Xd, xp, y, tr, te, cfg, seed):
    """Fit one modality configuration; returns test-set predictions."""
    torch.manual_seed(seed)
    model = BranchMLP(Xd.shape[1], xp.shape[1], cfg["hidden"],
                      cfg["dropout"], mode)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xd_t = torch.tensor(Xd[tr], dtype=torch.float32)
    xp_t = torch.tensor(xp[tr], dtype=torch.float32)
    y_t = torch.tensor(y[tr], dtype=torch.float32)
    n = len(xd_t)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch_size"]):
            j = perm[i : i + cfg["batch_size"]]
            loss = nn.functional.mse_loss(
                model(xd_t[j], xp_t[j]), y_t[j]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(Xd[te], dtype=torch.float32),
            torch.tensor(xp[te], dtype=torch.float32),
        ).numpy()
    return pred


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def metrics(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr

    return {
        "mse": mse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }
