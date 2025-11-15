# train_flow_velocity.py

import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from architecture import FlowTSVelocityWrapper


DATA_DIR = "data"
WINDOW_LEN = 30        
TAU_VALUES = [1, 2, 3]  
TRAIN_FRACTION = 0.8    
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_multi_index_features() -> pd.DataFrame:
    """
    Load per index feature CSVs and align them on the common date intersection.
    Returns a single DataFrame with all features.
    """

    files = {
        "djia": os.path.join(DATA_DIR, "djia_features.csv"),
        "snp500": os.path.join(DATA_DIR, "snp500_features.csv"),
        "nasdaq": os.path.join(DATA_DIR, "nasdaq_features.csv"),
        "russel2000": os.path.join(DATA_DIR, "russel2000_features.csv"),
        "nyse": os.path.join(DATA_DIR, "nyse_features.csv"),
        "nysesmcap": os.path.join(DATA_DIR, "nysesmcap_features.csv"),
    }

    frames = []
    for name, path in files.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # add prefix so columns are unique per index
        df = df.add_prefix(f"{name}_")
        frames.append(df)

    features = pd.concat(frames, axis=1, join="inner")
    features = features.sort_index()
    # drop early rows with missing values from rolling windows
    features = features.dropna()
    return features


def standardise_features(df: pd.DataFrame, train_end_index: int):
    """
    Standardise features using mean and std computed on the training period only.
    Returns standardised DataFrame and the mean and std arrays for later use.
    """

    train_df = df.iloc[:train_end_index]
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0).replace(0.0, 1.0)

    df_std = (df - mean) / std
    return df_std, mean.values.astype(np.float32), std.values.astype(np.float32)


class FlowVelocityDataset(Dataset):
    """
    Dataset that provides (window, tau, velocity target) triples.

    Input window is a slice of length L ending at time t inclusive.
    Target is v_star(t, tau) = (x_{t+tau} minus x_t) divided by tau.
    """

    def __init__(
        self,
        X_all: np.ndarray,
        centers: List[int],
        tau_values: List[int],
        window_len: int,
    ):
        super().__init__()
        self.X_all = X_all          # shape (N, F)
        self.centers = centers      # list of center time indices t
        self.tau_values = tau_values
        self.window_len = window_len
        self.K = len(tau_values)

    def __len__(self) -> int:
        # each center time is paired with all tau values
        return len(self.centers) * self.K

    def __getitem__(self, idx: int):
        """
        Returns:
            x_window: Tensor with shape (L, F)
            tau: Tensor with shape (,) scalar
            v_star: Tensor with shape (F,)
        """
        center_index = self.centers[idx // self.K]
        tau_idx = idx % self.K
        tau = self.tau_values[tau_idx]

        t = center_index
        L = self.window_len

        x_window = self.X_all[t - L + 1 : t + 1]          # (L, F)
        x_t = self.X_all[t]                               # (F,)
        x_future = self.X_all[t + tau]                    # (F,)
        v_star = (x_future - x_t) / float(tau)            # (F,)

        x_window = torch.from_numpy(x_window).float()
        tau_tensor = torch.tensor(float(tau), dtype=torch.float32)
        v_star = torch.from_numpy(v_star).float()

        return x_window, tau_tensor, v_star


def build_datasets(df_features: pd.DataFrame):
    """
    Build training and validation datasets with chronological split.

    Returns:
        train_ds, val_ds, dim_x, num_train_centers, num_val_centers
    """

    X_all = df_features.values.astype(np.float32)   # (N, F)
    N, F = X_all.shape

    max_tau = max(TAU_VALUES)
    L = WINDOW_LEN

    # valid center times t must satisfy
    # t >= L minus 1 and t + max_tau < N
    t_min = L - 1
    t_max = N - 1 - max_tau
    centers_all = list(range(t_min, t_max + 1))
    num_centers = len(centers_all)

    # split centers by time
    num_train_centers = int(TRAIN_FRACTION * num_centers)
    train_centers = centers_all[:num_train_centers]
    val_centers = centers_all[num_train_centers:]

    train_ds = FlowVelocityDataset(X_all, train_centers, TAU_VALUES, WINDOW_LEN)
    val_ds = FlowVelocityDataset(X_all, val_centers, TAU_VALUES, WINDOW_LEN)

    return train_ds, val_ds, F, num_train_centers, len(val_centers)


def train_epoch(
    model: FlowTSVelocityWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single training epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    mse = nn.MSELoss()

    for x_window, tau, v_star in loader:
        x_window = x_window.to(device)          # (B, L, F)
        tau = tau.to(device)                    # (B,)
        v_star = v_star.to(device)              # (B, F)

        optimizer.zero_grad()

        v_pred = model(x_window, tau)           # (B, F)
        loss = mse(v_pred, v_star)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_epoch(
    model: FlowTSVelocityWrapper,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate on validation loader. Returns average loss.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    mse = nn.MSELoss()

    with torch.no_grad():
        for x_window, tau, v_star in loader:
            x_window = x_window.to(device)
            tau = tau.to(device)
            v_star = v_star.to(device)

            v_pred = model(x_window, tau)
            loss = mse(v_pred, v_star)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    # 1. Load features and split train and validation by time
    df_raw = load_multi_index_features()
    N_total = len(df_raw)

    # temporary split index for standardisation
    max_tau = max(TAU_VALUES)
    L = WINDOW_LEN
    t_min = L - 1
    t_max = N_total - 1 - max_tau
    centers_all = list(range(t_min, t_max + 1))
    num_centers = len(centers_all)
    num_train_centers = int(TRAIN_FRACTION * num_centers)

    # approximate index in the DataFrame for the end of training period
    # training centers use indices up to centers_all[num_train_centers minus 1]
    train_end_time_index = centers_all[num_train_centers - 1] + 1

    # 2. Standardise features using only training period stats
    df_std, mean_vec, std_vec = standardise_features(df_raw, train_end_time_index)

    # 3. Build datasets and loaders
    train_ds, val_ds, dim_x, n_train_centers, n_val_centers = build_datasets(df_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print(f"Number of days in DataFrame: {len(df_std)}")
    print(f"Feature dimension dim_x: {dim_x}")
    print(f"Window length L: {WINDOW_LEN}")
    print(f"Train centers: {n_train_centers}, validation centers: {n_val_centers}")
    print(f"Train examples: {len(train_ds)}, validation examples: {len(val_ds)}")

    # 4. Build model
    model = FlowTSVelocityWrapper(
        dim_x=dim_x,
        window_len=WINDOW_LEN,
        dim_model=128,
        num_heads=4,
        num_layers_enc=3,
        num_layers_dec=2,
        num_registers=2,
        dim_ff=256,
        dim_time=32,
        dropout=0.0,
    )
    model.to(DEVICE)

    # 5. Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 6. Training loop
    best_val_loss = math.inf

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate_epoch(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:03d}  "
            f"train loss {train_loss:.6f}  "
            f"val loss {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean_vec,
                    "std": std_vec,
                    "dim_x": dim_x,
                    "window_len": WINDOW_LEN,
                    "tau_values": TAU_VALUES,
                },
                os.path.join("checkpoints", "best_flow_velocity.pt"),
            )

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
