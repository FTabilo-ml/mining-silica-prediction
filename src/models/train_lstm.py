# --------------------- src/models/train_lstm.py ---------------------
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib


# --------------------- Config ---------------------
DATA_PATH   = Path("data/processed/train.parquet")
RESULTS_DIR = Path("results/lstm")
MODELS_DIR  = Path("models")
TARGET      = "% Silica Concentrate"

# ventana y horizonte
LOOKBACK = 24          # horas hacia atrás
HORIZON  = 1           # predecir 1 hora adelante

# entrenamiento
BATCH_SIZE = 512
EPOCHS     = 25
LR         = 1e-3
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.2
PATIENCE   = 5

USE_GPU = True
DEVICE  = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
SEED    = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------- Utilidades ---------------------
def train_val_test_split(df: pd.DataFrame, test_days=30, val_days=30
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide cronológicamente: train / val / test."""
    test_cut = df.index.max() - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)
    train = df.loc[:val_cut]
    val   = df.loc[val_cut + pd.Timedelta(hours=1):test_cut]
    test  = df.loc[test_cut + pd.Timedelta(hours=1):]
    return train, val, test


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Compatibilidad con versiones de sklearn
    try:
        from sklearn.metrics import root_mean_squared_error as _rmse
        return float(_rmse(y_true, y_pred))
    except Exception:
        return float(mean_squared_error(y_true, y_pred, squared=False))


class SeqDataset(Dataset):
    """Crea muestras [lookback, n_features] -> y (escala) con salto horizonte."""
    def __init__(self, X: ArrayLike, y: ArrayLike, lookback: int, horizon: int, step: int = 1):
        self.X = np.asarray(X, dtype=np.float32, order="C")
        self.y = np.asarray(y, dtype=np.float32)
        self.lookback = int(lookback)
        self.horizon  = int(horizon)
        self.step     = int(step)

        assert self.X.ndim == 2, "X debe ser [time, features]"
        assert self.y.ndim == 1, "y debe ser vector"

        self._starts = np.arange(
            0, len(self.X) - self.lookback - self.horizon + 1, self.step, dtype=int
        )

    def __len__(self) -> int:
        return self._starts.shape[0]

    def __getitem__(self, i: int):
        s = self._starts[i]
        e = s + self.lookback
        x = self.X[s:e]                                # [L, F]
        y = self.y[e + self.horizon - 1]               # escalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):          # x: [B, L, F]
        out, _ = self.lstm(x)      # [B, L, H]
        last = out[:, -1, :]       # [B, H]
        y = self.head(last).squeeze(-1)  # [B]
        return y


def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
        lr: float = 1e-3,
        patience: int = 5,
        device: str = "cpu"):

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    # ← asegurar que best_state existe desde el inicio
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_ctr = 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vl_losses.append(loss_fn(pred, yb).item())

        tr = float(np.mean(tr_losses))
        vl = float(np.mean(vl_losses))
        history.append((ep, tr, vl))
        print(f"[{ep:02d}/{epochs}] train_mse={tr:.5f} | val_mse={vl:.5f}")

        if vl < best_val - 1e-6:
            best_val = vl
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return history



def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


# --------------------- Main ---------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Datos
    df = pd.read_parquet(DATA_PATH)
    assert isinstance(df.index, pd.DatetimeIndex), "El índice debe ser DatetimeIndex."

    train_df, val_df, test_df = train_val_test_split(df)

    X_cols = [c for c in df.columns if c != TARGET]

    # to_numpy para evitar ExtensionArray y forzar float32
    X_train = train_df[X_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train_df[TARGET].to_numpy(dtype=np.float32, copy=True)

    X_val   = val_df[X_cols].to_numpy(dtype=np.float32, copy=True)
    y_val   = val_df[TARGET].to_numpy(dtype=np.float32, copy=True)

    X_test  = test_df[X_cols].to_numpy(dtype=np.float32, copy=True)
    y_test  = test_df[TARGET].to_numpy(dtype=np.float32, copy=True)

    # 2) Escalado (fit sólo en train)
    x_scaler = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)
    X_val_sc   = x_scaler.transform(X_val)
    X_test_sc  = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_sc  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # 3) Datasets + Loaders
    # 3) Datasets
    ds_tr = SeqDataset(X_train_sc, y_train_sc, LOOKBACK, HORIZON)
    ds_va = SeqDataset(X_val_sc,   y_val_sc,   LOOKBACK, HORIZON)
    ds_te = SeqDataset(X_test_sc,  y_test_sc,  LOOKBACK, HORIZON)

    # Loaders: uno para entrenar y uno para evaluar train
    dl_tr_train = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
                            pin_memory=(DEVICE=="cuda"))
    dl_tr_eval  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=(DEVICE=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                    pin_memory=(DEVICE=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                    pin_memory=(DEVICE=="cuda"))

    # 4) Modelo
    n_features = X_train_sc.shape[1]
    model = LSTMRegressor(n_features, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT)

    # 5) Entrenamiento
    print(f"Entrenando en dispositivo: {DEVICE.upper()}")
    history = fit(model, dl_tr_train, dl_va, EPOCHS, LR, PATIENCE, DEVICE)

    
    # 6) Evaluación (en escala original)
    yhat_tr_sc = evaluate(model, dl_tr_eval, DEVICE)
    yhat_va_sc = evaluate(model, dl_va, DEVICE)
    yhat_te_sc = evaluate(model, dl_te, DEVICE)

    yhat_tr = y_scaler.inverse_transform(yhat_tr_sc.reshape(-1, 1)).ravel()
    yhat_va = y_scaler.inverse_transform(yhat_va_sc.reshape(-1, 1)).ravel()
    yhat_te = y_scaler.inverse_transform(yhat_te_sc.reshape(-1, 1)).ravel()

    # Alinear longitudes con el dataset (cada dataset ya es desfasado por lookback+horizon-1)
    y_tr_true = y_train[LOOKBACK + HORIZON - 1:]
    y_va_true = y_val[LOOKBACK + HORIZON - 1:]
    y_te_true = y_test[LOOKBACK + HORIZON - 1:]

    metrics = {
        "train_rmse": rmse_metric(y_tr_true, yhat_tr),
        "train_mae":  float(mean_absolute_error(y_tr_true, yhat_tr)),
        "train_r2":   float(r2_score(y_tr_true, yhat_tr)),

        "val_rmse": rmse_metric(y_va_true, yhat_va),
        "val_mae":  float(mean_absolute_error(y_va_true, yhat_va)),
        "val_r2":   float(r2_score(y_va_true, yhat_va)),

        "test_rmse": rmse_metric(y_te_true, yhat_te),
        "test_mae":  float(mean_absolute_error(y_te_true, yhat_te)),
        "test_r2":   float(r2_score(y_te_true, yhat_te)),
    }
    print("Métricas:", metrics)

    # 7) Guardar artefactos
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # modelo y scalers
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler},
                MODELS_DIR / "lstm_scalers.pkl")

    # métricas
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "lstm_metrics.csv", index=False)

    # curva simple de pred vs real en test
    idx_test = test_df.index[LOOKBACK + HORIZON - 1:]
    pred_df = pd.DataFrame({"y_true": y_te_true, "y_pred": yhat_te}, index=idx_test)
    pred_df.to_csv(RESULTS_DIR / "lstm_test_preds.csv")

    print("✅ LSTM entrenado y guardado.")
    print(f"Model: {MODELS_DIR / 'lstm_model.pt'}")
    print(f"Metrics: {RESULTS_DIR / 'lstm_metrics.csv'}")
    print(f"Preds test: {RESULTS_DIR / 'lstm_test_preds.csv'}")


if __name__ == "__main__":
    main()
