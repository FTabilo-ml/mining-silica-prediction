# ------------------  src/models/train_transformer.py  ------------------
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

DATA_PATH   = Path("data/processed/train.parquet")
RESULTS_DIR = Path("results/deep")
MODELS_DIR  = Path("models")
TARGET      = "% Silica Concentrate"

# -------- dataset --------
class SeqDataset(Dataset):
    def __init__(self, X, y, lookback, horizon, step=1):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.L = lookback; self.H = horizon; self.step = step
        last = len(X) - horizon
        self.ends = np.arange(lookback, last, step, dtype=np.int64)

    def __len__(self): return len(self.ends)

    def __getitem__(self, i):
        t = self.ends[i]
        x = self.X[t-self.L:t, :]
        y = self.y[t + (self.H - 1)]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

def split_time(df, test_days=30, val_days=30):
    test_cut = df.index.max() - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)
    return df.loc[:val_cut], df.loc[val_cut+pd.Timedelta(hours=1):test_cut], df.loc[test_cut+pd.Timedelta(hours=1):]

def metrics(y_true, y_pred):
    return (root_mean_squared_error(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))

# -------- modelo Transformer --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # 👇 Anotación para que Pylance sepa que es Tensor
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TSTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos  = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, F]
        z = self.proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)

def run_epoch(model, loader, criterion, opt=None, device="cpu"):
    training = opt is not None
    model.train(training)
    losses, Y, P = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if training: opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        losses.append(loss.item())
        Y.append(yb.detach().cpu().numpy()); P.append(pred.detach().cpu().numpy())
    y = np.concatenate(Y); p = np.concatenate(P)
    rmse, mae, r2 = metrics(y, p)
    return float(np.mean(losses)), rmse, mae, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=180)
    ap.add_argument("--horizon",  type=int, default=1)
    ap.add_argument("--batch",    type=int, default=512)
    ap.add_argument("--epochs",   type=int, default=10)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--step",     type=int, default=5)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_parquet(DATA_PATH)
    X_cols = [c for c in df.columns if c != TARGET]
    X = df[X_cols].astype(np.float32).values
    y = df[TARGET].astype(np.float32).values

    tr, va, te = split_time(df)
    n_tr, n_va, n_te = len(tr), len(va), len(te)
    X_tr, X_va, X_te = X[:n_tr], X[n_tr:n_tr+n_va], X[-n_te:]
    y_tr, y_va, y_te = y[:n_tr], y[n_tr:n_tr+n_va], y[-n_te:]

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)

    ds_tr = SeqDataset(X_tr, y_tr, args.lookback, args.horizon, step=args.step)
    ds_va = SeqDataset(X_va, y_va, args.lookback, args.horizon, step=1)
    ds_te = SeqDataset(X_te, y_te, args.lookback, args.horizon, step=1)
    pin = (device == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, pin_memory=pin)

    model = TSTransformer(n_features=X.shape[1], d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    crit = nn.MSELoss()

    best_rmse = float("inf"); best_path = MODELS_DIR / "transformer_best.pt"
    for ep in range(1, args.epochs+1):
        trm = run_epoch(model, dl_tr, crit, opt, device)
        vam = run_epoch(model, dl_va, crit, None, device)
        print(f"[{ep:02d}] train rmse={trm[1]:.4f}  val rmse={vam[1]:.4f}  (mae={vam[2]:.4f} r2={vam[3]:.4f})")
        if vam[1] < best_rmse:
            best_rmse = vam[1]
            # Evitar objetos inseguros en pickle para PyTorch 2.6
            sc_mean  = np.array(scaler.mean_,  dtype=np.float32).tolist()
            sc_scale = np.array(scaler.scale_, dtype=np.float32).tolist()
            meta = {"cols": X_cols, "lookback": args.lookback, "horizon": args.horizon,
                    "scaler_mean": sc_mean, "scaler_scale": sc_scale}
            torch.save({"model": model.state_dict(), "meta": meta}, best_path)

    # PyTorch 2.6: permitir cargar nuestro checkpoint propio
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    _, rmse, mae, r2 = run_epoch(model, dl_te, crit, None, device)
    pd.DataFrame([{"model":"Transformer", "lookback":args.lookback, "horizon":args.horizon,
                  "rmse":rmse, "mae":mae, "r2":r2}]).to_csv(RESULTS_DIR / "transformer_scores.csv", index=False)
    print("✅ Transformer listo. Test RMSE/MAE/R2:", rmse, mae, r2, "\nGuardado:", best_path)

if __name__ == "__main__":
    main()
