# -*- coding: utf-8 -*-
# --------------------------- dashboard/app.py ---------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import torch
from torch import nn
from torch.serialization import add_safe_globals, safe_globals
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DATA_PATH   = ROOT / "data/processed/train.parquet"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results/dashboard"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PATH = MODELS_DIR / "model_xgb.pkl"
LSTM_PATH = MODELS_DIR / "lstm_model_patched.pt"
TR_PATH   = MODELS_DIR / "transformer_best_patched.pt"
TARGET    = "% Silica Concentrate"

# ----------------------------------------------------------------------
#  PyTorch 2.6: loader seguro (allowlist de objetos NumPy)
# ----------------------------------------------------------------------
try:
    _np_reconstruct = getattr(np.core.multiarray, "_reconstruct")  # type: ignore[attr-defined]
except Exception:
    _np_reconstruct = None

_allow: list[Any] = [np.ndarray]
if _np_reconstruct is not None:
    _allow.append(_np_reconstruct)
add_safe_globals(cast(list[Any], _allow))  # type: ignore[arg-type]

def load_ckpt_safe(path: Path, device: str = "cpu"):
    allow = list(_allow)
    try:
        with safe_globals(cast(list[Any], allow)):  # type: ignore[arg-type]
            return torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        return torch.load(str(path), map_location=device, weights_only=False)

# ----------------------------------------------------------------------
#  Modelos
# ----------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 head_width: int | None = None, dropout: float = 0.1):
        super().__init__()
        head_width = head_width or hidden
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, head_width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_width, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe
        return x + pe[:, :x.size(1), :]

class TSTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos  = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)

# ----------------------------------------------------------------------
#  Utilidades
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def build_scaler(mean: np.ndarray | list | None, scale: np.ndarray | list | None) -> StandardScaler | None:
    if mean is None or scale is None:
        return None
    sc = StandardScaler()
    sc.mean_ = np.asarray(mean, dtype=np.float64)
    sc.scale_ = np.asarray(scale, dtype=np.float64)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc

def align_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=cols, fill_value=0.0)

def to_sequences(X: np.ndarray, lookback: int, step: int = 1):
    if X.shape[0] < lookback + 1:
        return np.array([], dtype=np.int64), np.empty((0, lookback, X.shape[1]), dtype=np.float32)
    idx = np.arange(lookback, len(X), step, dtype=np.int64)
    seq = np.stack([X[i - lookback:i, :] for i in idx]).astype(np.float32, copy=False)
    return idx, seq

def metrics(y, p):
    return dict(
        rmse=float(mean_squared_error(y, p, squared=False)),
        mae=float(mean_absolute_error(y, p)),
        r2=float(r2_score(y, p)),
    )

def col_or_zeros(df: pd.DataFrame, name: str) -> pd.Series:
    return (df[name] if name in df.columns else pd.Series(0.0, index=df.index, dtype=float))

def best_lag(y: pd.Series, p: pd.Series, max_lag: int = 6) -> int:
    """Busca el lag (±max_lag horas) con mayor correlación. Robusto a NaNs."""
    z = pd.concat([y, p], axis=1).dropna()
    if z.empty:
        return 0
    yv, pv = z.iloc[:, 0].to_numpy(), z.iloc[:, 1].to_numpy()
    lags = list(range(-max_lag, max_lag + 1))
    scores: list[float] = []
    for L in lags:
        if L < 0:
            a, b = yv[-L:], pv[:len(pv) + L]
        elif L > 0:
            a, b = yv[:len(yv) - L], pv[L:]
        else:
            a, b = yv, pv
        if len(a) < 5:
            scores.append(np.nan); continue
        try:
            scores.append(float(np.corrcoef(a, b)[0, 1]))
        except Exception:
            scores.append(np.nan)
    if not np.isfinite(scores).any():
        return 0
    return int(lags[int(np.nanargmax(scores))])

def remove_bias(y: pd.Series, p: pd.Series, window: int = 48) -> pd.Series:
    z = pd.concat([y, p], axis=1).dropna().iloc[-window:]
    if z.empty:
        return p
    bias = float((z.iloc[:, 1] - z.iloc[:, 0]).median())
    return p - bias

# ----------------------------------------------------------------------
#  Carga de modelos
# ----------------------------------------------------------------------
def load_xgb():
    with open(XGB_PATH, "rb") as f:
        return pickle.load(f)

def _infer_lstm_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    hidden = int(state_dict.get("lstm.weight_hh_l0", torch.empty(0, 128)).shape[1]) if "lstm.weight_hh_l0" in state_dict else 128
    if any(k.startswith("lstm.weight_hh_l") for k in state_dict):
        layers = 1 + max(int(k.split("l")[-1]) for k in state_dict if k.startswith("lstm.weight_hh_l"))
    else:
        layers = 2
    head_w = int(state_dict.get("head.0.weight", torch.empty(64, hidden)).shape[0]) if "head.0.weight" in state_dict else hidden
    return hidden, layers, head_w

def load_lstm(n_features: int, device: str = "cpu"):
    ckpt = load_ckpt_safe(LSTM_PATH, device)
    state_dict = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if state_dict is None:
        raise RuntimeError("Checkpoint LSTM no contiene pesos")

    hidden, layers, head_w = _infer_lstm_dims(state_dict)

    cols = ckpt.get("cols") if isinstance(ckpt, dict) else None
    lookback = int(ckpt.get("lookback", 180)) if isinstance(ckpt, dict) else 180
    horizon  = int(ckpt.get("horizon", 1)) if isinstance(ckpt, dict) else 1
    scaler = build_scaler(ckpt.get("scaler_mean"), ckpt.get("scaler_scale")) if isinstance(ckpt, dict) else None

    model = LSTMRegressor(n_features=n_features, hidden=hidden, layers=layers,
                          head_width=head_w, dropout=0.1).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cols, lookback, horizon, scaler

def load_transformer(n_features: int, device: str = "cpu"):
    ckpt = load_ckpt_safe(TR_PATH, device)
    state_dict = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if state_dict is None:
        raise RuntimeError("Checkpoint Transformer no contiene pesos")

    cols = ckpt.get("cols") if isinstance(ckpt, dict) else None
    lookback = int(ckpt.get("lookback", 180)) if isinstance(ckpt, dict) else 180
    horizon  = int(ckpt.get("horizon", 1)) if isinstance(ckpt, dict) else 1
    scaler = build_scaler(ckpt.get("scaler_mean"), ckpt.get("scaler_scale")) if isinstance(ckpt, dict) else None

    model = TSTransformer(n_features=n_features, d_model=128, nhead=8, num_layers=2,
                          dim_feedforward=256, dropout=0.2).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cols, lookback, horizon, scaler

# ----------------------------------------------------------------------
#  Predicción (con ajuste explícito de horizon)
# ----------------------------------------------------------------------
@torch.inference_mode()
def predict_lstm(model: LSTMRegressor, df: pd.DataFrame, cols: list[str] | None,
                 lookback: int, scaler: StandardScaler | None, device: str,
                 horizon_hours: int = 0) -> pd.Series:
    cols_use = cols or [c for c in df.columns if c != TARGET]
    Xdf = align_cols(df, cols_use).astype(np.float32)
    X = Xdf.values
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    idx, Xseq = to_sequences(X, lookback, step=1)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    xb = torch.from_numpy(Xseq).to(device)
    pred = model(xb).cpu().numpy().ravel()
    idx_time = df.index[idx] + pd.Timedelta(hours=int(horizon_hours))
    return pd.Series(pred, index=idx_time, dtype=float)

@torch.inference_mode()
def predict_transformer(model: TSTransformer, df: pd.DataFrame, cols: list[str] | None,
                        lookback: int, scaler: StandardScaler | None, device: str,
                        horizon_hours: int = 0) -> pd.Series:
    cols_use = cols or [c for c in df.columns if c != TARGET]
    Xdf = align_cols(df, cols_use).astype(np.float32)
    X = Xdf.values
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    idx, Xseq = to_sequences(X, lookback, step=1)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    xb = torch.from_numpy(Xseq).to(device)
    pred = model(xb).cpu().numpy().ravel()
    idx_time = df.index[idx] + pd.Timedelta(hours=int(horizon_hours))
    return pd.Series(pred, index=idx_time, dtype=float)

def predict_xgb(model, df: pd.DataFrame) -> pd.Series:
    X_cols = [c for c in df.columns if c != TARGET]
    yhat = model.predict(df[X_cols])
    return pd.Series(yhat, index=df.index, dtype=float)

# ----------------------------------------------------------------------
#  UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Silica Prediction — Demo", layout="wide")
st.title("Demo predicción de % Silica (XGB + LSTM + Transformer)")

df = load_data()
base_cols = [c for c in df.columns if c != TARGET]
device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Dispositivo: **{device.upper()}** — filas: {len(df):,} — features: {len(base_cols)}")

# Carga modelos
xgb = None
try:
    xgb = load_xgb()
    st.success("✅ XGBoost cargado.")
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar XGB: {e}")

try:
    lstm, lstm_cols, lstm_L, lstm_H, lstm_scaler = load_lstm(n_features=len(base_cols), device=device)
    st.success(f"✅ LSTM cargado. lookback={lstm_L}, horizon={lstm_H}")
except Exception as e:
    lstm = None; lstm_cols = None; lstm_L = 180; lstm_H = 1; lstm_scaler = None
    st.error(f"No se pudo cargar LSTM: {e}")

try:
    trf, trf_cols, trf_L, trf_H, trf_scaler = load_transformer(n_features=len(base_cols), device=device)
    st.success(f"✅ Transformer cargado. lookback={trf_L}, horizon={trf_H}")
except Exception as e:
    trf = None; trf_cols = None; trf_L = 180; trf_H = 1; trf_scaler = None
    st.error(f"No se pudo cargar Transformer: {e}")

# Parámetros (tabs)
tabs = st.tabs(["General", "Elegir series"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    hist_hours = c1.number_input("Horas a visualizar", min_value=6, max_value=24*14, value=24*3, step=6)
    ens_w_xgb = c2.slider("Peso XGB", 0.0, 1.0, 0.34, 0.01)
    ens_w_lstm = c3.slider("Peso LSTM", 0.0, 1.0, 0.33, 0.01)
    ens_w_trf = c4.slider("Peso Transformer", 0.0, 1.0, 0.33, 0.01)

    c5, c6 = st.columns(2)
    auto_lag = c5.checkbox("Auto-ajustar desfase (±6h)", value=True)
    calib_bias = c6.checkbox("Calibrar offset reciente", value=True)
    st.caption("Los modelos secuenciales usan contexto automáticamente (lookback+1 h previas).")

with tabs[1]:
    show_series = st.multiselect(
        "Elegir series a graficar",
        ["Real", "XGB", "LSTM", "Transformer", "ensemble"],
        default=["Real", "XGB", "Transformer", "ensemble"]
    )

# Normaliza pesos del ensamble
w_sum = max(ens_w_xgb + ens_w_lstm + ens_w_trf, 1e-8)
wx, wl, wt = ens_w_xgb/w_sum, ens_w_lstm/w_sum, ens_w_trf/w_sum

# Ventana a mostrar
end_time = df.index.max()
start_time = end_time - pd.Timedelta(hours=int(hist_hours))
view = df.loc[start_time:end_time].copy()

# Predicciones (con CONTEXTO para LSTM/Transformer)
preds: dict[str, pd.Series] = {}

if xgb is not None:
    try:
        preds["XGB"] = predict_xgb(xgb, view)
    except Exception as e:
        st.warning(f"XGB no predijo: {e}")

if lstm is not None:
    try:
        ctx_start = start_time - pd.Timedelta(hours=int(lstm_L + 1))
        ctx_df = df.loc[ctx_start:end_time]
        pl_ctx = predict_lstm(lstm, ctx_df, lstm_cols, int(lstm_L), lstm_scaler, device,
                              horizon_hours=int(lstm_H))
        preds["LSTM"] = pl_ctx.reindex(view.index)  # sólo lo visible
    except Exception as e:
        st.warning(f"LSTM no predijo: {e}")

if trf is not None:
    try:
        ctx_start = start_time - pd.Timedelta(hours=int(trf_L + 1))
        ctx_df = df.loc[ctx_start:end_time]
        pt_ctx = predict_transformer(trf, ctx_df, trf_cols, int(trf_L), trf_scaler, device,
                                     horizon_hours=int(trf_H))
        preds["Transformer"] = pt_ctx.reindex(view.index)
    except Exception as e:
        st.warning(f"Transformer no predijo: {e}")

# -------- Diagnóstico de desfase (antes de ajustar lag/offset) --------
y = view[TARGET].copy()

# -------- Diagnóstico de desfase (barrido de correlación) --------
def lag_sweep(y_ser: pd.Series, p_ser: pd.Series, max_lag: int = 24):
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for L in lags:
        ps = p_ser.shift(L, freq="H")
        z = pd.concat([y_ser, ps], axis=1).dropna()
        if len(z) < 12:
            corrs.append(np.nan)
        else:
            a = z.iloc[:, 0].to_numpy(dtype=float)
            b = z.iloc[:, 1].to_numpy(dtype=float)
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    best = 0 if not np.isfinite(corrs).any() else int(lags[int(np.nanargmax(corrs))])
    return best, pd.Series(corrs, index=lags, name="corr")

best_lags: dict[str, int] = {}      # <-- DECLARACIÓN AQUÍ

if preds:
    preds_raw = {k: v.copy() for k, v in preds.items() if isinstance(v, pd.Series)}
    rows = []
    for k in ["XGB", "LSTM", "Transformer"]:
        if k in preds_raw:
            L, curve = lag_sweep(y, preds_raw[k], max_lag=24)
            best_lags[k] = L             # <-- LO GUARDAMOS AQUÍ
            arr = curve.to_numpy(dtype=float)
            max_corr = float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
            rows.append({"model": k, "best_lag_h": L, "max_corr": max_corr})
    if rows:
        st.subheader("Diagnóstico de desfase")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(8, 3.2))
        for k in ["XGB", "LSTM", "Transformer"]:
            if k in preds_raw:
                _, curve = lag_sweep(y, preds_raw[k], max_lag=24)
                curve.plot(ax=ax2, label=k, lw=1.4)
        ax2.set_xlabel("Lag (horas)  [positivo = pred atrasada]")
        ax2.set_ylabel("Correlación")
        ax2.set_title("Barrido de correlación vs lag")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)
# -------- Fin diagnóstico --------
# --- Pre-alineación basada en correlación ---
for k, lag in best_lags.items():
    if k in preds and lag != 0:
        preds[k] = preds[k].shift(lag, freq="H").reindex(view.index)
# --------------------------------------------

if preds:
    preds_raw = {k: v.copy() for k, v in preds.items() if isinstance(v, pd.Series)}
    rows = []
    for k in ["XGB", "LSTM", "Transformer"]:
        if k in preds_raw:
            L, curve = lag_sweep(y, preds_raw[k], max_lag=24)
            arr = curve.to_numpy(dtype=float)
            max_corr = float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
            rows.append({"model": k, "best_lag_h": L, "max_corr": max_corr})
    if rows:
        st.subheader("Diagnóstico de desfase")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 3.2))
        for k in ["XGB", "LSTM", "Transformer"]:
            if k in preds_raw:
                _, curve = lag_sweep(y, preds_raw[k], max_lag=24)
                curve.plot(ax=ax2, label=k, lw=1.4)
        ax2.set_xlabel("Lag (horas)  [positivo = pred atrasada]")
        ax2.set_ylabel("Correlación")
        ax2.set_title("Barrido de correlación vs lag")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)
# -------- Fin diagnóstico --------

# Ajustes de diagnóstico (lag/offset)
for name in list(preds.keys()):
    s = preds[name]
    if calib_bias:
        s = remove_bias(y, s)
    if auto_lag:
        L = best_lag(y, s, max_lag=6)
        if L != 0:
            s = s.shift(L, freq="H").reindex(view.index)
    preds[name] = s



# Ensamble y métricas
dfp = pd.DataFrame(index=view.index)
for k, s in preds.items():
    dfp[k] = s.reindex(dfp.index)

if not dfp.empty:
    dfp["ensemble"] = wx*col_or_zeros(dfp, "XGB") + wl*col_or_zeros(dfp, "LSTM") + wt*col_or_zeros(dfp, "Transformer")

    rows = []
    for k in [c for c in ["XGB", "LSTM", "Transformer", "ensemble"] if c in dfp.columns]:
        yk = y.loc[dfp.index].to_numpy(dtype=float)
        pk = dfp[k].to_numpy(dtype=float)
        mask = ~np.isnan(pk)
        if mask.sum() == 0:
            m = dict(rmse=np.nan, mae=np.nan, r2=np.nan)
        else:
            m = metrics(yk[mask], pk[mask])
        rows.append({"model": k, **m})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("Aún no hay predicciones para mostrar (revisa carga de modelos).")

# ---------------- Plot ----------------
colors = {"Real": "tab:blue", "XGB": "tab:orange", "LSTM": "tab:green",
          "Transformer": "tab:red", "ensemble": "tab:purple"}

fig, ax = plt.subplots(figsize=(12, 4))
y.plot(ax=ax, label="Real", lw=1.4, color=colors["Real"])

for k in ["XGB", "LSTM", "Transformer", "ensemble"]:
    if k in dfp.columns and k in show_series:
        s = dfp[k]
        if not s.isna().all():
            s.plot(ax=ax, label=k, lw=1.2, alpha=0.95, color=colors[k])

ax.set_title("Últimas horas — Real vs Predicciones")
ax.set_xlabel("date"); ax.set_ylabel("% Silica Concentrate")
ax.legend(ncol=4)
st.pyplot(fig, clear_figure=True)

# Snapshot descargable
if not dfp.empty:
    snap = pd.concat({"y_true": y, **{k: v for k, v in preds.items()}, "ensemble": dfp.get("ensemble", pd.Series(dtype=float))}, axis=1)
    out = RESULTS_DIR / "last_snapshot.csv"
    snap.to_csv(out, index=True)
    st.success(f"Snapshot guardado en {out}")
    st.download_button("Descargar snapshot CSV", data=snap.to_csv().encode("utf-8"),
                       file_name="last_snapshot.csv", mime="text/csv")
