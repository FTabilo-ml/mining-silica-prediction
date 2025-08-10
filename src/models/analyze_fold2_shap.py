# ---------------- src/models/analyze_fold2_shap.py ----------------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------ config ------------------
DATA_PATH   = Path("data/processed/train.parquet")
RESULTS_DIR = Path("results/fold2")
TARGET      = "% Silica Concentrate"

USE_GPU = True
DEVICE  = "cuda" if USE_GPU else "cpu"

N_SPLITS   = 5
TEST_DAYS  = 30
GAP_HOURS  = 24
FOLD_TO_ANALYZE = 2            # ← fold “difícil”
TOPK = 15                      # top features para gráficos

# ------------------ helpers ------------------
def rolling_origin_splits(index, n_splits=5, test_days=30, gap_hours=24):
    idx = pd.DatetimeIndex(index)
    max_date = idx.max()
    splits = []
    for k in range(n_splits, 0, -1):
        test_end   = max_date - pd.Timedelta(days=(k-1)*test_days)
        test_start = test_end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
        train_end  = test_start - pd.Timedelta(hours=gap_hours)
        tr_mask = idx <= train_end
        te_mask = (idx >= test_start) & (idx <= test_end)
        splits.append((np.where(tr_mask)[0], np.where(te_mask)[0], test_start, test_end))
    return splits

def metrics(y_true, y_pred):
    return dict(
        rmse = mean_squared_error(y_true, y_pred, squared=False),
        mae  = mean_absolute_error(y_true, y_pred),
        r2   = r2_score(y_true, y_pred),
    )

def get_shap_contribs(model: XGBRegressor, X: pd.DataFrame, feature_names):
    """
    Obtiene contribuciones SHAP usando Booster + DMatrix (API válida en XGBoost 2.x).
    Por defecto fuerza CPU para evitar warnings de dispositivos mezclados.
    """
    booster = model.get_booster()
    try:
        booster.set_param({"device": "cuda"})  # usa "cuda" si quieres forzar GPU
    except Exception:
        pass
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    contrib = booster.predict(dtest, pred_contribs=True)
    return contrib  # shape: (n_samples, n_features + 1)  [última col = bias]

# ------------------ main ------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) datos
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET]).astype(np.float32)
    y = df[TARGET]

    # 2) arma los splits y toma el fold target
    splits = rolling_origin_splits(df.index, N_SPLITS, TEST_DAYS, GAP_HOURS)
    tr_idx, te_idx, test_start, test_end = splits[FOLD_TO_ANALYZE - 1]

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    # 3) entrena XGB
    params = dict(
        n_estimators=1100, max_depth=3, learning_rate=0.02,
        subsample=0.90, colsample_bytree=0.88,
        tree_method="hist", device=DEVICE, random_state=42
    )
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    # 4) predicciones y métricas
    y_hat = model.predict(X_te)
    m = metrics(y_te, y_hat)
    pd.DataFrame([{
        "fold": FOLD_TO_ANALYZE, "test_start": test_start, "test_end": test_end, **m
    }]).to_csv(RESULTS_DIR / "fold2_metrics.csv", index=False)
    print("Fold 2 metrics:", m)

    # 5) curva real vs pred (bandas ±2σ de residuales rolling 6h)
    preds = pd.DataFrame({"y_true": y_te.values, "y_pred": y_hat}, index=X_te.index).sort_index()
    preds["residual"]  = preds["y_true"] - preds["y_pred"]
    roll = preds["residual"].rolling(6, min_periods=1).std()
    band_hi = preds["y_pred"] + 2 * roll
    band_lo = preds["y_pred"] - 2 * roll

    plt.figure(figsize=(12,5))
    preds["y_true"].plot(label="Real", alpha=0.8)
    preds["y_pred"].plot(label="Pred", alpha=0.8)
    plt.fill_between(preds.index, band_lo, band_hi, alpha=0.2, label="±2σ resid (6h)")
    plt.title("Fold 2 · Predicción vs Real")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fold2_pred_vs_real.png", dpi=150)
    plt.close()

    preds.to_parquet(RESULTS_DIR / "fold2_preds.parquet")

    # 6) SHAP nativo (contribuciones)
    feature_names = list(X_te.columns)
    contrib = get_shap_contribs(model, X_te, feature_names)

    shap_vals = pd.DataFrame(
        contrib[:, :-1], columns=feature_names, index=X_te.index
    ).astype(float)

    # Importancia global: media del |SHAP|
    imp = shap_vals.abs().mean(axis=0).sort_values(ascending=False)
    imp_df = imp.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})
    imp_df.to_csv(RESULTS_DIR / "fold2_shap_importance.csv", index=False)

    # Top-K barh (tipos amigos de Pylance)
    top = imp.head(TOPK)[::-1]
    labels  = top.index.astype(str).tolist()
    heights = top.to_numpy(dtype=float).tolist()
    plt.figure(figsize=(8, 6))
    plt.barh(labels, heights)
    plt.title(f"Fold 2 · Importancia SHAP (Top {TOPK})")
    plt.xlabel("mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fold2_shap_top_bar.png", dpi=150)
    plt.close()

    # 7) Waterfall simple del peor timestamp (por posición segura)
    worst_pos = int(preds["residual"].abs().to_numpy().argmax())
    worst_ts  = preds.index[worst_pos]
    w_row = pd.Series(
        shap_vals.iloc[worst_pos], index=feature_names, dtype=float
    ).sort_values(ascending=True)

    base_attr = model.get_booster().attr("base_score")
    base = float(base_attr) if base_attr is not None else 0.0
    _ = float(w_row.sum() + base)  # contribución total (no graficada, útil si lo quieres mostrar)

    plt.figure(figsize=(9, 6))
    w_row.tail(TOPK).plot(kind="barh")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(
        f"Fold 2 · SHAP waterfall (top {TOPK})\n"
        f"{worst_ts} — y_true={preds.iloc[worst_pos]['y_true']:.3f} | "
        f"y_pred={preds.iloc[worst_pos]['y_pred']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fold2_shap_waterfall_worst.png", dpi=150)
    plt.close()

    # 8) Drift visual en TOP-5 features
    comp_feats = list(imp.head(5).index)
    for f in comp_feats:
        plt.figure(figsize=(7,4))
        X_tr[f].plot(kind="kde", label="train", alpha=0.8)
        X_te[f].plot(kind="kde", label="test",  alpha=0.8)
        plt.legend(); plt.title(f"Fold 2 · {f} (train vs test)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"fold2_kde_{f.replace(' ', '_')}.png", dpi=150)
        plt.close()

    print("✅ Fold 2 analizado. Artefactos en:", RESULTS_DIR)

if __name__ == "__main__":
    main()
