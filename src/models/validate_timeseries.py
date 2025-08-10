# ------------------ src/models/validate_timeseries.py ------------------
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ------------------ config ------------------
DATA_PATH   = Path("data/processed/train.parquet")
BEST_MODEL  = Path("models/best_model.pkl")       # opcional (para tomar params)
RESULTS_DIR = Path("results")
TARGET      = "% Silica Concentrate"

USE_GPU = True
DEVICE  = "cuda" if USE_GPU else "cpu"

N_SPLITS     = 5          # número de ventanas de backtest
TEST_DAYS    = 30         # tamaño de cada test
GAP_HOURS    = 24         # gap para evitar fuga temporal (horas)
EARLY_STOP   = 50         # freno temprano por si aplica

# ------------------ helpers ------------------
def load_base_params():
    """Toma hiperparámetros del best_model si existe; si no, usa buenos defaults."""
    try:
        m = joblib.load(BEST_MODEL)
        params = m.get_params()
        # asegura modo correcto
        params.update(dict(tree_method="hist", device=DEVICE, random_state=42))
        return params
    except Exception:
        return dict(
            n_estimators=1100, max_depth=3, learning_rate=0.02,
            subsample=0.90, colsample_bytree=0.88,
            tree_method="hist", device=DEVICE, random_state=42
        )

def rolling_origin_splits(index, n_splits=5, test_days=30, gap_hours=24):
    """Genera cortes tipo backtesting: cada split testea 30 días, con gap configurable."""
    idx = pd.DatetimeIndex(index)
    max_date = idx.max()
    splits = []
    for k in range(n_splits, 0, -1):
        test_end   = max_date - pd.Timedelta(days=(k-1)*test_days)
        test_start = test_end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
        train_end  = test_start - pd.Timedelta(hours=gap_hours)

        train_mask = idx <= train_end
        test_mask  = (idx >= test_start) & (idx <= test_end)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits

def metrics(y_true, y_pred):
    return dict(
        rmse = mean_squared_error(y_true, y_pred, squared=False),
        mae  = mean_absolute_error(y_true, y_pred),
        r2   = r2_score(y_true, y_pred),
    )

def assign_shift(ts):
    h = ts.hour
    if 0 <= h < 8:   return "Turno A (00-07)"
    if 8 <= h < 16:  return "Turno B (08-15)"
    return "Turno C (16-23)"

def psi(expected, actual, buckets=10):
    """Population Stability Index simple sobre residuales."""
    e = np.asarray(expected)
    a = np.asarray(actual)
    qs = np.percentile(e, np.linspace(0, 100, buckets+1))
    e_counts, _ = np.histogram(e, bins=qs)
    a_counts, _ = np.histogram(a, bins=qs)
    e_perc = e_counts / max(len(e), 1) + 1e-6
    a_perc = a_counts / max(len(a), 1) + 1e-6
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

# ------------------ main ------------------
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1) datos
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2) backtesting temporal (ventanas de 30 días con gap de 24 h)
    splits = rolling_origin_splits(df.index, N_SPLITS, TEST_DAYS, GAP_HOURS)
    base_params = load_base_params()

    fold_rows = []
    preds_all = []  # para juntar todas las predicciones de CV

    for i, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        model = XGBRegressor(**base_params)
        # eval_set ayuda con early stopping si aplica (no siempre corta)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False
        )

        y_hat = model.predict(X_te)
        m = metrics(y_te, y_hat)
        fold_rows.append({"fold": i, **m})

        preds_all.append(pd.DataFrame({
            "date": X_te.index, "y_true": y_te.values, "y_pred": y_hat
        }).set_index("date"))

        print(f"[Fold {i}] RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # 3) guardar métricas por fold y promedio
    cv_df = pd.DataFrame(fold_rows).sort_values("fold")
    avg = cv_df[["rmse","mae","r2"]].mean().to_dict()
    cv_df.loc[len(cv_df)] = {"fold": "mean", **avg}
    cv_path = RESULTS_DIR / "cv_scores.csv"
    cv_df.to_csv(cv_path, index=False)
    print("\nCV guardado en:", cv_path)
    print(cv_df)

    # 4) juntar todas las predicciones de CV
    preds_cv = pd.concat(preds_all).sort_index()
    preds_cv["residual"]   = preds_cv["y_true"] - preds_cv["y_pred"]
    preds_cv["abs_resid"]  = preds_cv["residual"].abs()
    preds_path = RESULTS_DIR / "preds_cv.parquet"
    preds_cv.to_parquet(preds_path)
    print("Predicciones CV guardadas en:", preds_path)

    # 5) curva predicción vs real + bandas de error (rolling 6h sobre residuales)
    roll = preds_cv["residual"].rolling(6, min_periods=1).std()
    band_hi = preds_cv["y_pred"] + 2*roll
    band_lo = preds_cv["y_pred"] - 2*roll

    plt.figure(figsize=(12,5))
    preds_cv["y_true"].plot(label="Real", alpha=0.8)
    preds_cv["y_pred"].plot(label="Pred", alpha=0.8)
    plt.fill_between(preds_cv.index, band_lo, band_hi, alpha=0.2, label="±2σ resid (6h)")
    plt.title("Predicción vs Real (CV concatenado)")
    plt.legend(); plt.tight_layout()
    fig_path = RESULTS_DIR / "pred_vs_real_bands.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Gráfico guardado en:", fig_path)

    # 6) deriva por mes y por turno
# --- Deriva por mes y por turno ---
    drift = preds_cv.copy()

    # Asegura que el índice sea DatetimeIndex (para Pylance y para Pandas)
    if not isinstance(drift.index, pd.DatetimeIndex):
        drift.index = pd.to_datetime(drift.index)

    idx_dt = pd.DatetimeIndex(drift.index)

    # Opción 1: mantener to_period (mensual)
    drift["month"] = idx_dt.to_period("M").astype(str)

    # Opción 2 (equivalente y aún más “amigable” con linters):
    # drift["month"] = idx_dt.strftime("%Y-%m")

    drift["shift"] = [assign_shift(ts) for ts in idx_dt]


    monthly = drift.groupby("month").agg(
        y_true_mean=("y_true","mean"),
        y_pred_mean=("y_pred","mean"),
        rmse=("residual", lambda r: float(np.sqrt(np.mean(np.square(r))))),
        mae=("abs_resid","mean"),
        resid_mean=("residual","mean"),
        resid_std=("residual","std")
    ).reset_index()
    monthly.to_csv(RESULTS_DIR / "drift_by_month.csv", index=False)

    # PSI de residuales vs primer mes como baseline
    base_month = monthly["month"].iloc[0]
    psi_rows = []
    base_res = drift.loc[drift["month"] == base_month, "residual"]
    for m in monthly["month"]:
        cur_res = drift.loc[drift["month"] == m, "residual"]
        psi_rows.append({"month": m, "psi_residuals": psi(base_res, cur_res)})
    pd.DataFrame(psi_rows).to_csv(RESULTS_DIR / "psi_residuals_by_month.csv", index=False)

    # turno
    by_shift = drift.groupby("shift").agg(
        rmse=("residual", lambda r: float(np.sqrt(np.mean(np.square(r))))),
        mae=("abs_resid","mean"),
        resid_mean=("residual","mean")
    ).reset_index()
    by_shift.to_csv(RESULTS_DIR / "drift_by_shift.csv", index=False)

    # plotting drift (MAE por mes)
    plt.figure(figsize=(12,4))
    monthly.plot(x="month", y="mae", kind="bar", legend=False)
    plt.title("MAE por mes (deriva)")
    plt.ylabel("MAE"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drift_mae_by_month.png", dpi=150)
    plt.close()

    # plotting drift (MAE por turno)
    plt.figure(figsize=(7,4))
    by_shift.plot(x="shift", y="mae", kind="bar", legend=False)
    plt.title("MAE por turno")
    plt.ylabel("MAE"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drift_mae_by_shift.png", dpi=150)
    plt.close()

    print("✅ Validación temporal y análisis de deriva completados.")

if __name__ == "__main__":
    main()
