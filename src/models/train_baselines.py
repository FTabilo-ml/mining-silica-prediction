# ----------------------  src/models/train_baselines.py  ---------------------
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------- #
DATA_PATH   = Path("data/processed/train.parquet")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
TARGET      = "% Silica Concentrate"

USE_GPU = True                     # ← cambia a False para CPU
DEVICE  = "cuda" if USE_GPU else "cpu"

# --------------------------------------------------------------------------- #
# UTILIDADES
# --------------------------------------------------------------------------- #
def train_val_test_split(df, test_days=30, val_days=30):
    test_cut = df.index.max() - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)
    train = df.loc[:val_cut]
    val   = df.loc[val_cut + pd.Timedelta(hours=1): test_cut]
    test  = df.loc[test_cut + pd.Timedelta(hours=1):]
    return train, val, test

def metrics(y_true, y_pred):
    return (
        mean_squared_error(y_true, y_pred, squared=False),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )

def save_results(rows, csv_path):
    out = pd.DataFrame(rows, columns=["model", "rmse", "mae", "r2"])
    csv_path.parent.mkdir(exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(out)

# --------------------------------------------------------------------------- #
# 1 · CARGA DE DATOS
# --------------------------------------------------------------------------- #
df = pd.read_parquet(DATA_PATH)
train, val, test = train_val_test_split(df)

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val,   y_val   = val.drop(columns=[TARGET]),   val[TARGET]
X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

rows = []

# --------------------------------------------------------------------------- #
# 2 · BASELINES
# --------------------------------------------------------------------------- #
rows.append(("naive_last",    *metrics(y_test, y_test.shift(1).bfill())))
rows.append(("moving_avg_3h", *metrics(y_test, y_test.rolling(3).mean().shift(1).bfill())))

# --------------------------------------------------------------------------- #
# 3 · RANDOM FOREST (CPU, subsample 5%)
# --------------------------------------------------------------------------- #
print("· Optuna tuning Random Forest… (subsample 5 %)")

sample_frac = 0.05
idx_small   = X_train.sample(frac=sample_frac, random_state=42).index
X_small, y_small = X_train.loc[idx_small], y_train.loc[idx_small]

def rf_objective(trial):
    rf = RandomForestRegressor(
        n_estimators      = trial.suggest_int("n_estimators", 100, 400),
        max_depth         = trial.suggest_int("max_depth",     5,  15),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8),
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_small, y_small)
    pred = rf.predict(X_val)
    return mean_squared_error(y_val, pred, squared=False)

study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(rf_objective, n_trials=25, show_progress_bar=False)

# --- Construye params sin duplicar n_estimators
rf_params = study_rf.best_params.copy()
rf_params["n_estimators"] = rf_params["n_estimators"] * 2

best_rf = RandomForestRegressor(
    **rf_params,
    random_state=42,
    n_jobs=-1,
)
best_rf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
rows.append(("RandomForest", *metrics(y_test, best_rf.predict(X_test))))

# --------------------------------------------------------------------------- #
# 4 · XGBOOST  (GPU o CPU según flag)
# --------------------------------------------------------------------------- #
print(f"· Optuna tuning XGBoost on {DEVICE.upper()}…")

def xgb_objective(trial):
    model = XGBRegressor(
        n_estimators     = trial.suggest_int("n_estimators", 400, 1200),
        max_depth        = trial.suggest_int("max_depth",     3,   10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample",        0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state=42,
        tree_method="hist",
        device=DEVICE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return mean_squared_error(y_val, model.predict(X_val), squared=False)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=False)

best_xgb = XGBRegressor(
    **study_xgb.best_params,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device=DEVICE,
)
best_xgb.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), verbose=False)
rows.append(("XGBoost", *metrics(y_test, best_xgb.predict(X_test))))

# --------------------------------------------------------------------------- #
# 5 · GUARDAR MÉTRICAS Y MODELO
# --------------------------------------------------------------------------- #
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

save_results(rows, RESULTS_DIR / "model_compare.csv")
joblib.dump(best_xgb, MODELS_DIR / "best_model.pkl")

print("🎉 Entrenamiento base completado — dispositivo XGB:", DEVICE)
