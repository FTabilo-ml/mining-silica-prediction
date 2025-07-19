# src/features/make_features.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ---------- Parámetros de ingeniería ----------
LAGS = [1, 3, 6, 12]           # horas
ROLL_WINDOWS = [3, 6]          # horas
TOP_LAG_VARS = [
    "Amina Flow", "% Silica Feed", "Ore Pulp Density",
    "% Iron Feed", "Ore Pulp pH"
]


# ---------- Funciones auxiliares ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hora del día en forma cíclica."""
    hours = df.index.hour
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
        },
        index=df.index,
    )


def build_lag_frames(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Devuelve una lista de DataFrames con los lags especificados."""
    return [df.shift(lag).add_suffix(f"_lag{lag}") for lag in LAGS]


def build_roll_frames(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Media y desviación de ventanas móviles para TOP_LAG_VARS."""
    frames = []
    for w in ROLL_WINDOWS:
        rolled = (
            df[TOP_LAG_VARS]
            .rolling(window=w, min_periods=1)
            .agg(["mean", "std"])
        )
        rolled.columns = [f"{var}_{stat}_w{w}h" for var, stat in rolled.columns]
        frames.append(rolled)
    return frames


def make_features(in_csv: Path, out_parquet: Path) -> None:
    # 1. Carga eficiente (float32)
    df = pd.read_csv(
        in_csv,
        sep=",",
        decimal=",",
        quotechar='"',
        parse_dates=["date"],
        index_col="date",
        dtype="float32",
        engine="python",
    ).sort_index()

    # 2. Construye todas las piezas en memoria
    frames = [
        df,                        # originales
        add_time_features(df),     # hora_sin/cos
        *build_lag_frames(df),     # lags
        *build_roll_frames(df),    # rolling windows
    ]

    # 3. Concatena una sola vez y elimina filas con NaNs (por lags)
    df_feat = pd.concat(frames, axis=1).dropna().astype("float32")

    # 4. Guarda en parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_parquet)
    print(f"✅ Features guardadas en {out_parquet}  →  shape {df_feat.shape}")


# ---------- Ejecución por CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera matriz de features")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/MiningProcess_Flotation_Plant_Database.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train.parquet"),
    )
    # parse_known_args para ignorar flags de Jupyter (--f=kernel‑uuid.json)
    args, _ = parser.parse_known_args()
    make_features(args.input, args.output)
