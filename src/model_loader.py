import os
import joblib
from typing import Any, Optional
from .settings import settings

# Importar torch si está disponible (para LSTM/Transformer)
try:
    import torch
except Exception:
    torch = None


class ModelNotFound(Exception):
    """Se lanza cuando la ruta de modelo no existe."""
    pass


def _ensure(path: str) -> None:
    if not os.path.exists(path):
        raise ModelNotFound(f"Modelo no encontrado en: {path}")


def load_model(kind: Optional[str] = None) -> Any:
    """
    Carga el modelo según el tipo: 'xgb', 'lstm' o 'transformer'.
    Si kind es None, usa settings.MODEL_KIND.
    """
    k = (kind or settings.MODEL_KIND).lower()

    if k == "xgb":
        _ensure(settings.MODEL_XGB_PATH)
        return joblib.load(settings.MODEL_XGB_PATH)

    if k == "lstm":
        if torch is None:
            raise RuntimeError("PyTorch no instalado para LSTM")
        _ensure(settings.MODEL_LSTM_PATH)
        m = torch.load(settings.MODEL_LSTM_PATH, map_location="cpu")
        try:
            m.eval()
        except Exception:
            pass
        return m

    if k == "transformer":
        if torch is None:
            raise RuntimeError("PyTorch no instalado para Transformer")
        _ensure(settings.MODEL_TRANSF_PATH)
        m = torch.load(settings.MODEL_TRANSF_PATH, map_location="cpu")
        try:
            m.eval()
        except Exception:
            pass
        return m

    raise ValueError(f"MODEL_KIND no soportado: {k}")


def get_feature_order(model) -> Optional[list[str]]:
    """
    Obtiene el orden de features esperado por el modelo, si está disponible.
    - Para sklearn/xgboost con API sklearn: feature_names_in_
    - Para XGBoost booster nativo: booster.feature_names
    """
    # sklearn / xgboost (API sklearn)
    if hasattr(model, "feature_names_in_"):
        try:
            return list(getattr(model, "feature_names_in_"))
        except Exception:
            pass

    # XGBoost booster nativo
    booster = None
    try:
        get_booster = getattr(model, "get_booster", None)
        if callable(get_booster):
            booster = get_booster()
        if booster is not None and hasattr(booster, "feature_names"):
            names = getattr(booster, "feature_names")
            if names is not None:
                return list(names)
    except Exception:
        pass

    return None
