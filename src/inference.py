from typing import Dict, Any, Optional
import numpy as np

def predict_row(model: Any, feats: Dict[str, float], order: Optional[list[str]]) -> float:
    # Construye X respetando el orden del modelo; si no hay, orden alfabético
    cols = order or sorted(feats.keys())
    X = np.array([[feats.get(c, 0.0) for c in cols]], dtype=float)

    # XGBoost/sklearn -> .predict
    y = model.predict(X)
    try:
        return float(y[0])
    except Exception:
        return float(np.asarray(y).reshape(-1)[0])
