# src/main.py
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .settings import settings
from .schemas import Features, BatchFeatures
from .model_loader import load_model, ModelNotFound, get_feature_order
from .inference import predict_row

# (Opcional) escritura a Blob si tienes db.append_prediction_to_blob
try:
    from .db import append_prediction_to_blob  # type: ignore
except Exception:  # pragma: no cover
    append_prediction_to_blob = None  # desactivado si no existe

# ---------- Logging estructurado ----------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("uvicorn.error")

def log_json(event: str, **kwargs):
    logging.getLogger("uvicorn.access").info(json.dumps({"event": event, **kwargs}))

# ---------- App ----------
app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# CORS abierto (ajusta orígenes en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limitar tamaño de payload (1 MB)
MAX_BYTES = 1_000_000

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    try:
        if cl and int(cl) > MAX_BYTES:
            return JSONResponse({"detail": "Payload too large"}, status_code=413)
    except Exception:
        pass
    return await call_next(request)

# ---------- Estado de modelo ----------
MODEL = None
MODEL_KIND = settings.MODEL_KIND  # "xgb" por defecto
FEATURE_ORDER: Optional[list[str]] = None
STARTUP_TS = time.time()

@app.on_event("startup")
def _startup():
    global MODEL, FEATURE_ORDER
    MODEL = load_model(MODEL_KIND)
    FEATURE_ORDER = get_feature_order(MODEL)
    log.info(
        "✅ Modelo cargado",
        extra={"model_kind": MODEL_KIND, "n_features": len(FEATURE_ORDER) if FEATURE_ORDER else None},
    )

# ---------- Endpoints básicos ----------
@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTUP_TS, 2)}

@app.get("/ready")
def ready():
    ok = MODEL is not None
    return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

@app.get("/version")
def version():
    return {
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "model_kind": MODEL_KIND,
        "n_features": len(FEATURE_ORDER) if FEATURE_ORDER else None,
        "features": FEATURE_ORDER,
        "version": "1.0.0",
    }

@app.get("/schema")
def schema():
    return {
        "model_kind": MODEL_KIND,
        "required_features": FEATURE_ORDER or [],
        "count": len(FEATURE_ORDER) if FEATURE_ORDER else 0,
    }

# ---------- Predicción (una fila) ----------
@app.post("/predict")
def predict(payload: Features, model: Optional[str] = Query(None)):
    global MODEL, MODEL_KIND, FEATURE_ORDER
    try:
        # Soporta cambio de modelo via query param (?model=xgb|lstm|transformer)
        if model and model.lower() != MODEL_KIND:
            MODEL = load_model(model.lower())
            MODEL_KIND = model.lower()
            FEATURE_ORDER = get_feature_order(MODEL)

        feats = payload.as_dict
        if not feats:
            raise HTTPException(422, "Body debe ser un JSON con {feature: valor}")

        t0 = time.perf_counter()
        pred = predict_row(MODEL, feats, FEATURE_ORDER)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        missing = []
        if FEATURE_ORDER:
            missing = [c for c in FEATURE_ORDER if c not in feats]

        # (Opcional) Guardar registro en Blob si hay función/config
        if append_prediction_to_blob:
            try:
                record = {
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "prediction": float(pred),
                    "model_kind": MODEL_KIND,
                    "elapsed_ms": elapsed_ms,
                }
                append_prediction_to_blob(record)  # type: ignore
            except Exception as e:  # no rompas la predicción por fallar el log
                log.warning(f"No se pudo escribir en Blob: {e}")

        log_json("predict_ok", model=MODEL_KIND, elapsed_ms=elapsed_ms)
        return {
            "prediction": float(pred),
            "model_kind": MODEL_KIND,
            "elapsed_ms": elapsed_ms,
            "n_features": len(FEATURE_ORDER) if FEATURE_ORDER else None,
            "missing_features": missing[:20],  # muestra primeras 20 faltantes
        }

    except HTTPException:
        raise
    except ModelNotFound as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error("❌ Error en /predict: %s", e, exc_info=True)
        raise HTTPException(500, f"Prediction error: {type(e).__name__}: {e}")

# ---------- Predicción por lotes ----------
@app.post("/predict/batch")
def predict_batch_api(payload: BatchFeatures, model: Optional[str] = Query(None)):
    global MODEL, MODEL_KIND, FEATURE_ORDER
    try:
        if model and model.lower() != MODEL_KIND:
            MODEL = load_model(model.lower())
            MODEL_KIND = model.lower()
            FEATURE_ORDER = get_feature_order(MODEL)

        rows = payload.rows or []
        if not rows:
            raise HTTPException(422, "rows vacío")

        t0 = time.perf_counter()
        preds = [float(predict_row(MODEL, r, FEATURE_ORDER)) for r in rows]
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        log_json("predict_batch_ok", model=MODEL_KIND, count=len(preds), elapsed_ms=elapsed_ms)
        return {
            "predictions": preds,
            "count": len(preds),
            "model_kind": MODEL_KIND,
            "elapsed_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except ModelNotFound as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error("❌ Error en /predict/batch: %s", e, exc_info=True)
        raise HTTPException(500, f"Prediction error: {type(e).__name__}: {e}")
