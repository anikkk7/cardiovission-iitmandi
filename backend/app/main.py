"""CardioVision AI API.   Run from repo root:  uvicorn backend.app.main:app --reload --port 8000"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import PatientInput, PredictResponse, META, DISCLAIMER, VIS_NOTE
from backend.app.store import ModelStore, LOW_MAX, MODERATE_MAX

state = {}


@asynccontextmanager
async def lifespan(app):
    state["store"] = ModelStore()
    yield


app = FastAPI(title="CardioVision AI", version="0.2.0", lifespan=lifespan,
              description="Clinical decision-support prototype (research/education only). " + VIS_NOTE)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    s = state.get("store")
    return {"status": "ok" if s else "loading", "models_loaded": list(s.bundles) if s else [],
            "n_features": len(s.features) if s else 0}


@app.get("/model-metrics")
def model_metrics():
    """Held-out + nested-CV metrics exactly as produced by training (metrics.json), plus global SHAP importance."""
    s = state["store"]
    return {"metrics": s.metrics, "global_importance": s.global_importance,
            "risk_bands": {"low_below": LOW_MAX, "high_from": MODERATE_MAX,
                           "note": "Presentation bands for colouring only; not clinical cut-offs."},
            "disclaimer": DISCLAIMER}


@app.get("/feature-schema")
def feature_schema():
    return {"features": META["features"], "notes": META["notes"]}


@app.get("/demo-patients")
def demo_patients():
    """Real dataset rows that were held out from training for ALL four models."""
    return META["demo_patients"]


@app.post("/predict", response_model=PredictResponse)
def predict(patient: PatientInput, top_k: int = Query(8, ge=1, le=20)):
    try:
        return state["store"].predict(patient.features, top_k=top_k)
    except Exception as e:  # surface model errors clearly instead of a bare 500
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")
