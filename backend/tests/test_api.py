import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from fastapi.testclient import TestClient
from backend.app.main import app

ROOT = Path(__file__).resolve().parents[2]
META = json.load(open(ROOT / "feature_metadata.json"))


def test_all():
    with TestClient(app) as c:
        h = c.get("/health").json(); assert h["status"] == "ok" and len(h["models_loaded"]) == 4
        m = c.get("/model-metrics").json(); assert set(m["metrics"]["targets"]) == {"CAD", "LAD", "LCX", "RCA"}
        demo = META["demo_patients"]
        out = {}
        for k, d in demo.items():
            r = c.post("/predict", json={"features": d["features"]}); assert r.status_code == 200, r.text
            out[k] = r.json()
            # probability must equal the raw model output (no hardcoding / post-hoc editing)
            for t in ["CAD", "LAD", "LCX", "RCA"]:
                b = joblib.load(ROOT / "models" / f"{t.lower()}_model.joblib")
                X = pd.DataFrame([d["features"]])[b["features"]]
                p = float(b["pipeline"].predict_proba(X)[0, 1])
                got = out[k]["cad"] if t == "CAD" else out[k]["vessels"][t]
                assert abs(got["probability"] - p) < 1e-4, (t, got["probability"], p)
            assert not set(r.json()["explanations"]["CAD"]["top_features"][0]) - {"feature", "value", "imputed", "shap_value", "direction"}
        assert out["high_risk_example"]["cad"]["probability"] > out["low_risk_example"]["cad"]["probability"]
        # validation
        assert c.post("/predict", json={"features": {"LAD": 1}}).status_code == 422       # leakage col rejected
        assert c.post("/predict", json={"features": {"Age": 400}}).status_code == 422     # implausible
        assert c.post("/predict", json={"features": {"HTN": 2}}).status_code == 422       # not binary
        assert c.post("/predict", json={"features": {}}).status_code == 422
        partial = c.post("/predict", json={"features": {"Age": 60, "Sex": 1}}).json()
        assert partial["input_summary"]["n_provided"] == 2 and partial["input_summary"]["n_imputed"] == 53
        print(json.dumps({k: {"CAD": v["cad"]["probability"], **{t: x["probability"] for t, x in v["vessels"].items()}} for k, v in out.items()}, indent=1))


def test_shap_additivity():
    """base_value + sum(all SHAP values) must reproduce the model output (log-odds for linear, probability for RF)."""
    from backend.app.store import ModelStore
    st = ModelStore()
    from backend.ml.explain_core import shap_values
    feats = META["demo_patients"]["high_risk_example"]["features"]
    X = pd.DataFrame([feats])[st.features]
    for t in ["CAD", "LAD", "LCX", "RCA"]:
        ex, unit, linear = st.explainers[t]; pipe = st.bundles[t]["pipeline"]
        sv, base, _ = shap_values(ex, pipe, X, linear)
        p = float(pipe.predict_proba(X)[0, 1])
        recon = base + sv.sum()
        target = np.log(p / (1 - p)) if unit == "log-odds" else p
        assert abs(recon - target) < 5e-3, (t, unit, recon, target)
