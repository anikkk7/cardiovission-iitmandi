"""Offline: save SHAP background data + global feature importance so the API
does not need the raw dataset at runtime. Uses TRAINING rows only."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.ml.data import load_raw, build_features, ROOT
from backend.ml.explain_core import build_explainer, shap_values

X, _ = build_features(load_raw())
split = json.load(open(ROOT / "models" / "split_indices.json"))
train_ids = split["CAD"]["train"]
assert train_ids == split["LAD"]["train"] == split["RCA"]["train"], "targets must share one split"
Xtr = X.loc[train_ids]
joblib.dump(Xtr, ROOT / "models" / "shap_background.joblib")

out = {}
for t in ["CAD", "LAD", "LCX", "RCA"]:
    bundle = joblib.load(ROOT / "models" / f"{t.lower()}_model.joblib")
    ex, unit, linear = build_explainer(bundle["pipeline"], Xtr)
    sv, base, _ = shap_values(ex, bundle["pipeline"], Xtr, linear)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    out[t] = {"model": bundle["model_name"], "unit": unit, "n_samples": len(Xtr),
              "features": [{"feature": X.columns[i], "mean_abs_shap": round(float(imp[i]), 5)} for i in order]}
    print(t, bundle["model_name"], unit, [f["feature"] for f in out[t]["features"][:5]])
json.dump(out, open(ROOT / "models" / "global_importance.json", "w"), indent=2)
