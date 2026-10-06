"""Loads trained artifacts once and runs prediction + SHAP. Nothing is hardcoded: every number comes from the models."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from backend.ml.explain_core import build_explainer, shap_values

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
# Presentation bands only (NOT clinical cut-offs): used for colouring/labels in the UI.
LOW_MAX, MODERATE_MAX = 0.30, 0.60


def risk_category(p: float) -> str:
    return "LOW" if p < LOW_MAX else ("MODERATE" if p < MODERATE_MAX else "HIGH")


class ModelStore:
    def __init__(self):
        self.metrics = json.load(open(ROOT / "metrics.json"))
        self.global_importance = json.load(open(ROOT / "models" / "global_importance.json"))
        bg = joblib.load(ROOT / "models" / "shap_background.joblib")
        self.bundles, self.explainers = {}, {}
        for t in TARGETS:
            b = joblib.load(ROOT / "models" / f"{t.lower()}_model.joblib")
            self.bundles[t] = b
            self.explainers[t] = build_explainer(b["pipeline"], bg)
        self.features = self.bundles["CAD"]["features"]
        assert all(self.bundles[t]["features"] == self.features for t in TARGETS)

    def predict(self, values: dict, top_k: int = 8) -> dict:
        row = {f: (values.get(f) if values.get(f) is not None else np.nan) for f in self.features}
        X = pd.DataFrame([row], columns=self.features)
        missing = [f for f in self.features if np.isnan(row[f])]
        preds, expl = {}, {}
        for t in TARGETS:
            b = self.bundles[t]; pipe = b["pipeline"]
            p = float(pipe.predict_proba(X)[0, 1])
            preds[t] = dict(probability=round(p, 4), risk_category=risk_category(p),
                            predicted_positive=bool(p >= b["threshold"]), decision_threshold=round(b["threshold"], 3),
                            model=b["model_name"],
                            cv_roc_auc=self.metrics["targets"][t]["model_comparison_nested_cv"][b["model_name"]]["cv_roc_auc_mean"])
            ex, unit, linear = self.explainers[t]
            sv, base, imputed = shap_values(ex, pipe, X, linear)
            order = np.argsort(-np.abs(sv[0]))[:top_k]
            expl[t] = dict(unit=unit, base_value=round(base, 4), top_features=[
                dict(feature=self.features[i], value=float(imputed.iloc[0, i]), imputed=self.features[i] in missing,
                     shap_value=round(float(sv[0, i]), 4),
                     direction="increases_risk" if sv[0, i] > 0 else "decreases_risk") for i in order])
        return dict(cad=preds["CAD"], vessels={k: preds[k] for k in ["LAD", "LCX", "RCA"]}, explanations=expl,
                    input_summary=dict(n_features_total=len(self.features), n_provided=len(self.features) - len(missing),
                                       n_imputed=len(missing), imputed_features=missing))
