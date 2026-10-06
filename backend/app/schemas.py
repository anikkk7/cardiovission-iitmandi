"""Pydantic models + input validation driven by feature_metadata.json (real dataset columns only)."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
META = json.load(open(ROOT / "feature_metadata.json"))
FEATURES = {f["name"]: f for f in META["features"]}
DISCLAIMER = ("Clinical decision-support prototype. Predictions are for educational/research purposes only and are "
              "NOT a diagnosis or a substitute for professional medical evaluation or diagnostic imaging.")
VIS_NOTE = ("The 3D visualization represents predicted vessel-level risk and does not depict an actual anatomical lesion.")


def _bounds(f):
    """Generous plausibility bounds: dataset range padded by 50% of its span (never below 0 for non-negative features)."""
    span = f["max"] - f["min"]
    lo, hi = f["min"] - 0.5 * span, f["max"] + 0.5 * span
    return (max(lo, 0.0) if f["min"] >= 0 else lo), hi


class PatientInput(BaseModel):
    """Any subset of the model's features. Omitted/null values are imputed (median of training data) and reported back."""
    features: Dict[str, Optional[float]] = Field(..., description="feature name -> value (see GET /feature-schema)")

    @model_validator(mode="after")
    def check(self):
        errors = []
        unknown = [k for k in self.features if k not in FEATURES]
        if unknown:
            errors.append(f"unknown feature(s): {unknown}")
        for k, v in self.features.items():
            f = FEATURES.get(k)
            if f is None or v is None:
                continue
            if f["type"] == "binary" and v not in (0, 1):
                errors.append(f"{k}: must be 0 or 1")
            elif f["type"] == "ordinal" and (v != int(v) or not (f["min"] <= v <= f["max"])):
                errors.append(f"{k}: must be an integer between {int(f['min'])} and {int(f['max'])}")
            elif f["type"] == "numeric":
                lo, hi = _bounds(f)
                if not (lo <= v <= hi):
                    errors.append(f"{k}: {v} outside plausible range [{lo:.4g}, {hi:.4g}]")
        if not any(v is not None for v in self.features.values()):
            errors.append("at least one feature value is required")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class TargetPrediction(BaseModel):
    probability: float
    risk_category: str
    predicted_positive: bool
    decision_threshold: float
    model: str
    cv_roc_auc: float = Field(description="nested-CV ROC-AUC of this model: how far to trust it")


class FeatureContribution(BaseModel):
    feature: str
    value: float
    imputed: bool
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class Explanation(BaseModel):
    unit: str
    base_value: float
    top_features: List[FeatureContribution]


class InputSummary(BaseModel):
    n_features_total: int
    n_provided: int
    n_imputed: int
    imputed_features: List[str]


class PredictResponse(BaseModel):
    cad: TargetPrediction
    vessels: Dict[str, TargetPrediction]
    explanations: Dict[str, Explanation]
    input_summary: InputSummary
    disclaimer: str = DISCLAIMER
    visualization_note: str = VIS_NOTE
