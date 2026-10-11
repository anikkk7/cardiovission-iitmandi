"""SHAP helpers shared by the offline artifact builder and the API.

Linear models  -> shap.LinearExplainer on the scaled space; values are in log-odds.
Tree models    -> shap.TreeExplainer; values are in the model's raw output
                  (probability for RandomForest, log-odds for boosted trees).
The `unit` is always returned so the UI never mixes scales silently.
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression


def _avoid_base_python_module_shadowing():
    """Keep stray files in the base Python folder from shadowing installed packages."""
    cleaned = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        path = Path(entry)
        try:
            resolved = path.resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if resolved == Path(sys.base_prefix).resolve() and (resolved / "matplotlib.py").exists():
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


_avoid_base_python_module_shadowing()
import shap


def split_pipeline(pipe):
    """Return (preprocessing steps pipeline or None, final classifier)."""
    clf = pipe.steps[-1][1]
    pre = pipe[:-1] if len(pipe.steps) > 1 else None
    return pre, clf


def build_explainer(pipe, background: pd.DataFrame):
    pre, clf = split_pipeline(pipe)
    if isinstance(clf, LogisticRegression):
        bg = pre.transform(background)
        masker = shap.maskers.Independent(bg, max_samples=len(bg))  # use ALL background rows, deterministic
        return shap.LinearExplainer(clf, masker), "log-odds", True
    unit = "probability" if clf.__class__.__name__ == "RandomForestClassifier" else "log-odds"
    return shap.TreeExplainer(clf), unit, False


def shap_values(explainer, pipe, X: pd.DataFrame, linear: bool):
    """X: raw-feature frame (may contain NaN). Returns (values[n, f], base_value, imputed_frame)."""
    imputer = pipe.named_steps["impute"]
    imputed = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    pre, _ = split_pipeline(pipe)
    matrix = pre.transform(X)  # imputed (+scaled for linear)
    sv = explainer.shap_values(matrix)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, f, classes)
        sv = sv[:, :, 1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
    return sv, base, imputed
