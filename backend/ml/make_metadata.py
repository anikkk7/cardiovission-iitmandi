"""Build feature_metadata.json from the real dataset (no invented columns)."""
import json, sys
from pathlib import Path
import joblib, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_raw, build_features, build_targets, ROOT, BINARY_YN, VHD_ORDER

raw = load_raw(); X, const = build_features(raw); Y = build_targets(raw)
binary = [c for c in X.columns if set(X[c].unique()) <= {0, 1}]
ordinal = {"VHD": VHD_ORDER}
feats = []
for c in X.columns:
    kind = "binary" if c in binary else ("ordinal" if c in ordinal or X[c].nunique() <= 5 else "numeric")
    f = {"name": c, "type": kind, "min": float(X[c].min()), "max": float(X[c].max()),
         "median": float(X[c].median()), "n_unique": int(X[c].nunique())}
    if kind == "ordinal" and c in ordinal: f["encoding"] = ordinal[c]
    if c == "Sex": f["encoding"] = {"Female": 0, "Male": 1}
    feats.append(f)

split = json.load(open(ROOT / "models" / "split_indices.json"))
test_ids = set(split["CAD"]["test"]) & set(split["LAD"]["test"]) & set(split["LCX"]["test"]) & set(split["RCA"]["test"])
# Per-target splits are independent; keep only patients that were held out for ALL four models.
demo_pool = Y.loc[sorted(test_ids)]
demos = {}
if len(demo_pool):
    hi = demo_pool[(demo_pool.CAD == 1) & (demo_pool.LAD == 1)].index.tolist()
    lo = demo_pool[demo_pool.sum(axis=1) == 0].index.tolist()
    for label, ids in [("high_risk_example", hi), ("low_risk_example", lo)]:
        if ids:
            i = ids[0]; demos[label] = {"dataset_row": int(i), "features": {k: float(v) for k, v in X.loc[i].items()},
                                       "true_labels_for_reference_only": {k: int(v) for k, v in Y.loc[i].items()}}
meta = {"n_patients": len(X), "features": feats, "dropped_constant_features": const,
        "leakage_columns_excluded": ["Cath", "LAD", "LCX", "RCA"], "targets": list(Y.columns),
        "target_prevalence": {k: round(float(Y[k].mean()), 3) for k in Y.columns},
        "heldout_for_all_models_row_count": len(test_ids), "demo_patients": demos,
        "notes": ["Sex encoded 1=Male (dataset spells female 'Fmale').", "BBB split into LBBB/RBBB indicators.",
                  "VHD ordinal: N=0, mild=1, Moderate=2, Severe=3.",
                  "Dataset has no total-cholesterol column; lipids are LDL, HDL, TG."]}
json.dump(meta, open(ROOT / "feature_metadata.json", "w"), indent=2)
print(len(feats), "features; held-out-for-all:", len(test_ids), "; demos:", {k: v["dataset_row"] for k, v in demos.items()})
# sanity: shipped models contain no leakage columns
for t in ["cad", "lad", "lcx", "rca"]:
    m = joblib.load(ROOT / "models" / f"{t}_model.joblib")
    assert not {"Cath", "LAD", "LCX", "RCA"} & set(m["features"])
print("leakage check passed for all 4 models")
