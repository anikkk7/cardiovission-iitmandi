"""Stage 1 training: nested CV model comparison + final held-out evaluation.

Per target (CAD, LAD, LCX, RCA):
  1. Shared stratified 80/20 train/test split (same patients for all 4 targets) (test is untouched until the very end).
  2. Compare models with NESTED stratified CV on the training set only
     (inner GridSearchCV for tuning, outer folds for an unbiased estimate).
  3. Pick the best model by outer-CV ROC-AUC (recall as tie-breaker).
  4. Re-tune on the full training set, choose a recall-oriented decision
     threshold from out-of-fold training predictions, evaluate once on test.
"""
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, fbeta_score)
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, train_test_split,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_raw, build_features, build_targets, ROOT

warnings.filterwarnings("ignore")
SEED = 42
MODELS_DIR = ROOT / "models"; MODELS_DIR.mkdir(exist_ok=True)


def candidates():
    def pipe(est, scale):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale: steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("clf", est)])
    return {
        "logreg": (pipe(LogisticRegression(max_iter=5000, class_weight="balanced"), True),
                   {"clf__C": [0.03, 0.1, 0.3, 1.0]}),
        "random_forest": (pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                      random_state=SEED, n_jobs=1), False),
                          {"clf__max_depth": [3, 6], "clf__min_samples_leaf": [2, 5]}),
        "gradient_boosting": (pipe(GradientBoostingClassifier(n_estimators=150, random_state=SEED), False),
                              {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]}),
        "xgboost": (pipe(XGBClassifier(n_estimators=150, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0), False),
                    {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]}),
    }


def metrics_at(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(threshold=round(float(thr), 4), accuracy=accuracy_score(y, yh),
                precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh),
                f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p),
                specificity=tn / (tn + fp), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def pick_threshold(y, p):
    """Recall-oriented: maximise F2 on OUT-OF-FOLD training predictions (never test)."""
    grid = np.linspace(0.2, 0.8, 61)
    return float(grid[int(np.argmax([fbeta_score(y, (p >= t).astype(int), beta=2) for t in grid]))])


def r(d): return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def main():
    raw = load_raw(); X, const_dropped = build_features(raw); Y = build_targets(raw)
    print(f"features={X.shape[1]} dropped_constant={const_dropped}")
    all_metrics = {"seed": SEED, "n_patients": len(X), "n_features": X.shape[1],
                   "dropped_constant_features": const_dropped,
                   "leakage_columns_excluded": ["Cath", "LAD", "LCX", "RCA"], "targets": {}}
    # ONE shared split for all four models (stratified on CAD + number of stenotic vessels)
    strat = Y["CAD"].astype(str) + "_" + (Y["LAD"] + Y["LCX"] + Y["RCA"]).astype(str)
    strat = strat.replace({"0_1": "0_0"})  # 1 patient: Cath normal but LAD labelled stenotic (singleton stratum)
    tr_idx, te_idx = train_test_split(X.index, test_size=0.2, stratify=strat, random_state=SEED)
    split_idx = {}
    for target in ["CAD", "LAD", "LCX", "RCA"]:
        y = Y[target]
        Xtr, Xte, ytr, yte = X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        split_idx[target] = {"train": Xtr.index.tolist(), "test": Xte.index.tolist()}
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
        comp = {}
        for name, (est, grid) in candidates().items():
            oof = np.zeros(len(ytr)); fold_auc, fold_rec = [], []
            for tr, va in outer.split(Xtr, ytr):
                gs = GridSearchCV(clone(est), grid, scoring="roc_auc",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
                gs.fit(Xtr.iloc[tr], ytr.iloc[tr])
                p = gs.predict_proba(Xtr.iloc[va])[:, 1]; oof[va] = p
                fold_auc.append(roc_auc_score(ytr.iloc[va], p))
                fold_rec.append(recall_score(ytr.iloc[va], (p >= .5).astype(int)))
            comp[name] = dict(cv_roc_auc_mean=np.mean(fold_auc), cv_roc_auc_std=np.std(fold_auc),
                              cv_recall_at_0_5_mean=np.mean(fold_rec),
                              cv_pooled=metrics_at(ytr, oof, 0.5))
            print(f"[{target}] {name:18s} nestedCV AUC={np.mean(fold_auc):.3f}±{np.std(fold_auc):.3f} "
                  f"recall@0.5={np.mean(fold_rec):.3f}", flush=True)
        best = max(comp, key=lambda m: (round(comp[m]["cv_roc_auc_mean"], 3), comp[m]["cv_recall_at_0_5_mean"]))
        est, grid = candidates()[best]
        gs = GridSearchCV(est, grid, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1).fit(Xtr, ytr)
        final = gs.best_estimator_
        oof = cross_val_predict(clone(final), Xtr, ytr, method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1))[:, 1]
        thr = pick_threshold(ytr, oof)
        pte = final.predict_proba(Xte)[:, 1]
        res = dict(best_model=best, best_params={k: v for k, v in gs.best_params_.items()},
                   n_train=len(ytr), n_test=len(yte), positives_train=int(ytr.sum()), positives_test=int(yte.sum()),
                   decision_threshold=round(thr, 3),
                   model_comparison_nested_cv={k: {a: (r(b) if isinstance(b, dict) else round(float(b), 4)) for a, b in v.items()} for k, v in comp.items()},
                   test_at_threshold_0_5=r(metrics_at(yte, pte, 0.5)),
                   test_at_tuned_threshold=r(metrics_at(yte, pte, thr)))
        all_metrics["targets"][target] = res
        t = res["test_at_tuned_threshold"]
        print(f"==> [{target}] best={best} thr={thr:.2f} TEST AUC={t['roc_auc']:.3f} recall={t['recall']:.3f} "
              f"prec={t['precision']:.3f} f1={t['f1']:.3f} acc={t['accuracy']:.3f}\n", flush=True)
        # Ship the train-only model so the reported test metrics describe the shipped artifact exactly.
        joblib.dump({"pipeline": final, "features": X.columns.tolist(), "threshold": thr,
                     "model_name": best, "target": target}, MODELS_DIR / f"{target.lower()}_model.joblib")
    json.dump(all_metrics, open(ROOT / "metrics.json", "w"), indent=2, default=float)
    json.dump(split_idx, open(MODELS_DIR / "split_indices.json", "w"))


if __name__ == "__main__":
    main()
