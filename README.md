# CardioVision AI

**See the Risk. Understand the Heart.**

Hackathon project for **Track A: Cardiovascular Risk Visualization & Prediction**.
Predicts overall coronary artery disease (CAD) and LAD / LCX / RCA stenosis probabilities from clinical features,
with explainability (SHAP) and an interactive 3D heart planned on top.

> **Clinical decision-support prototype. Predictions are for educational/research purposes only and are NOT a
> diagnosis or a substitute for professional medical evaluation or diagnostic imaging.**
> The dataset has vessel-level labels, not lesion coordinates. Any 3D view shows predicted vessel-level risk and
> does not depict an actual anatomical lesion.

## Project status

| Stage | Status |
|---|---|
| 1. Data audit + ML pipeline | **Done** (this repo) |
| 2. FastAPI backend (`/predict`, `/health`, `/model-metrics`) | Planned |
| 3. React dashboard | Planned |
| 4. 3D heart (React Three Fiber) | Planned |
| 5. SHAP explanations | Planned |
| 6-8. UI polish, demo mode, full docs | Planned |

## Dataset

Extended Z-Alizadeh Sani coronary artery disease dataset: 303 patients, 59 columns, no missing values, no duplicates.
File: `data/z_alizadeh_sani_extended.xlsx`.

| Target | Positives (of 303) |
|---|---|
| CAD (`Cath`) | 216 |
| LAD stenosis | 177 |
| LCX stenosis | 119 |
| RCA stenosis | 114 |

**Leakage control:** `Cath`, `LAD`, `LCX` and `RCA` are never used as input features (enforced by an assert in
`backend/ml/data.py`). 55 features are used; `Exertional CP` is dropped because it is constant.

**Encoding notes:** `Sex` 1 = male (dataset spells female "Fmale"); `BBB` split into `LBBB` / `RBBB` flags;
`VHD` ordinal (N=0, mild=1, Moderate=2, Severe=3). The dataset has no total-cholesterol column (LDL, HDL, TG only).
One patient has a normal Cath label but a stenotic LAD label; it is kept as-is.

**Source / licence:** please confirm the licence and citation for the dataset file you redistribute
(original UCI Z-Alizadeh Sani dataset: Alizadehsani et al.). Remove `data/` from the repo if redistribution isn't allowed.

## ML methodology

- Median imputation and scaling live **inside** the sklearn pipeline (fitted on training data only).
- One shared stratified 80/20 split for all four targets (242 train / 61 test), seed 42.
- Candidate models: logistic regression, random forest, gradient boosting, XGBoost (no deep learning; the data is small).
- Model selection by **nested stratified 5-fold CV** on the training set only (inner 3-fold grid search).
- Decision threshold chosen to favour recall (maximise F2) from out-of-fold **training** predictions.
- Test set is evaluated once, at the end. Shipped models are trained on the training split only, so the reported test
  metrics describe the shipped artifacts exactly.

## Results

| Target | Best model | Nested-CV AUC | Test AUC | Test recall @0.5 | Test precision @0.5 |
|---|---|---|---|---|---|
| CAD | Logistic regression | 0.950 ± 0.017 | 0.832 | 0.79 | 0.90 |
| LAD | Random forest | 0.880 ± 0.046 | 0.750 | 0.79 | 0.73 |
| LCX | Random forest | 0.751 ± 0.112 | 0.706 | 0.59 | 0.64 |
| RCA | Logistic regression | 0.715 ± 0.035 | 0.692 | 0.62 | 0.46 |

With the recall-oriented tuned thresholds (test set):

| Target | Threshold | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| CAD | 0.36 | 0.79 | 0.87 | 0.83 | 0.77 |
| LAD | 0.35 | 0.91 | 0.66 | 0.77 | 0.69 |
| LCX | 0.32 | 1.00 | 0.47 | 0.64 | 0.51 |
| RCA | 0.26 | 0.86 | 0.40 | 0.55 | 0.51 |

Full numbers (accuracy, F1, specificity, confusion matrices, all model comparisons) are in `metrics.json`.

## Limitations

- Only 303 patients. The 61-patient test set is noisy (21-43 positives per target); prefer the nested-CV figures.
- LCX and RCA models are weak (AUC about 0.7). Tuned thresholds raise recall at a real cost in precision.
- No external validation; single-centre data. Probabilities are not calibrated yet.
- Not a medical device.

## Installation and reproducing Stage 1

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python backend/ml/train.py          # trains 4 models, writes models/ and metrics.json (several minutes on 1 CPU)
python backend/ml/make_metadata.py  # writes feature_metadata.json + demo patients
```

## Project structure

```
cardiovision-ai/
├── backend/ml/
│   ├── data.py             # loading, encoding, leakage control
│   ├── train.py            # nested-CV comparison, threshold tuning, held-out evaluation
│   └── make_metadata.py    # feature_metadata.json + held-out demo patients
├── data/                   # dataset (.xlsx)
├── models/                 # {cad,lad,lcx,rca}_model.joblib, split_indices.json
├── reports/train_log.txt
├── metrics.json
├── feature_metadata.json
└── requirements.txt
```

## Roadmap

FastAPI backend with Pydantic validation and per-target SHAP, React + TypeScript + Tailwind dashboard,
React Three Fiber heart with LAD/LCX/RCA vessel risk colouring, demo-patient flow, and a full README/API section.
