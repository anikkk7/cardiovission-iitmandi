# CardioVision AI

CardioVision AI is a cardiovascular risk prediction and visualization prototype. It uses trained machine-learning models to estimate:

- overall coronary artery disease risk (`CAD`)
- vessel-level stenosis probability for `LAD`, `LCX`, and `RCA`
- patient-level SHAP explanations for why each prediction moved up or down
- a React dashboard with a procedural 3D heart visualization

This is a research and education prototype only. It is not a diagnosis, not a medical device, and not a substitute for clinical evaluation or imaging.

## Quick Start On Windows

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
.\scripts\start-dev.ps1
```

Then open:

- frontend: http://127.0.0.1:5173
- backend docs: http://127.0.0.1:8000/docs

To stop the development servers:

```powershell
.\scripts\stop-dev.ps1
```

## Manual Setup

Backend:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://127.0.0.1:8000`. You can override this by creating `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Tests And Build

Backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Frontend production build:

```powershell
cd frontend
npm run build
```

## What The App Does

The dashboard lets you enter clinical features or load real held-out demo patients. Empty fields are imputed with training medians. The backend returns:

- CAD probability and risk band
- LAD, LCX, and RCA stenosis probabilities
- model threshold and screening flag per target
- top SHAP feature contributions per target
- input summary showing which values were imputed

The 3D heart uses vessel-level predictions to color the LAD, LCX, and RCA. It does not show real lesion coordinates.

## API Endpoints

Run the backend with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | model load status |
| `GET /model-metrics` | metrics, global SHAP importance, and risk bands |
| `GET /feature-schema` | real model feature list with types and ranges |
| `GET /demo-patients` | held-out demo patients |
| `POST /predict?top_k=8` | predictions and SHAP explanations |

Example prediction body:

```json
{
  "features": {
    "Age": 63,
    "Sex": 1,
    "HTN": 1,
    "DM": 0
  }
}
```

Unknown fields, leakage columns (`Cath`, `LAD`, `LCX`, `RCA`), invalid binary values, and implausible numeric values are rejected.

## Model And Data Notes

Dataset: extended Z-Alizadeh Sani coronary artery disease dataset.

Current artifacts:

- `models/cad_model.joblib`
- `models/lad_model.joblib`
- `models/lcx_model.joblib`
- `models/rca_model.joblib`
- `models/shap_background.joblib`
- `models/global_importance.json`
- `metrics.json`
- `feature_metadata.json`

Leakage control is enforced in `backend/ml/data.py`: target columns are never used as input features.

## Reproducing The Model Artifacts

```powershell
.\.venv\Scripts\python.exe backend/ml/train.py
.\.venv\Scripts\python.exe backend/ml/make_metadata.py
.\.venv\Scripts\python.exe backend/ml/make_explainers.py
```

Training can take several minutes.

## Project Structure

```text
backend/app/        FastAPI app, schemas, model store
backend/ml/         data processing, training, SHAP helpers
backend/tests/      API and SHAP tests
frontend/           React + TypeScript + Vite dashboard
data/               dataset used for retraining
models/             trained model artifacts
reports/            training log
scripts/            setup/start/stop helpers for local development
```

## Limitations

- Small dataset: 303 patients.
- No external validation.
- LCX and RCA models are weaker than CAD/LAD.
- Risk bands are presentation bands, not clinical cutoffs.
- This project is for demonstration, research, and education only.
