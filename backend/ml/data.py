"""Dataset loading, cleaning and leakage control for CardioVision AI.

All transformations here are deterministic row-wise mappings (no statistics are
learned), so they are safe to apply before the train/test split. Anything that
learns from data (imputation, scaling) lives inside the sklearn Pipeline and is
fitted on training folds only.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "z_alizadeh_sani_extended.xlsx"

# Labels and anything derived from the catheterisation result. NEVER model inputs.
TARGET_COLS = {"CAD": "Cath", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
LEAKAGE_COLS = ["Cath", "LAD", "LCX", "RCA"]
POSITIVE = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}

BINARY_YN = ["Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
             "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
             "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "LVH",
             "Poor R Progression"]
VHD_ORDER = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_targets(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: (raw[v] == POSITIVE[v]).astype(int) for k, v in TARGET_COLS.items()})


def build_features(raw: pd.DataFrame):
    """Return (numeric feature frame, dropped constant columns). Leakage columns dropped explicitly."""
    X = raw.drop(columns=LEAKAGE_COLS).copy()
    X["Sex"] = (X["Sex"] == "Male").astype(int)          # 1 = male (dataset spells female as 'Fmale')
    for c in BINARY_YN:
        X[c] = (X[c] == "Y").astype(int)
    X["VHD"] = X["VHD"].map(VHD_ORDER)                    # ordinal severity
    X["LBBB"] = (X["BBB"] == "LBBB").astype(int)
    X["RBBB"] = (X["BBB"] == "RBBB").astype(int)
    X = X.drop(columns=["BBB"])
    const = [c for c in X.columns if X[c].nunique() <= 1]  # e.g. 'Exertional CP' is all 'N'
    X = X.drop(columns=const)
    assert not set(LEAKAGE_COLS) & set(X.columns), "target leakage!"
    return X, const
