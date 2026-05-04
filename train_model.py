"""Train and save the best Gradient Boosting Regressor model.

Run locally:
    python train_model.py

This script reproduces the GBR setup selected in the benchmark notebook and
saves a deployable model bundle for the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "FeIPSHC_bCD_ML_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "gbr_best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURES = ["Time(min)", "Temperature(°C)", "pH", "C0(mg/L)", "Dosage(g/L)"]
TARGET = "qe(mg/g)"
BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.85,
    "random_state": RANDOM_SEED,
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def evaluate_gbr(df: pd.DataFrame) -> dict:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    source = df["Source"].to_numpy() if "Source" in df.columns else None

    stratify = source if source is not None else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    model = GradientBoostingRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_model = GradientBoostingRegressor(**BEST_PARAMS)
    cv_r2 = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="r2").mean()

    return {
        "R2_train": float(r2_score(y_train, yhat_train)),
        "R2_cv": float(cv_r2),
        "R2_test": float(r2_score(y_test, yhat_test)),
        "RMSE_train": float(np.sqrt(mean_squared_error(y_train, yhat_train))),
        "RMSE_test": float(np.sqrt(mean_squared_error(y_test, yhat_test))),
        "MAE_train": float(mean_absolute_error(y_train, yhat_train)),
        "MAE_test": float(mean_absolute_error(y_test, yhat_test)),
    }


def train_final_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    """Train the final deployable model on the full dataset."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    model = GradientBoostingRegressor(**BEST_PARAMS)
    model.fit(X, y)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    metrics = evaluate_gbr(df)
    final_model = train_final_model(df)

    bundle = {
        "model": final_model,
        "features": FEATURES,
        "target": TARGET,
        "best_params": BEST_PARAMS,
        "metrics": metrics,
        "n_samples": int(len(df)),
        "training_note": "Final deployable GBR model trained on the full dataset using benchmark-selected hyperparameters.",
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {k: v for k, v in bundle.items() if k != "model"}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
