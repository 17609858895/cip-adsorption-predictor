# FeIPSHC-β-CD GBR Adsorption Prediction App

This repository contains a Streamlit GUI for predicting adsorption capacity `qe (mg/g)` using the best Gradient Boosting Regressor (GBR) model selected in the benchmark notebook.

## Model summary

- Best model: Gradient Boosting Regressor (GBR)
- Benchmark test R²: 0.9831
- Benchmark test RMSE: 4.4224 mg/g
- Benchmark test MAE: 3.1356 mg/g
- Training samples: 64
- Input features:
  - `Time(min)`
  - `Temperature(°C)`
  - `pH`
  - `C0(mg/L)`
  - `Dosage(g/L)`
- Target: `qe(mg/g)`

Best hyperparameters:

```python
{
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.85,
    "random_state": 42,
}
```

## Files

```text
.
├── app.py                              # Streamlit GUI
├── train_model.py                      # Re-train and save the GBR model bundle
├── requirements.txt                    # Python dependencies for Streamlit Cloud
├── README.md                           # Deployment and usage instructions
├── .gitignore
├── .streamlit/
│   └── config.toml                     # Light red Streamlit theme
├── data/
│   └── FeIPSHC_bCD_ML_Dataset.csv      # Training dataset
├── model/
│   ├── gbr_best_model.joblib           # Saved model bundle
│   └── model_metadata.json             # Feature order, metrics, and parameters
└── reference_App_1.py                  # Original reference style file
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The app can still run if `model/gbr_best_model.joblib` is missing, because it will train the GBR model from `data/FeIPSHC_bCD_ML_Dataset.csv` at startup.

## Deploy on Streamlit Community Cloud from GitHub

1. Create a new GitHub repository.
2. Upload all files and folders in this project while preserving the folder structure.
3. Go to Streamlit Community Cloud and choose **New app**.
4. Select your GitHub repository and branch.
5. Set **Main file path** to:

```text
app.py
```

6. Deploy.

## Batch prediction CSV format

The batch prediction tab accepts a CSV with these columns:

```text
Time(min),Temperature(°C),pH,C0(mg/L),Dosage(g/L)
```

The app also accepts common variants such as `Time (min)`, `Temperature (°C)`, `C0 (mg/L)`, and `Dosage (g/L)`.

## Important note

Predictions outside the training-data range are extrapolations. Please validate extrapolated predictions experimentally.
