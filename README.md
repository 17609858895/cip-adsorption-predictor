# 🔬 ML Prediction of CIP Adsorption on Fe@IPSHC/β-CD

Streamlit web app that predicts the equilibrium adsorption capacity (qₑ, mg g⁻¹)
of ciprofloxacin (CIP) on Fe@IPSHC/β-CD using an optimized Gradient Boosting
Regression (GBR) model trained on 64 batch-experiment data points.

## 📁 Files

| File | Description |
|---|---|
| `app.py`                          | Streamlit application (trains GBR on startup) |
| `FeIPSHC_bCD_ML_Dataset.csv`      | Training dataset (64 samples × 5 features + target) |
| `requirements.txt`                | Python dependencies (version ranges, not pinned) |
| `runtime.txt`                     | Pins Python 3.11 for Streamlit Cloud |
| `README.md`                       | This file |

## 🛠️ Local installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. First load takes ~1–2 s while the
GBR model is trained from the bundled CSV; subsequent navigation is instant
(cached).

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (must be **public** for the free tier).
2. Sign in at <https://share.streamlit.io>.
3. **New app** → select repo, branch `main`, main file `app.py`.
4. **Deploy** — live in 1–2 minutes.

The bundled `runtime.txt` ensures Streamlit Cloud uses Python 3.11 (so
all numpy/scikit-learn wheels are pre-built and deployment is fast).

## 🧪 Model details

| Property | Value |
|---|---|
| Algorithm        | Gradient Boosting Regression (sklearn) |
| Hyperparameters  | n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.85 |
| Training samples | 51 (80% of 64, stratified by experiment type) |
| Test samples     | 13 (20%) |
| Test R²          | 0.983 |
| Test RMSE        | 4.42 mg g⁻¹ |

## 📥 Inputs

- Solution pH (1–14)
- Initial CIP concentration C₀ (mg L⁻¹)
- Adsorbent dosage (g L⁻¹)
- Temperature (°C)
- Adsorption time (min)

## 📤 Outputs

- Predicted equilibrium adsorption capacity qₑ (mg g⁻¹)
- Estimated removal efficiency (%)
- CSV export of the input conditions and prediction

## ⚠️ Disclaimer

The model was trained on a limited dataset (n = 64). Predictions far
outside the training ranges (pH < 3 or > 10, C₀ > 300 mg L⁻¹, T > 45 °C)
should be treated as extrapolations and validated experimentally.

## 📄 License

MIT
