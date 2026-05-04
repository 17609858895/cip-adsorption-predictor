# 🔬 ML Prediction of CIP Adsorption on Fe@IPSHC/β-CD

A lightweight Streamlit web app that predicts the equilibrium adsorption capacity (q_e, mg g⁻¹) of ciprofloxacin (CIP) on Fe@IPSHC/β-CD using an optimized Gradient Boosting Regression (GBR) model trained on 64 batch-experiment data points.

## 🚀 Live demo

Deployed on Streamlit Community Cloud:

```
https://<your-username>-<repo-name>.streamlit.app
```

## 📁 Files

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `gbr_model.pkl` | Pre-trained GBR model + feature ranges + metrics (~86 KB) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## 🛠️ Local installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Sign in at https://share.streamlit.io with your GitHub account.
3. Click **New app** → select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. The app will be live in 1–2 minutes.

## 🧪 Model details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regression (sklearn) |
| Hyperparameters | n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.85 |
| Training samples | 51 (80% of 64) |
| Test samples | 13 (20%) |
| Test R² | 0.983 |
| Test RMSE | 4.42 mg g⁻¹ |

## 📥 Inputs

- Solution pH (1–14)
- Initial CIP concentration C₀ (mg/L)
- Adsorbent dosage (g/L)
- Temperature (°C)
- Adsorption time (min)

## 📤 Outputs

- Predicted equilibrium adsorption capacity q_e (mg g⁻¹)
- Estimated removal efficiency (%) computed via mass balance
- CSV export of the input conditions and prediction result

## ⚠️ Disclaimer

The model was trained on a limited dataset (n = 64). Predictions outside the training ranges (especially pH < 3 or > 10, C₀ > 300 mg/L, T > 45 °C) should be treated as extrapolations and validated experimentally.

## 📄 License

MIT
