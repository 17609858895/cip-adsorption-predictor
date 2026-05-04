"""
Streamlit GUI — ML prediction of CIP adsorption capacity on Fe@IPSHC/β-CD.

The GBR model is re-trained at app startup from the bundled CSV
(takes <1 s for 64 samples), and the trained estimator is cached via
@st.cache_resource. This avoids any sklearn-version pickle compatibility
issues on Streamlit Cloud.
"""

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIP Adsorption Prediction (Fe@IPSHC/β-CD)",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — light pink theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        max-width: 640px;
        margin: auto;
        background-color: #fdf2f8;
        padding: 2rem 2rem 4rem 2rem;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif;
    }
    .custom-title {
        font-size: 1.65rem;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 0.4rem;
        color: #1f2937;
        letter-spacing: -0.01em;
    }
    .custom-desc {
        font-size: 1.0rem;
        color: #4b5563;
        line-height: 1.55;
        margin-bottom: 1.6rem;
    }
    .stNumberInput label {
        font-size: 0.97rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.25rem;
    }
    .stNumberInput input {
        font-size: 1.0rem;
        color: #1f2937;
    }
    .stButton>button {
        background-color: #ec4899;
        color: white;
        font-weight: 600;
        font-size: 1.0rem;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
        margin-top: 1.3rem;
        transition: background-color 0.15s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #db2777;
        color: white;
    }
    .stDownloadButton>button {
        background-color: white;
        color: #374151;
        font-weight: 500;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-top: 0.8rem;
        padding: 0.5rem 1rem;
        font-size: 0.95rem;
    }
    .stDownloadButton>button:hover {
        background-color: #f9fafb;
        border-color: #d1d5db;
    }
    div[data-testid="stAlert"] {
        background-color: #fce7f3 !important;
        border: 1px solid #f9a8d4 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlert"] p {
        color: #831843 !important;
        font-size: 1.04rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    div[data-testid="stAlert"][kind="warning"] {
        background-color: #fef3c7 !important;
        border: 1px solid #fcd34d !important;
    }
    div[data-testid="stAlert"][kind="warning"] p {
        color: #78350f !important;
    }
    [data-testid="stMetricValue"] {
        color: #be185d;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-size: 0.85rem;
    }
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #374151 !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Train (or load cached) GBR model
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["Time(min)", "Temperature(°C)", "pH", "C0(mg/L)", "Dosage(g/L)"]
TARGET = "qe(mg/g)"
DATA_FILE = "FeIPSHC_bCD_ML_Dataset.csv"


@st.cache_resource(show_spinner="Training GBR model...")
def train_model():
    df = pd.read_csv(Path(__file__).parent / DATA_FILE)

    X = df[FEATURES].values
    y = df[TARGET].values
    src = df["Source"].values if "Source" in df.columns else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=src
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    metrics = {
        "R2_train":  float(r2_score(y_tr, model.predict(X_tr))),
        "R2_test":   float(r2_score(y_te, model.predict(X_te))),
        "RMSE_test": float(np.sqrt(mean_squared_error(y_te, model.predict(X_te)))),
        "n_train":   len(y_tr),
        "n_test":    len(y_te),
    }

    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in FEATURES}

    return model, ranges, metrics


model, ranges, metrics = train_model()


# ─────────────────────────────────────────────────────────────────────────────
# Language switch
# ─────────────────────────────────────────────────────────────────────────────
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

text = {
    "English": {
        "title": "🔬 ML prediction of ciprofloxacin (CIP) adsorption on Fe@IPSHC/β-CD",
        "description": "Predict the equilibrium adsorption capacity qₑ (mg g⁻¹) of CIP on Fe@IPSHC/β-CD under specified experimental conditions.",
        "input_labels": [
            "🌡 Solution pH",
            "💧 Initial CIP concentration, C₀ (mg L⁻¹)",
            "🧪 Adsorbent dosage (g L⁻¹)",
            "🌤 Temperature (°C)",
            "⏱ Adsorption time (min)"
        ],
        "button_predict": "🔍 Predict adsorption capacity",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted equilibrium adsorption capacity:",
        "result_unit": "mg g⁻¹",
        "removal_label": "Estimated removal efficiency",
        "file_name": "cip_prediction_result.csv",
        "model_info": "📊 Model information",
        "model_type": "Algorithm",
        "perf": "Test R²",
        "rmse": "Test RMSE (mg g⁻¹)",
        "n_total": "Training samples",
        "warn_range": "⚠️ Input is outside the training range; prediction may be less reliable:",
    },
    "中文": {
        "title": "🔬 Fe@IPSHC/β-CD 对环丙沙星（CIP）吸附量的机器学习预测",
        "description": "在指定实验条件下预测 Fe@IPSHC/β-CD 对环丙沙星（CIP）的平衡吸附容量 qₑ（mg g⁻¹）。",
        "input_labels": [
            "🌡 溶液 pH",
            "💧 初始 CIP 浓度 C₀（mg L⁻¹）",
            "🧪 吸附剂投加量（g L⁻¹）",
            "🌤 温度（°C）",
            "⏱ 吸附时间（min）"
        ],
        "button_predict": "🔍 预测吸附量",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的平衡吸附容量：",
        "result_unit": "mg g⁻¹",
        "removal_label": "预估去除率",
        "file_name": "CIP_预测结果.csv",
        "model_info": "📊 模型信息",
        "model_type": "算法",
        "perf": "测试集 R²",
        "rmse": "测试集 RMSE (mg g⁻¹)",
        "n_total": "训练样本数",
        "warn_range": "⚠️ 输入超出训练范围，预测可能不可靠：",
    }
}[lang]

# ─────────────────────────────────────────────────────────────────────────────
# Title and description
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="custom-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="custom-desc">{text["description"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Inputs: pH → C0 → Dosage → Temp → Time
# ─────────────────────────────────────────────────────────────────────────────
pH = st.number_input(text["input_labels"][0],
                     min_value=1.0, max_value=14.0, value=6.0, step=0.1,
                     format="%.2f")
c0 = st.number_input(text["input_labels"][1],
                     min_value=0.0, value=5.0, step=0.5,
                     format="%.2f")
dosage = st.number_input(text["input_labels"][2],
                         min_value=0.0, value=0.4, step=0.05,
                         format="%.2f")
temperature = st.number_input(text["input_labels"][3],
                              min_value=0.0, value=25.0, step=1.0,
                              format="%.2f")
ads_time = st.number_input(text["input_labels"][4],
                           min_value=0.0, value=1440.0, step=10.0,
                           format="%.2f")

# ─────────────────────────────────────────────────────────────────────────────
# Range warning
# ─────────────────────────────────────────────────────────────────────────────
def is_oor(val, key):
    lo, hi = ranges[key]
    return val < lo or val > hi


user_inputs = {
    "pH": pH,
    "C0(mg/L)": c0,
    "Dosage(g/L)": dosage,
    "Temperature(°C)": temperature,
    "Time(min)": ads_time,
}
oor_features = [k for k, v in user_inputs.items() if is_oor(v, k)]
if oor_features:
    st.warning(text["warn_range"] + " " + ", ".join(oor_features))

# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
prediction = None
removal = None
df_result = None

if st.button(text["button_predict"]):
    # Model expects: Time, Temperature, pH, C0, Dosage
    input_data = np.array([[ads_time, temperature, pH, c0, dosage]], dtype=float)
    prediction = float(model.predict(input_data)[0])
    prediction = max(prediction, 0.0)
    removal = min(100 * prediction * dosage / c0, 100) if c0 > 0 else 0

    st.success(
        f"{text['result_prefix']}  qₑ = **{prediction:.2f} {text['result_unit']}**"
        f"   ·   {text['removal_label']}: **{removal:.1f}%**"
    )

    df_result = pd.DataFrame([{
        "pH": pH,
        "C0 (mg/L)": c0,
        "Dosage (g/L)": dosage,
        "Temperature (degC)": temperature,
        "Time (min)": ads_time,
        "qe (mg/g)": round(prediction, 2),
        "Removal (%)": round(removal, 1)
    }], columns=["pH", "C0 (mg/L)", "Dosage (g/L)", "Temperature (degC)",
                 "Time (min)", "qe (mg/g)", "Removal (%)"])

# ─────────────────────────────────────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────────────────────────────────────
if prediction is not None and df_result is not None:
    towrite = BytesIO()
    df_result.to_csv(towrite, index=False, encoding="utf-8-sig")
    st.download_button(
        label=text["button_export"],
        data=towrite.getvalue(),
        file_name=text["file_name"],
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Model info expander
# ─────────────────────────────────────────────────────────────────────────────
with st.expander(text["model_info"]):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(text["model_type"], "GBR")
    c2.metric(text["perf"], f"{metrics['R2_test']:.3f}")
    c3.metric(text["rmse"], f"{metrics['RMSE_test']:.2f}")
    c4.metric(text["n_total"], metrics["n_train"])
