from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------
# 0) Paths and model config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "FeIPSHC_bCD_ML_Dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "gbr_best_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"

RANDOM_SEED = 42
FEATURES = ["Time(min)", "Temperature(°C)", "pH", "C0(mg/L)", "Dosage(g/L)"]
TARGET = "qe(mg/g)"
BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.85,
    "random_state": RANDOM_SEED,
}

COLUMN_ALIASES = {
    "Time (min)": "Time(min)",
    "Time": "Time(min)",
    "time": "Time(min)",
    "Temperature (°C)": "Temperature(°C)",
    "Temperature": "Temperature(°C)",
    "Temp": "Temperature(°C)",
    "temp": "Temperature(°C)",
    "C0 (mg/L)": "C0(mg/L)",
    "C0": "C0(mg/L)",
    "Dosage (g/L)": "Dosage(g/L)",
    "Dosage": "Dosage(g/L)",
    "qe (mg/g)": "qe(mg/g)",
}

# -------------------------
# 1) Page config
# -------------------------
st.set_page_config(
    page_title="GBR Adsorption Prediction | FeIPSHC-β-CD",
    page_icon="🔬",
    layout="centered",
)

# -------------------------
# 2) Light red style, adapted from App (1).py
# -------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --bg1:#fff7f7;
  --bg2:#ffe8ec;
  --card:#ffffffd9;
  --border:rgba(190,18,60,0.14);
  --text:#241116;
  --muted:#6b4b55;
  --accent:#e11d48;
  --accent2:#fb7185;
  --accent3:#be123c;
  --soft:#fff1f2;
}

.stApp{
  max-width: 900px;
  margin: 0 auto;
  padding: 2.4rem 2.2rem 3.8rem 2.2rem;
  background:
    radial-gradient(circle at top left, rgba(251,113,133,0.20), transparent 30%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

html, body, [class*="css"]{
  font-family:'Inter','Segoe UI',sans-serif;
  color:var(--text);
  font-size:20px !important;
  line-height:1.6;
}

.block-container{ padding-top:0.7rem !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p,
.stMarkdown li{ font-size:1.08rem !important; }

.header-card,
.input-card,
.metric-card,
.warning-card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:22px;
  box-shadow:0 16px 35px rgba(190,18,60,0.08);
}

.header-card{
  padding:24px 24px 20px 24px;
  margin-bottom:18px;
}

.title{
  font-size:1.9rem;
  font-weight:900;
  letter-spacing:-0.03em;
  margin:0 0 10px 0;
}

.desc{
  font-size:1.12rem;
  color:var(--muted);
  margin:0;
  line-height:1.7;
}

.pill-row{ margin-top:16px; }
.pill{
  display:inline-block;
  background:var(--soft);
  color:var(--accent3);
  border:1px solid rgba(225,29,72,0.16);
  border-radius:999px;
  padding:6px 12px;
  margin:4px 6px 0 0;
  font-size:0.85rem;
  font-weight:800;
}

.input-card{ padding:16px 20px 8px 20px; margin:14px 0 14px 0; }
.section-title{ font-size:1.18rem; font-weight:900; margin:0; }

.stTabs [data-baseweb="tab-list"]{ gap:8px; }
.stTabs [data-baseweb="tab"]{
  background:#fff;
  border:1px solid var(--border);
  border-radius:14px 14px 0 0;
  padding:12px 18px;
  font-weight:800;
}

.stRadio > label,
.stNumberInput label,
.stFileUploader label{
  font-weight:900 !important;
  font-size:1.05rem !important;
  color:var(--text) !important;
}

div[role="radiogroup"] label{ font-size:1rem !important; font-weight:800 !important; }
div[data-baseweb="input"] input{ font-size:1.06rem !important; padding:12px 14px !important; }

.stButton > button,
.stDownloadButton > button{
  width:100%;
  font-weight:900;
  font-size:1.08rem !important;
  padding:0.9rem 1.2rem;
  border-radius:15px;
  margin-top:14px;
}

.stButton > button{
  background:linear-gradient(135deg, var(--accent), var(--accent2));
  color:white;
  border:none;
  box-shadow:0 14px 26px rgba(225,29,72,0.22);
}

.stDownloadButton > button{
  background:white;
  color:var(--accent3);
  border:1px solid rgba(225,29,72,0.22);
}

.result-card{
  background:linear-gradient(135deg, rgba(225,29,72,0.12), rgba(251,113,133,0.14));
  border:1px solid rgba(225,29,72,0.22);
  border-radius:22px;
  padding:20px 22px;
  margin-top:18px;
}
.result-label{ color:var(--muted); font-weight:800; font-size:1rem; margin:0; }
.result-text{ font-size:1.55rem; font-weight:900; margin:4px 0 0 0; color:var(--accent3); }

.metric-card{ padding:16px 18px; margin:8px 0; }
.metric-title{ color:var(--muted); font-size:0.9rem; font-weight:800; margin:0; }
.metric-value{ color:var(--accent3); font-size:1.3rem; font-weight:900; margin:0; }

.warning-card{ padding:14px 18px; margin-top:14px; color:var(--muted); }

hr{ border:none; border-top:1px solid rgba(190,18,60,0.13); margin:18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# 3) Data/model helpers
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        renamed[col] = COLUMN_ALIASES.get(clean, clean)
    return df.rename(columns=renamed)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return normalize_columns(df)


def train_final_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(**BEST_PARAMS)
    model.fit(df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float))
    return model


@st.cache_resource
def load_model_and_metadata(model_mtime: float | None, data_mtime: float | None):
    df = load_dataset()

    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
            metadata = bundle if isinstance(bundle, dict) else {}
            return model, metadata, df, "loaded"
        except Exception:
            # Fall back to training from CSV so the app still runs on Streamlit Cloud.
            pass

    model = train_final_model(df)
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "best_params": BEST_PARAMS,
        "n_samples": int(len(df)),
        "training_note": "Model trained at app startup from CSV because the saved model bundle was unavailable.",
    }
    return model, metadata, df, "trained"


def get_assets():
    model_mtime = os.path.getmtime(MODEL_PATH) if MODEL_PATH.exists() else None
    data_mtime = os.path.getmtime(DATA_PATH) if DATA_PATH.exists() else None
    return load_model_and_metadata(model_mtime, data_mtime)


def make_prediction_frame(c0: float, time_min: float, ph: float, dosage: float, temp: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "Time(min)": time_min,
            "Temperature(°C)": temp,
            "pH": ph,
            "C0(mg/L)": c0,
            "Dosage(g/L)": dosage,
        }],
        columns=FEATURES,
    )


def predict_df(model, input_df: pd.DataFrame) -> np.ndarray:
    input_df = normalize_columns(input_df.copy())
    missing = [col for col in FEATURES if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = input_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cols = list(X.columns[X.isna().any()])
        raise ValueError(f"Non-numeric or missing values found in: {bad_cols}")
    return model.predict(X.to_numpy(dtype=float))


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


# -------------------------
# 4) Load assets
# -------------------------
try:
    model, metadata, df_train, model_status = get_assets()
except Exception as exc:
    st.error(f"Failed to load model/data: {exc}")
    st.stop()

ranges = df_train[FEATURES].agg(["min", "median", "max"])
metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}

# -------------------------
# 5) Language text
# -------------------------
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

text = {
    "English": {
        "title": "🔬 GBR prediction of adsorption capacity for FeIPSHC-β-CD",
        "description": "A Streamlit GUI for predicting equilibrium adsorption capacity qe (mg/g) using the benchmark-selected Gradient Boosting Regressor model.",
        "pills": ["Best model: GBR", "Test R²: 0.9831", "64 training samples", "Light red UI"],
        "input_section": "Input experimental conditions",
        "batch_section": "Batch prediction from CSV",
        "single_tab": "Single prediction",
        "batch_tab": "Batch prediction",
        "labels": {
            "c0": "💧 Initial concentration, C0 (mg/L)",
            "time": "⏱ Adsorption time (min)",
            "ph": "🌡 Solution pH",
            "dosage": "🧪 Adsorbent dosage (g/L)",
            "temp": "🌤 Temperature (°C)",
        },
        "predict_button": "🔍 Predict qe",
        "result_prefix": "Predicted adsorption capacity",
        "export_button": "📁 Export prediction CSV",
        "batch_help": "Upload a CSV containing Time(min), Temperature(°C), pH, C0(mg/L), and Dosage(g/L). Common variants with spaces are also accepted.",
        "download_template": "⬇️ Download CSV template",
        "download_batch": "📁 Export batch predictions",
        "metrics_title": "Model information",
        "range_title": "Training data range",
        "note": "Note: predictions outside the training-data range are extrapolations and should be validated experimentally.",
        "debug_title": "Debug / self-check",
    },
    "中文": {
        "title": "🔬 FeIPSHC-β-CD 吸附量的 GBR 机器学习预测",
        "description": "基于 benchmark 中筛选出的 Gradient Boosting Regressor（GBR）最佳模型，预测平衡吸附量 qe（mg/g）。",
        "pills": ["最佳模型：GBR", "测试集 R²：0.9831", "训练样本：64", "浅红色界面"],
        "input_section": "输入实验条件",
        "batch_section": "CSV 批量预测",
        "single_tab": "单次预测",
        "batch_tab": "批量预测",
        "labels": {
            "c0": "💧 初始浓度 C0 (mg/L)",
            "time": "⏱ 吸附时间 (min)",
            "ph": "🌡 溶液 pH",
            "dosage": "🧪 吸附剂投加量 (g/L)",
            "temp": "🌤 温度 (°C)",
        },
        "predict_button": "🔍 预测 qe",
        "result_prefix": "预测的平衡吸附量",
        "export_button": "📁 导出预测 CSV",
        "batch_help": "上传包含 Time(min)、Temperature(°C)、pH、C0(mg/L)、Dosage(g/L) 的 CSV；带空格的常见列名也可识别。",
        "download_template": "⬇️ 下载 CSV 模板",
        "download_batch": "📁 导出批量预测结果",
        "metrics_title": "模型信息",
        "range_title": "训练数据范围",
        "note": "提示：超出训练数据范围的输入属于外推预测，建议结合实验结果验证。",
        "debug_title": "调试 / 自检",
    },
}[lang]

# -------------------------
# 6) Header
# -------------------------
st.markdown(
    f"""
<div class="header-card">
  <div class="title">{text['title']}</div>
  <p class="desc">{text['description']}</p>
  <div class="pill-row">
    {''.join([f'<span class="pill">{p}</span>' for p in text['pills']])}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# 7) Main tabs
# -------------------------
tab_single, tab_batch, tab_info = st.tabs([text["single_tab"], text["batch_tab"], text["metrics_title"]])

with tab_single:
    st.markdown(
        f"""
<div class="input-card">
  <div class="section-title">🧩 {text['input_section']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        c0 = st.number_input(
            text["labels"]["c0"],
            min_value=0.0,
            value=float(ranges.loc["median", "C0(mg/L)"]),
            step=1.0,
        )
        ph = st.number_input(
            text["labels"]["ph"],
            min_value=0.0,
            max_value=14.0,
            value=float(ranges.loc["median", "pH"]),
            step=0.1,
        )
        temp = st.number_input(
            text["labels"]["temp"],
            min_value=0.0,
            value=float(ranges.loc["median", "Temperature(°C)"]),
            step=1.0,
        )
    with col_b:
        time_min = st.number_input(
            text["labels"]["time"],
            min_value=0.0,
            value=float(ranges.loc["median", "Time(min)"]),
            step=10.0,
        )
        dosage = st.number_input(
            text["labels"]["dosage"],
            min_value=0.0,
            value=float(ranges.loc["median", "Dosage(g/L)"]),
            step=0.1,
        )

    single_input = make_prediction_frame(c0, time_min, ph, dosage, temp)

    if st.button(text["predict_button"]):
        pred = float(predict_df(model, single_input)[0])
        st.markdown(
            f"""
<div class="result-card">
  <p class="result-label">✅ {text['result_prefix']}</p>
  <p class="result-text">{pred:.2f} mg/g</p>
</div>
""",
            unsafe_allow_html=True,
        )

        result_df = single_input.copy()
        result_df["Predicted qe(mg/g)"] = round(pred, 4)
        st.download_button(
            text["export_button"],
            data=csv_download_bytes(result_df),
            file_name="gbr_single_prediction.csv",
            mime="text/csv",
        )

    st.markdown(f"<div class='warning-card'>⚠️ {text['note']}</div>", unsafe_allow_html=True)

with tab_batch:
    st.markdown(
        f"""
<div class="input-card">
  <div class="section-title">📦 {text['batch_section']}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write(text["batch_help"])

    template_df = pd.DataFrame(
        [{
            "Time(min)": float(ranges.loc["median", "Time(min)"]),
            "Temperature(°C)": float(ranges.loc["median", "Temperature(°C)"]),
            "pH": float(ranges.loc["median", "pH"]),
            "C0(mg/L)": float(ranges.loc["median", "C0(mg/L)"]),
            "Dosage(g/L)": float(ranges.loc["median", "Dosage(g/L)"]),
        }]
    )
    st.download_button(
        text["download_template"],
        data=csv_download_bytes(template_df),
        file_name="gbr_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            preds = predict_df(model, batch_df)
            output_df = normalize_columns(batch_df.copy())
            output_df["Predicted qe(mg/g)"] = np.round(preds, 4)
            st.dataframe(output_df, use_container_width=True, hide_index=True)
            st.download_button(
                text["download_batch"],
                data=csv_download_bytes(output_df),
                file_name="gbr_batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

with tab_info:
    metric_cols = st.columns(3)
    values = [
        ("R² test", metrics.get("R2_test", 0.9831)),
        ("RMSE test", metrics.get("RMSE_test", 4.4224)),
        ("MAE test", metrics.get("MAE_test", 3.1356)),
    ]
    for col, (name, value) in zip(metric_cols, values):
        with col:
            st.markdown(
                f"""
<div class="metric-card">
  <p class="metric-title">{name}</p>
  <p class="metric-value">{float(value):.4f}</p>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader(text["range_title"])
    display_ranges = ranges.T.rename(columns={"min": "Min", "median": "Median", "max": "Max"})
    st.dataframe(display_ranges, use_container_width=True)

    st.markdown("---")
    st.write("**Best GBR parameters**")
    st.json(BEST_PARAMS)

    with st.expander(f"🧾 {text['debug_title']}", expanded=False):
        st.write("Model status:", model_status)
        st.write("Model type:", type(model).__name__)
        st.write("Feature order:", FEATURES)
        st.write("Training rows:", len(df_train))
        if METADATA_PATH.exists():
            st.write("Metadata file:", str(METADATA_PATH.name))
