# app_streamlit.py
# UI موحّد: (Build KB) + (RAG Answer من app.rag.pipeline) + (Stroke RF)

import json
import sys
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

from app.rag.pipeline import get_answer, get_stroke_answer

# -----------------------
# Constants / Caching
# -----------------------
KB_BUILD_SCRIPT = Path("app/rag/build_index.py")  # يبني app/index/faiss_hnsw.index + metas.json
RF_MODEL_DEFAULT = Path("data/processed/stroke_rf.joblib")
RF_THRESHOLD = 0.05  # المطلوب بالمشروع

@st.cache_resource
def load_rf_model(path: str):
    return joblib.load(path)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="MAD Drug - MedRAG", layout="wide")
st.title("MAD Drug: RAG (FAISS HNSW) + LLM (GPU) + Stroke Risk (RF)")

tab1, tab2 = st.tabs([
    "RAG (Abstracts → Evidence → Arabic Summary)",
    "Stroke Risk (RandomForest)"
])

# =======================
# TAB 1: RAG + RF injection
# =======================
with tab1:
    st.subheader("A) Knowledge Base (FAISS HNSW)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("يبني الفهرس: app/index/faiss_hnsw.index + metas.json")
        st.code(f"{sys.executable} {KB_BUILD_SCRIPT}", language="bash")
    with col2:
        build_btn = st.button("Build / Rebuild Index")

    if build_btn:
        if not KB_BUILD_SCRIPT.exists():
            st.error("❌ app/rag/build_index.py غير موجود.")
        else:
            with st.spinner("Building FAISS index..."):
                r = subprocess.run([sys.executable, str(KB_BUILD_SCRIPT)], capture_output=True, text=True)
            st.text_area("stdout", r.stdout, height=200)
            st.text_area("stderr", r.stderr, height=140)
            if r.returncode == 0:
                st.success("✅ Index built.")
            else:
                st.error("❌ Build failed. اقرأ stderr.")

    st.subheader("B) Ask a medical case (RAG)")

    case_text = st.text_area(
        "Case / Question",
        height=180,
        placeholder="اكتب وصف الحالة أو السؤال هنا…"
    )
    top_k = st.slider("Top-K Evidence", min_value=1, max_value=10, value=3)

    st.subheader("C) Patient Features (for Stroke Risk)")
    st.caption("إذا تركته فارغاً، سيتم تشغيل RAG فقط بدون حساب Stroke Risk.")

    features_json = st.text_area(
        "Features JSON (match stroke dataset columns)",
        value='{"gender":"Male","age":67,"hypertension":0,"heart_disease":1,"ever_married":"Yes","work_type":"Private","Residence_type":"Urban","avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"}',
        height=130
    )

    ask_btn = st.button("Run RAG Answer")
    if ask_btn:
        # 1) Compute RF risk (optional)
        risk_info = None
        risk_block = ""
        if features_json.strip():
            try:
                rf = load_rf_model(str(RF_MODEL_DEFAULT))
                feat = json.loads(features_json)
                X = pd.DataFrame([feat])
                proba = float(rf.predict_proba(X)[0, 1])
                label = "HIGH" if proba >= RF_THRESHOLD else "LOW"
                risk_info = {"proba": proba, "label": label, "threshold": RF_THRESHOLD}
                risk_block = f"[Stroke Risk RF] proba={proba:.3f}, label={label}, threshold={RF_THRESHOLD}"
            except Exception as e:
                risk_block = f"[Stroke Risk RF] unavailable: {e}"

        # 2) Minimal integration: inject risk into the query text
        combined_case = case_text.strip()
        if risk_block:
            combined_case = combined_case + "\n\n" + risk_block

        # 3) Run RAG
        with st.spinner("Running RAG (retrieval + Qwen) ..."):
           feat = json.loads(features_json)
           res = get_stroke_answer(case_text, stroke_features=feat, top_k=top_k)

        st.markdown("### Output (educational, not a final diagnosis)")
        st.write(res.get("answer", res))

        if risk_block:
            st.markdown("### Stroke Risk (RF)")
            st.write(risk_block)

        # Retrieved evidence display
        evid = res.get("evidences") or res.get("sources") or res.get("retrieved") or None
        if evid:
            st.markdown("### Retrieved Evidence")
            if isinstance(evid, list):
                for i, e in enumerate(evid, start=1):
                    st.write(f"[{i}] {e}")
            else:
                st.write(evid)

# =======================
# TAB 2: RF training + prediction
# =======================
with tab2:
    st.subheader("Stroke Risk (RandomForest)")

    model_path = Path(st.text_input("RF model path", str(RF_MODEL_DEFAULT)))
    csv_train_path = st.text_input("Training CSV path", "data/raw/healthcare-dataset-stroke-data.csv")

    colA, colB = st.columns(2)
    with colA:
        train_btn = st.button("Train / Re-train RF")
    with colB:
        st.write(f"Decision threshold: **{RF_THRESHOLD}** (رفع الحساسية)")

    if train_btn:
        from train_stroke_rf import train
        train(csv_train_path, str(model_path))
        st.success("✅ RF trained & saved.")
        # تنظيف كاش الموديل بعد إعادة التدريب
        load_rf_model.clear()

    if model_path.exists():
        st.info("RF model loaded.")

        raw_json = st.text_area(
            "Features JSON (match your CSV columns):",
            value='{"gender":"Male","age":67,"hypertension":0,"heart_disease":1,"ever_married":"Yes","work_type":"Private","Residence_type":"Urban","avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"}',
            height=150
        )

        if st.button("Predict Risk"):
            try:
                rf = load_rf_model(str(model_path))
                feat = json.loads(raw_json)
                X = pd.DataFrame([feat])
                proba = float(rf.predict_proba(X)[0, 1])
                label = "HIGH-RISK (>= threshold)" if proba >= RF_THRESHOLD else "LOW-RISK (< threshold)"
                st.metric("Stroke Risk Probability", f"{proba:.3f}")
                st.write("Risk label:", label)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("RF model not found. Train it first.")
