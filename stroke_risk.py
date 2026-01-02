from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

# جذر المشروع (med-rag)
BASE_DIR = Path(__file__).resolve().parents[2]

# نفس المسار اللي حفظناه في stroke_thresholds.py
MODEL_PATH = BASE_DIR / "models" / "stroke_rf_pipeline.joblib"

# العتبة اللي اخترناها من التجارب
STROKE_THRESHOLD = 0.05

_model = None


def _load_model():
    """تحميل الموديل (Pipeline) من ملف joblib مع كاش بسيط."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Stroke model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_stroke_risk(features: Dict) -> Dict:
    """
    توقع خطر السكتة الدماغية لمريض واحد.

    features لازم يحتوي نفس الأعمدة اللي دربنا عليها:
      gender, age, hypertension, heart_disease, ever_married,
      work_type, Residence_type, avg_glucose_level, bmi, smoking_status

    القيم التصنيفية لازم تكون بنفس الأسامي الموجودة في الداتا الأصلية:
      - gender: "Male" / "Female" / "Other"
      - ever_married: "Yes" / "No"
      - work_type: "Private" / "Self-employed" / "Govt_job" / "children" / "Never_worked"
      - Residence_type: "Urban" / "Rural"
      - smoking_status: "formerly smoked" / "never smoked" / "smokes" / "Unknown"
    """
    model = _load_model()

    # ✅ (1) تحقق من وجود كل الأعمدة المطلوبة قبل بناء DataFrame
    required = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ]
    missing = [c for c in required if c not in features]
    if missing:
        raise ValueError(f"Missing stroke features: {missing}")

    # نبني DataFrame من سطر واحد
    df = pd.DataFrame([features])

    # ✅ (2) تأكد من أن الأعمدة الرقمية أرقام (بشكل آمن)
    num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"Invalid numeric value for {c}: {features.get(c)}")

    # Pipeline جواته الـ preprocessing + RandomForest
    proba = model.predict_proba(df)[0]
    prob_no_stroke = float(proba[0])
    prob_stroke = float(proba[1])

    # تطبيق العتبة
    prediction = int(prob_stroke >= STROKE_THRESHOLD)

    # تصنيف لفظي بسيط حسب الاحتمال
    if prob_stroke < 0.05:
        risk_label = "منخفض جداً"
    elif prob_stroke < 0.15:
        risk_label = "منخفض"
    elif prob_stroke < 0.30:
        risk_label = "متوسط"
    else:
        risk_label = "مرتفع"

    return {
        "prediction": prediction,          # 0 = لا يوجد Stroke متوقع، 1 = مشتبه Stroke
        "prob_no_stroke": round(prob_no_stroke, 3),
        "prob_stroke": round(prob_stroke, 3),
        "risk_label": risk_label,
        "threshold": STROKE_THRESHOLD,
    }
