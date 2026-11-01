# app/rag/common_utils.py
"""
دوال مشتركة صغيرة كانت مكرّرة (بنفس الاسم وبنفس المنطق تقريبًا) في أكثر
من ملف داخل المشروع: evidence_judge.py, explainability.py, pipeline.py,
tools.py, clinical_signals.py, retriever.py, agent_controller.py,
hybrid_feedback.py, semantic_feedback.py.

تم دمجها هنا في مكان واحد، وتم فعليًا تعديل كل تلك الملفات لتستورد من
هذا الملف بدل تعريف نسخة محلية خاصة بها. يعني الآن يوجد نسخة واحدة فقط
من كل دالة في كامل المشروع.

لا يوجد أي تغيير في السلوك — كل دالة هنا هي نفس المنطق (أو أشمل نسخة
موجودة أصلاً بين النسخ المكررة)، فقط تم توحيدها في مكان واحد.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict

import numpy as np
import yaml

# نمط Regex يطابق أي حرف عربي (المدى Unicode لحروف اللغة العربية).
_AR_RE = re.compile(r"[\u0600-\u06FF]")


def _contains_arabic(text: str) -> bool:
    """
    يفحص إذا كان النص يحتوي على حرف عربي واحد على الأقل.

    الاستخدام في المشروع:
        تُستدعى في بداية معالجة "الحالة السريرية" (case text) القادمة من
        المستخدم، لتحديد هل النص مكتوب بالعربي (وبالتالي يحتاج ترجمة أو
        معالجة خاصة قبل تمريره لموديلات الاسترجاع/التوليد التي تعمل
        بالإنجليزي)، أو أنه مكتوب أصلاً بالإنجليزي فتتم معالجته مباشرة.
        كانت معرّفة بنفس الشكل حرفيًا في: clinical_signals.py, pipeline.py,
        retriever.py — الآن نسخة واحدة فقط هنا.

    Parameters
    ----------
    text : str
        النص المُدخل المطلوب فحصه (مثلاً وصف حالة المريض). يقبل None أو
        نص فارغ بدون أن يسبب خطأ.

    Returns
    -------
    bool
        True إذا وُجد حرف عربي واحد على الأقل داخل النص، وإلا False.
    """
    return bool(_AR_RE.search(text or ""))


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    يحوّل أي قيمة إلى float بأمان، بدون ما يكسر البرنامج لو القيمة غير
    صالحة (نص فاضي، None، نص غير رقمي، أو رقم غير منتهٍ NaN/Infinity).

    الاستخدام في المشروع:
        تُستخدم بكثرة في evidence_judge.py و explainability.py و
        pipeline.py و tools.py عند قراءة درجات الثقة/الدعم (support
        score) القادمة من نتائج الاسترجاع أو من قواميس JSON قد تحتوي
        قيمًا مفقودة أو تالفة، فبدل ما يصير Exception يرجّع قيمة افتراضية
        آمنة (عادة 0.0) ويكمل التنفيذ عادي.

    Parameters
    ----------
    x : Any
        القيمة المطلوب تحويلها لعدد عشري. ممكن تكون رقم، نص، None، أو أي
        نوع آخر.
    default : float, اختياري (افتراضي 0.0)
        القيمة التي تُرجَع في حال فشل التحويل (قيمة غير رقمية، أو NaN/Inf).

    Returns
    -------
    float
        القيمة العشرية الناتجة، أو `default` إذا تعذّر التحويل.
    """
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _low(x: Any) -> str:
    """
    يحوّل أي قيمة إلى نص صغير الأحرف (lowercase) بعد إزالة المسافات
    الزائدة من الطرفين، بدون أن يفشل لو القيمة None.

    الاستخدام في المشروع:
        دالة مساعدة صغيرة تُستخدم في evidence_judge.py و
        explainability.py قبل أي عملية مطابقة نصية (مثل البحث عن كلمة
        مفتاحية داخل نص الحالة أو داخل عنوان مصدر طبي)، حتى تكون
        المقارنة غير حساسة لحالة الأحرف (Case) ولا للمسافات الزائدة.

    Parameters
    ----------
    x : Any
        القيمة المطلوب تحويلها لنص. ممكن تكون None أو أي نوع قابل
        للتحويل إلى str.

    Returns
    -------
    str
        النص بعد التحويل لحروف صغيرة وإزالة المسافات الزائدة، أو نص
        فارغ "" إذا كانت القيمة None.
    """
    return str(x or "").strip().lower()


def _load_cfg() -> Dict[str, Any]:
    """
    يقرأ ملف الإعدادات config.yaml من المسار الحالي ويرجّعه كقاموس
    بايثون (dict).

    الاستخدام في المشروع:
        تُستدعى من agent_controller.py و clinical_signals.py (وأي ملف
        يحتاج قراءة الإعدادات مباشرة بدل الاعتماد على كائن CFG محمّل
        مسبقًا) لجلب قيم مثل عتبات الثقة (thresholds)، تفعيل/تعطيل
        الميزات (مثل agentic_mode أو clinical_signals)، وأسماء الموديلات.

    Parameters
    ----------
    (لا يوجد) — الدالة لا تأخذ أي معطيات، وتقرأ دائمًا ملف "config.yaml"
    من مجلد التشغيل الحالي.

    Returns
    -------
    Dict[str, Any]
        محتوى ملف config.yaml كقاموس. إذا لم يوجد الملف أو حدث أي خطأ
        أثناء القراءة (ملف تالف، صلاحيات، إلخ)، تُرجِع قاموسًا فارغًا {}
        بدل ما تكسر البرنامج بالكامل.
    """
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _as_float32_2d(vec: Any) -> np.ndarray:
    """
    يحوّل أي متجه (vector) إلى مصفوفة NumPy من نوع float32 وبشكل ثنائي
    الأبعاد (2D) دائمًا — حتى لو كان المُدخل متجهًا أحادي البعد (1D).

    الاستخدام في المشروع:
        تُستخدم في hybrid_feedback.py قبل تمرير أي متجه (embedding) إلى
        فهرس FAISS للبحث بالتشابه، لأن FAISS يتوقع دائمًا مصفوفة بشكل
        (عدد العينات × عدد الأبعاد) وليس متجهًا مسطحًا، وبنوع float32
        تحديدًا (وليس float64 الافتراضي في numpy).

    Parameters
    ----------
    vec : Any
        المتجه المُدخل (مثلاً ناتج تضمين نصي من موديل embeddings)، على
        شكل list أو numpy array بأي عدد أبعاد (1D أو 2D).

    Returns
    -------
    np.ndarray
        نفس المتجه بعد التحويل إلى float32، وبشكل ثنائي الأبعاد دائمًا
        (إذا كان مُدخلًا كمتجه 1D بطول N، يصير شكله (1, N)).
    """
    arr = np.asarray(vec, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr
