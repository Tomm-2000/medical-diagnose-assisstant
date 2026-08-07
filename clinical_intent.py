# -*- coding: utf-8 -*-
"""MedRAG clinical intent/safety descriptor.

The descriptor routes the case and exposes temporal/safety facts.  It never
returns a diagnosis.  All flags are derived from the shared negation-aware
clinical context so ``no seizure`` or ``no thunderclap headache`` cannot become
positive hemorrhage cues, and a short duration cannot become TIA unless the
neurological deficit explicitly resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import re

from app.rag.clinical_context import analyze_case_context, affirmed_terms

_AR_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass
class ClinicalIntent:
    acute_neuro: bool = False
    focal_neuro: bool = False
    hemorrhage_warning: bool = False
    transient_resolved_episode: bool = False
    persistent_deficit: bool = False
    chronic_or_mimic: bool = False
    out_of_domain: bool = False
    arabic_input: bool = False
    source: str = "clinical_intent_v4_negation_temporal"
    context_profile: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def describe_clinical_intent(case_text: str) -> Dict[str, Any]:
    """Return high-level routing/safety descriptors, never a diagnosis."""
    text = str(case_text or "")
    context = analyze_case_context(text)

    out_of_domain_terms = [
        "chest pain", "st elevation", "stemi", "myocardial infarction",
        "acute coronary syndrome", "coronary",
    ]
    out_of_domain = bool(affirmed_terms(text, out_of_domain_terms)) and not bool(context.get("focal_neuro"))

    result = ClinicalIntent(
        acute_neuro=bool(context.get("acute_onset")),
        focal_neuro=bool(context.get("focal_neuro")),
        hemorrhage_warning=bool(context.get("hemorrhage_warning")),
        transient_resolved_episode=bool(context.get("transient_resolved_episode")),
        persistent_deficit=bool(context.get("persistent_deficit")),
        chronic_or_mimic=bool(context.get("chronic_context") or context.get("mimic_present")),
        out_of_domain=out_of_domain,
        arabic_input=bool(_AR_RE.search(text)),
        context_profile=context,
    ).to_dict()

    # Backward-compatible concise fields remain at the top level.  The nested
    # profile is for auditing and regression debugging.
    return result
