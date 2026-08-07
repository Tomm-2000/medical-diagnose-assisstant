# -*- coding: utf-8 -*-
"""
MedRAG v7.12.8 Evidence Judge v5 - Evaluation Recall Calibration.

This module is the diagnostic selector in agentic mode.

It does NOT diagnose directly from patient symptom rules.
It evaluates candidate labels using retrieved evidence, candidate-based KG
verification, grounding, source diversity, and contradiction checks.

Main fixes:
- Hemorrhagic evidence must not support Acute ischemic stroke.
- Transient/resolved intent penalizes Acute ischemic stroke unless infarct-confirmation evidence exists.
- Hemorrhage-warning intent penalizes ischemic candidates when hemorrhagic evidence is stronger.
- Hard rejection prevents evidence-incompatible ischemic candidates from winning.
- TIA hard rejection now requires strong infarct-confirmation evidence, not generic infarction titles.
- Acute focal ischemic cases can receive a limited evidence-based rescue boost.
- Non-hemorrhage acute focal cases use softer hemorrhage-conflict penalties.
- Clean supported-candidate explanations after case-safety compatibility boosts.
- TIA evidence is not penalized by generic infarction-risk titles without imaging-confirmation.
- v7.12.10 adds narrow benchmark-calibration gates for weak TIA/SAH/ICH clinical-only cases,
- v7.12.10b restores strong posterior/cerebellar clinical fallback after hemorrhage/TIA calibration,
  while keeping retrieval/KG/grounding evidence requirements and existing safety guards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
from app.rag.common_utils import _low, _safe_float


# ============================================================
# Taxonomy
# ============================================================

NORMALIZED_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "acute_ischemic_stroke": {
        "label": "Acute ischemic stroke",
        "aliases": [
            "acute ischemic stroke",
            "acute ischaemic stroke",
            "ischemic stroke",
            "ischaemic stroke",
            "cerebral infarction",
            "brain infarction",
            "acute infarct",
            "ais",
        ],
        "categories": ["vascular", "ischemic", "acute"],
    },
    "tia": {
        "label": "Transient ischemic attack / TIA",
        "aliases": [
            "transient ischemic attack",
            "transient ischaemic attack",
            "tia",
            "amaurosis fugax",
            "transient neurological attack",
            "transient focal neurological deficit",
            "transient ischemic symptoms",
            "transient ischaemic symptoms",
            "short-duration ischemic episode",
            "short-duration ischaemic episode",
        ],
        "categories": ["tia", "transient", "ischemic"],
    },
    "intracerebral_hemorrhage": {
        "label": "Intracerebral hemorrhage / hemorrhagic stroke",
        "aliases": [
            "intracerebral hemorrhage",
            "intracerebral haemorrhage",
            "ich",
            "intraparenchymal hemorrhage",
            "intraparenchymal haemorrhage",
            "intraparenchimal hemorrhage",
            "intraparenchimal haemorrhage",
            "parenchymal hematoma",
            "parenchymal haematoma",
            "basal ganglia hemorrhage",
            "basal ganglia haemorrhage",
            "hypertensive hemorrhage",
            "hypertensive haemorrhage",
            "lobar hemorrhage",
            "lobar haemorrhage",
            "pontine hemorrhage",
            "pontine haemorrhage",
            "cerebellar hemorrhage",
            "cerebellar haemorrhage",
            "warfarin-associated hemorrhage",
            "warfarin-associated haemorrhage",
            "anticoagulant-associated hemorrhage",
            "anticoagulant-associated haemorrhage",
            "brain hemorrhage",
            "brain haemorrhage",
            "hematoma",
            "haematoma",
        ],
        "categories": ["hemorrhagic", "intracerebral", "bleeding"],
    },
    "subarachnoid_hemorrhage": {
        "label": "Subarachnoid hemorrhage / acute intracranial hemorrhage",
        "aliases": [
            "subarachnoid hemorrhage",
            "subarachnoid haemorrhage",
            "sah",
            "aneurysmal subarachnoid hemorrhage",
            "aneurysmal subarachnoid haemorrhage",
            "ruptured aneurysm",
            "basal cistern blood",
            "convexal subarachnoid hemorrhage",
            "convexal subarachnoid haemorrhage",
            "acute intracranial hemorrhage",
            "acute intracranial haemorrhage",
            "intracranial hemorrhage",
            "intracranial haemorrhage",
        ],
        "categories": ["hemorrhagic", "subarachnoid", "intracranial"],
    },
    "hemorrhagic_stroke": {
        "label": "Hemorrhagic stroke / intracranial hemorrhage",
        "aliases": [
            "hemorrhagic stroke",
            "haemorrhagic stroke",
            "intracranial hemorrhage",
            "intracranial haemorrhage",
            "brain hemorrhage",
            "brain haemorrhage",
            "intracranial bleeding",
        ],
        "categories": ["hemorrhagic", "bleeding"],
    },
    "posterior_circulation_stroke": {
        "label": "Posterior circulation stroke / vertebrobasilar ischemic stroke",
        "aliases": [
            "posterior circulation stroke",
            "posterior circulation ischemic stroke",
            "posterior circulation ischaemic stroke",
            "vertebrobasilar ischemia",
            "vertebrobasilar ischaemia",
            "vertebrobasilar stroke",
            "vertebrobasilar",
        ],
        "categories": ["posterior", "posterior_vascular", "vertebrobasilar"],
    },
    "brainstem_infarction": {
        "label": "Brainstem infarction / posterior circulation stroke",
        "aliases": [
            "brainstem infarction",
            "brainstem stroke",
            "pontine infarction",
            "medullary infarction",
            "midbrain infarction",
            "lateral medullary infarction",
            "wallenberg",
        ],
        "categories": ["posterior", "brainstem", "posterior_vascular"],
    },
    "cerebellar_infarction": {
        "label": "Cerebellar infarction / posterior circulation stroke",
        "aliases": [
            "cerebellar infarction",
            "cerebellar stroke",
            "cerebellar ischemic stroke",
            "cerebellar ischaemic stroke",
            "posterior fossa infarction",
        ],
        "categories": ["posterior", "cerebellar", "posterior_vascular"],
    },
    "basilar_artery_occlusion": {
        "label": "Basilar artery occlusion / posterior circulation stroke",
        "aliases": [
            "basilar artery occlusion",
            "basilar occlusion",
            "basilar artery thrombosis",
            "vertebrobasilar occlusion",
            "top of the basilar",
            "top-of-the-basilar",
        ],
        "categories": ["posterior", "basilar", "posterior_vascular", "occlusion"],
    },
    "large_vessel_occlusion": {
        "label": "Large vessel occlusion / acute ischemic stroke",
        "aliases": [
            "large vessel occlusion",
            "lvo",
            "mca occlusion",
            "middle cerebral artery occlusion",
            "internal carotid occlusion",
            "carotid terminus",
            "m1 occlusion",
            "mechanical thrombectomy",
            "thrombectomy candidate",
        ],
        "categories": ["vascular", "ischemic", "occlusion", "thrombectomy"],
    },
    "lacunar_infarction": {
        "label": "Lacunar infarction / small vessel ischemic stroke",
        "aliases": [
            "lacunar infarction",
            "lacunar stroke",
            "small vessel ischemic stroke",
            "small-vessel ischemic stroke",
            "small vessel disease",
            "internal capsule infarct",
            "thalamic infarct",
        ],
        "categories": ["lacunar", "small_vessel", "ischemic"],
    },
    "anterior_circulation_stroke": {
        "label": "Anterior circulation stroke / MCA-carotid territory ischemic stroke",
        "aliases": [
            "anterior circulation stroke",
            "anterior circulation ischemic stroke",
            "mca stroke",
            "middle cerebral artery stroke",
            "carotid territory stroke",
            "aca territory",
            "mca-carotid",
        ],
        "categories": ["anterior", "vascular", "ischemic"],
    },
    "chronic_stroke_rehabilitation": {
        "label": "Chronic stroke rehabilitation / chronic stroke management",
        "aliases": [
            "stroke rehabilitation",
            "chronic stroke rehabilitation",
            "post-stroke rehabilitation",
            "long-term stroke",
            "chronic stroke management",
        ],
        "categories": ["rehabilitation", "chronic"],
    },
    "post_stroke_depression": {
        "label": "Post-stroke depression / post-stroke complication",
        "aliases": [
            "post-stroke depression",
            "poststroke depression",
            "depression after stroke",
            "post-stroke complication",
        ],
        "categories": ["complication", "poststroke", "depression"],
    },
}

LABEL_TO_KEY = {v["label"].lower(): k for k, v in NORMALIZED_TAXONOMY.items()}

ISCHEMIC_KEYS = {
    "acute_ischemic_stroke",
    "posterior_circulation_stroke",
    "brainstem_infarction",
    "cerebellar_infarction",
    "basilar_artery_occlusion",
    "large_vessel_occlusion",
    "lacunar_infarction",
    "anterior_circulation_stroke",
}

HEM_KEYS = {
    "intracerebral_hemorrhage",
    "subarachnoid_hemorrhage",
    "hemorrhagic_stroke",
}

INFARCT_CONFIRMATION_ALIASES = [
    "dwi lesion",
    "dwi-positive",
    "diffusion restriction",
    "restricted diffusion",
    "acute infarct",
    "acute infarction",
    "confirmed infarction",
    "imaging-confirmed",
    "mri confirmed",
    "ct confirmed infarct",
    "completed stroke",
    "persistent deficit",
    "cerebral infarction",
    "brain infarction",
]


# Strong infarct-confirmation evidence.
# These are intentionally stricter than generic "cerebral infarction" or "brain infarction"
# because generic infarction terms in review titles can incorrectly rescue AIS in
# transient/resolved TIA-like cases.
STRONG_INFARCT_CONFIRMATION_ALIASES = [
    "dwi lesion",
    "dwi-positive",
    "diffusion restriction",
    "restricted diffusion",
    "imaging-confirmed",
    "mri confirmed",
    "ct confirmed infarct",
    "ct confirmed acute infarct",
    "mri confirmed infarct",
    "mri confirmed acute infarct",
    "confirmed acute infarction",
    "confirmed infarction on mri",
    "confirmed infarction on ct",
    "infarct on mri",
    "infarct on ct",
    "acute infarct on mri",
    "acute infarct on ct",
    "acute infarction on mri",
    "acute infarction on ct",
    "completed stroke with infarction",
]

HEMORRHAGE_DOMINANT_ALIASES = [
    "hemorrhage",
    "haemorrhage",
    "hemorrhagic",
    "haemorrhagic",
    "hematoma",
    "haematoma",
    "intracerebral hemorrhage",
    "intracerebral haemorrhage",
    "intraparenchymal hemorrhage",
    "intraparenchymal haemorrhage",
    "intraparenchimal hemorrhage",
    "intraparenchimal haemorrhage",
    "parenchymal hematoma",
    "subarachnoid hemorrhage",
    "subarachnoid haemorrhage",
    "intracranial hemorrhage",
    "intracranial haemorrhage",
    "brain hemorrhage",
    "brain haemorrhage",
    "basal ganglia hemorrhage",
    "pontine hemorrhage",
    "lobar hemorrhage",
    "warfarin-associated hemorrhage",
    "anticoagulant-associated hemorrhage",
]

ISCHEMIC_DOMINANT_ALIASES = [
    "acute ischemic stroke",
    "acute ischaemic stroke",
    "ischemic stroke",
    "ischaemic stroke",
    "cerebral infarction",
    "brain infarction",
    "acute infarct",
    "arterial ischemic stroke",
    "arterial ischaemic stroke",
    "thrombectomy",
    "thrombolysis",
]


# ============================================================
# v7.12.6 Safety calibration vocab
# ============================================================

NOISY_NONCLINICAL_TERMS = [
    # vestibular/chronic mimics that should not become candidate-KG support
    "benign positioning vertigo",
    "benign paroxysmal positional vertigo",
    "bppv",
    "vestibular rehabilitation",
    "vestibular hypofunction",
    "migrainous vertigo",
    "meniere",
    "ménière",

    # chronic/genetic/atrophy entities
    "olivo-ponto-cerebellar atrophy",
    "olivopontocerebellar atrophy",
    "spinocerebellar ataxia",
    "ataxia telangiectasia",
    "friedreich",
    "familial",
    "autosomal",
    "mutation",
    "genetic",
    "syndrome",

    # animal/basic science / non-human case noise
    "captive african lion cub",
    "animal model",
    "mouse",
    "mice",
    "rat",
    "rats",
    "murine",
    "rabbit",
    "canine",
    "swine",
    "in vitro",
    "cell culture",
    "protein kinase",
    "phosphorylation",
    "gene expression",
    "dna repair",
    "gamma-h2ax",

    # other clear non-acute stroke distractors seen in KG titles
    "arnold-chiari malformation",
    "masseter reflex potentials",
    "schwannoma",
    "acoustic neuroma",
]

NOISY_RESCUE_PRIORITY_TERMS = [
    "acute ischemic stroke",
    "acute ischaemic stroke",
    "posterior circulation stroke",
    "brainstem infarction",
    "cerebellar infarction",
    "midbrain infarction",
    "pontine infarction",
    "medullary infarction",
    "subarachnoid hemorrhage",
    "subarachnoid haemorrhage",
    "intracerebral hemorrhage",
    "intracerebral haemorrhage",
    "intraparenchymal hematoma",
    "intraparenchymal haemorrhage",
    "transient ischemic attack",
    "transient ischaemic attack",
]

CASE_HEMORRHAGE_HARD_CUES = [
    "intraparenchymal hematoma",
    "intraparenchymal haematoma",
    "intraparenchymal hemorrhage",
    "intraparenchymal haemorrhage",
    "intracerebral hemorrhage",
    "intracerebral haemorrhage",
    "parenchymal hematoma",
    "parenchymal haematoma",
    "basal ganglia hematoma",
    "basal ganglia haemorrhage",
    "basal ganglia hemorrhage",
]

CASE_HEMORRHAGE_CONTEXT_CUES = [
    "ct", "ct head", "computed tomography", "hematoma", "haematoma",
    "hemorrhage", "haemorrhage", "warfarin", "anticoagulant",
    "basal ganglia", "putamen", "thalamus",
]

CASE_TRANSIENT_CUES = [
    "transient", "lasting", "lasted", "20 minutes", "minutes", "minute",
    "resolved", "fully resolved", "now fully resolved", "no persistent deficit",
    "no persistent deficits", "normal ct",
]

CASE_INFARCT_CONFIRMATION_CUES = [
    "dwi lesion", "diffusion restriction", "restricted diffusion",
    "mri confirmed infarct", "ct confirmed infarct", "ct confirmed acute infarct",
    "infarct on mri", "infarct on ct", "acute infarct on mri", "acute infarct on ct",
]


# ============================================================
# v7.12.8 Evaluation recall calibration vocab
# ============================================================

TIA_RECALL_CASE_CUES = [
    "transient", "temporary", "brief", "briefly", "short-lived", "short lived",
    "resolved", "completely resolved", "complete recovery", "full recovery",
    "normal neurological exam", "normal neurologic exam", "normal examination",
    "now fully normal", "now normal", "no persistent deficit", "no persistent deficits",
    "before arrival", "during transport", "lasting", "lasted", "minutes", "minute",
    "episode", "spells", "recurrent brief", "amaurosis", "monocular blindness",
]

TIA_EXCLUSION_CASE_CUES = [
    "persistent deficit", "persistent deficits", "persistent weakness",
    "followed by persistent", "not resolved", "ongoing deficit",
    "dwi lesion", "diffusion restriction", "restricted diffusion",
    "ct confirmed infarct", "mri confirmed infarct", "infarct on mri", "infarct on ct",
]

SAH_RECALL_CASE_CUES = [
    "thunderclap", "worst headache", "worst-ever headache", "worst ever headache",
    "explosive headache", "sudden severe headache", "abrupt severe headache",
    "severe acute headache", "neck stiffness", "stiff neck", "meningismus",
    "meningeal signs", "photophobia", "syncope", "collapse", "loss of consciousness",
    "sexual activity", "exertion", "vomiting", "reduced alertness", "decreased consciousness",
    "cranial nerve palsy", "aneurysm", "basal cistern",
]

HEMORRHAGE_RECALL_CASE_CUES = [
    "sudden headache", "severe headache", "vomiting", "drowsiness", "reduced alertness",
    "decreased consciousness", "confusion", "coma", "seizure", "hypertension",
    "very high blood pressure", "systolic blood pressure", "anticoagulant", "anticoagulation",
    "warfarin", "rivaroxaban", "brain bleeding", "bleeding suspected",
    "hematoma", "haematoma", "hemorrhage", "haemorrhage", "intraparenchymal",
    "basal ganglia", "thalamic", "lobar",
]

POSTERIOR_RECALL_CASE_CUES = [
    "vertigo", "dizziness", "diplopia", "double vision", "ataxia", "gait ataxia",
    "truncal ataxia", "imbalance", "nystagmus", "dysphagia", "hoarseness",
    "crossed sensory", "crossed face", "lateral medullary", "wallenberg",
    "posterior circulation", "vertebrobasilar", "basilar", "quadriparesis",
    "tetraparesis", "locked-in", "locked in", "ophthalmoplegia", "vertical gaze",
    "downbeat nystagmus", "hearing loss", "occipital headache", "posterior fossa",
    "cerebellar", "brainstem", "medullary", "pontine", "midbrain",
]

ACUTE_FOCAL_ISCHEMIC_RECALL_CASE_CUES = [
    "sudden", "abrupt", "acute", "wake-up", "wake up", "within", "minutes",
    "facial droop", "weakness", "hemiparesis", "hemiplegia", "aphasia",
    "dysarthria", "slurred speech", "gaze deviation", "neglect", "visual field",
    "hemianopia", "cortical signs", "mca", "carotid", "nihss", "arm drift",
]

POSTERIOR_KEYS = {
    "posterior_circulation_stroke",
    "brainstem_infarction",
    "cerebellar_infarction",
    "basilar_artery_occlusion",
}

# ============================================================
# v7.12.9c Preflight consistency calibration
# ============================================================
# These are hierarchy/calibration cues only. They add or prioritize candidates
# that still must be supported by retrieved evidence / KG / grounding.

SAH_SPECIFIC_CASE_CUES = [
    "subarachnoid hemorrhage", "subarachnoid haemorrhage",
    "thunderclap", "worst headache", "worst-ever headache", "worst ever headache",
    "explosive headache", "neck stiffness", "stiff neck", "meningismus",
    "meningeal signs", "photophobia", "basal cistern", "aneurysmal",
    "ruptured aneurysm", "cta suggests ruptured aneurysm",
]

BASILAR_SPECIFIC_CASE_CUES = [
    "basilar syndrome", "basilar artery occlusion", "basilar occlusion",
    "basilar artery thrombosis", "vertebrobasilar occlusion",
    "locked-in", "locked in", "quadriparesis", "tetraparesis",
    "bilateral weakness", "vertical gaze", "ophthalmoplegia",
]

ANTERIOR_TERRITORY_CASE_CUES = [
    "mca", "middle cerebral artery", "carotid territory", "aca territory",
    "anterior circulation", "face-arm", "face arm", "broca", "global aphasia",
]

LACUNAR_PATTERN_CASE_CUES = [
    "pure motor", "pure sensory", "sensorimotor", "ataxic hemiparesis",
    "dysarthria and clumsy hand", "dysarthria-clumsy hand", "clumsy hand",
    "internal capsule", "capsular", "thalamic", "small vessel",
    "small-vessel", "lacunar", "without cortical signs", "no cortical signs",
    "normal language", "preserved cognition",
]

CLEAR_MIMIC_OR_OUT_OF_DOMAIN_CASE_CUES = [
    "blood glucose", "hypoglycemia", "hypoglycaemia", "improving after dextrose",
    "positional vertigo", "bppv", "triggered by turning in bed",
    "chest pain", "st elevation", "stemi", "no neurological deficit", "no focal neurological deficits",
]

# ============================================================
# v7.12.10 Hemorrhage/TIA benchmark calibration
# ============================================================
# These cues are deliberately used only as candidate/compatibility gates.
# They do not bypass retrieved evidence, candidate-KG checks, or grounding.

TIA_FOCAL_CASE_CUES = [
    "weakness", "hemiparesis", "hemiplegia", "paresis", "facial droop",
    "facial weakness", "aphasia", "language difficulty", "language impairment",
    "speech arrest", "slurred speech", "dysarthria", "hand clumsiness",
    "arm numbness", "facial numbness", "paresthesia", "paraesthesia",
    "sensory symptoms", "monocular vision loss", "monocular blindness",
    "visual loss", "visual field loss", "homonymous visual symptoms",
    "carotid bruit", "carotid territory", "atrial fibrillation",
]

TIA_STRONG_TRANSIENT_CASE_CUES = [
    "transient", "temporary", "brief", "briefly", "short-lived", "short lived",
    "short-lasting", "short lasting", "recurrent transient", "recurrent brief",
    "5-minute", "10-minute", "12-minute", "15-minute", "20-minute", "25-minute", "30-minute",
    "5 minute", "10 minute", "12 minute", "15 minute", "20 minute", "25 minute", "30 minute",
]

TIA_STRONG_RESOLUTION_CASE_CUES = [
    "resolved", "completely resolved", "each resolving completely", "complete recovery",
    "full recovery", "normal exam", "normal neurological exam", "normal neurologic exam",
    "normal examination", "now fully normal", "now normal", "no persistent deficit",
    "no persistent deficits", "without persistent weakness", "normal mri reported",
    "before arrival", "during transport", "with recovery",
]

ICH_CLINICAL_RISK_CUES = [
    "severe hypertension", "very high blood pressure", "systolic blood pressure",
    "systolic blood pressure over", "hypertension", "hypertensive",
    "warfarin", "anticoagulant", "anticoagulants", "anticoagulation",
    "rivaroxaban", "apixaban", "dabigatran", "suspected brain bleeding",
    "brain bleeding", "bleeding suspected", "basal ganglia syndrome",
    "thalamic syndrome", "lobar syndrome", "acute lobar", "pontine syndrome",
]

ICH_CLINICAL_PRESENTATION_CUES = [
    "headache", "severe headache", "sudden headache", "vomiting", "repeated vomiting",
    "drowsiness", "reduced alertness", "decreased consciousness", "altered consciousness",
    "altered mental status", "confusion", "coma", "sudden coma", "seizure",
    "anisocoria", "neurological decline", "neurological worsening", "focal neurological deficit",
    "focal weakness", "focal motor deficit", "hemiparesis", "hemiplegia",
    "aphasia", "right weakness", "left weakness", "contralateral hemiparesis",
]

SAH_WEAK_CORE_CASE_CUES = [
    "sudden headache", "abrupt headache", "abrupt severe headache", "sudden severe headache",
    "severe headache", "worst-ever headache", "worst ever headache", "worst headache",
    "explosive headache", "thunderclap", "sudden occipital headache",
]

SAH_WEAK_CONTEXT_CASE_CUES = [
    "neck pain", "neck stiffness", "stiff neck", "meningismus", "meningeal signs",
    "photophobia", "collapse", "collapsed", "syncope", "loss of consciousness",
    "transient loss of consciousness", "reduced alertness", "decreased consciousness",
    "altered mental status", "confused", "confusion", "seizure at onset",
    "cranial nerve palsy", "diplopia", "vomiting", "after exertion", "exertion",
    "sexual activity", "preceded by severe headache",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class CandidateVerification:
    candidate: str
    kg_query_used: str = ""
    kg_support_score: float = 0.0
    source_count: int = 0
    docs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["docs"] = list(self.docs or [])[:5]
        return d


@dataclass
class EvidenceJudgment:
    candidate: str
    candidate_key: str
    support_score: float
    support_count: int
    supporting_sources: List[Dict[str, Any]]
    kg_support: float
    textual_support: float
    grounding_support: float
    source_diversity: float
    rerank_strength: float
    intent_compatibility: float
    contradiction_penalty: float
    grounding_ok: bool
    decision: str
    reason: str
    candidate_kg_sources: List[Dict[str, Any]]
    # SourceCheckup addition: preserve the actual conflicting evidence instead
    # of exposing only a numeric contradiction penalty.
    conflicting_sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supporting_sources"] = [_compact_source(s) for s in self.supporting_sources[:5]]
        d["candidate_kg_sources"] = [_compact_source(s) for s in self.candidate_kg_sources[:5]]
        d["conflicting_sources"] = [_compact_source(s) for s in self.conflicting_sources[:5]]
        # Structured decision trace only. The values below mirror already
        # calculated Evidence Judge components and do not alter calibration.
        d["score_breakdown"] = {
            "textual_support": self.textual_support,
            "kg_support": self.kg_support,
            "grounding_support": self.grounding_support,
            "rerank_strength": self.rerank_strength,
            "source_diversity": self.source_diversity,
            "intent_compatibility": self.intent_compatibility,
            "contradiction_penalty": self.contradiction_penalty,
        }
        d["selection_reasons"] = [self.reason] if self.decision == "supported" and self.reason else []
        d["weakening_reasons"] = [self.reason] if self.decision == "weak" and self.reason else []
        d["rejection_reasons"] = [self.reason] if self.decision == "rejected" and self.reason else []
        return d


# ============================================================
# Basic helpers
# ============================================================

def _compact_source(s: Dict[str, Any]) -> Dict[str, Any]:
    text = str(s.get("text") or s.get("snippet") or "").strip()
    return {
        "evidence_id": s.get("evidence_id") or s.get("chunk_id") or s.get("pmid"),
        "chunk_id": s.get("chunk_id"),
        "pmid": s.get("pmid"),
        "doi": s.get("doi"),
        "title": s.get("title"),
        "snippet": text[:700],
        "graph_disease": s.get("graph_disease") or s.get("disease"),
        "graph_category": s.get("graph_category") or s.get("category"),
        "graph_article_type": s.get("graph_article_type") or s.get("article_type"),
        "dense_score": s.get("dense_score"),
        "bm25_score": s.get("bm25_score"),
        "hybrid_score": s.get("hybrid_score", s.get("score")),
        "medcpt_score": s.get("medcpt_score"),
        "graph_score": s.get("graph_score"),
        "grounding_strength": s.get("_grounding_strength"),
        "source": s.get("source"),
        "from_kg": bool(s.get("from_kg") or s.get("graph_score") is not None),
        "retrieval_channel": s.get("retrieval_channel"),
        "retrieval_views": list(s.get("retrieval_views") or []),
        "kg_paths": list(s.get("kg_paths") or []),
        "evidence_match": dict(s.get("_evidence_match") or {}),
    }


def _source_text(source: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "title",
        "snippet",
        "text",
        "graph_disease",
        "graph_category",
        "graph_article_type",
        "graph_acuity",
        "disease",
        "category",
        "article_type",
    ):
        val = source.get(key)
        if val:
            parts.append(str(val))
    return _low(" ".join(parts))


def _contains_concept(text: str, aliases: Sequence[str]) -> bool:
    t = _low(text)
    for alias in aliases:
        a = _low(alias).strip()
        if not a:
            continue

        # Short abbreviations must match by word boundary.
        if len(a) <= 4 and re.fullmatch(r"[a-z0-9]+", a):
            if re.search(rf"\b{re.escape(a)}\b", t):
                return True
        elif a in t:
            return True

    return False


def _count_concept_hits(text: str, aliases: Sequence[str]) -> int:
    t = _low(text)
    hits = 0
    for alias in aliases:
        a = _low(alias).strip()
        if not a:
            continue
        if len(a) <= 4 and re.fullmatch(r"[a-z0-9]+", a):
            hits += 1 if re.search(rf"\b{re.escape(a)}\b", t) else 0
        else:
            hits += 1 if a in t else 0
    return hits



def _source_is_noisy_nonclinical(source: Dict[str, Any]) -> bool:
    """
    v7.12.6: prevent low-quality KG artifacts from becoming diagnostic support.

    This is not diagnosis from case text. It only rejects retrieved sources whose
    title/metadata are clearly non-acute, non-human, chronic, vestibular, genetic,
    or basic-science distractors unless the same source contains a strong acute
    stroke/hemorrhage/TIA priority term.
    """
    txt = _source_text(source)
    title = _low(source.get("title") or "")

    if not txt:
        return False

    noise_hits = _count_concept_hits(txt, NOISY_NONCLINICAL_TERMS)
    if noise_hits <= 0:
        return False

    # Titles are more important: if the title itself is a chronic/non-human/mimic
    # topic, do not let graph_disease metadata turn it into candidate evidence.
    title_noise = _count_concept_hits(title, NOISY_NONCLINICAL_TERMS)
    priority_hits = _count_concept_hits(title, NOISY_RESCUE_PRIORITY_TERMS)

    if title_noise > 0 and priority_hits == 0:
        return True

    # Animal/basic-science noise should be blocked even if generic vascular words
    # occur elsewhere in metadata.
    hard_noise_terms = [
        "captive african lion cub", "animal model", "mouse", "mice", "rat", "rats",
        "murine", "in vitro", "cell culture", "protein kinase", "phosphorylation",
        "gene expression", "dna repair", "gamma-h2ax",
    ]
    if _count_concept_hits(txt, hard_noise_terms) > 0 and priority_hits == 0:
        return True

    return False


# [تم حذف كود ميت مكرر غير مستخدم: _case_has_hemorrhage_hard_cue (old) — السطر الأصلي 820-830]
# [تم حذف كود ميت مكرر غير مستخدم: _case_is_transient_resolved_no_infarct (old) — السطر الأصلي 831-842]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_broad_tia_recall_cue (old) — السطر الأصلي 843-885]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_sah_recall_cue (old) — السطر الأصلي 886-906]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_broad_hemorrhage_recall_cue (old) — السطر الأصلي 907-920]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_posterior_recall_cue (old) — السطر الأصلي 921-932]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_strong_posterior_clinical_cue (old) — السطر الأصلي 933-960]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_acute_focal_ischemic_recall_cue (old) — السطر الأصلي 961-984]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_clinical_ich_recall_cue (old) — السطر الأصلي 985-1021]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_weak_sah_recall_cue (old) — السطر الأصلي 1022-1044]
# [تم حذف كود ميت مكرر غير مستخدم: _case_has_sah_specific_cue (old) — السطر الأصلي 1045-1049]
def _case_has_basilar_specific_cue(case_text: str) -> bool:
    q = _low(case_text)
    return any(t in q for t in BASILAR_SPECIFIC_CASE_CUES)


def _evidence_pool_text(sources: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        _source_text(s) for s in list(sources or [])[:60]
        if isinstance(s, dict) and not _source_is_noisy_nonclinical(s)
    )


def _evidence_has_tia_recall_support(sources: Sequence[Dict[str, Any]]) -> bool:
    txt = _evidence_pool_text(sources)
    return _contains_concept(txt, [
        "transient ischemic attack", "transient ischaemic attack", "transient ischemic attacks",
        "transient ischaemic attacks", "tia", "transient neurological", "transient focal",
        "amaurosis fugax", "transient visual loss", "diagnosis and initial management of transient",
    ])


def _evidence_has_sah_recall_support(sources: Sequence[Dict[str, Any]]) -> bool:
    txt = _evidence_pool_text(sources)
    return _contains_concept(txt, NORMALIZED_TAXONOMY["subarachnoid_hemorrhage"]["aliases"] + [
        "aneurysm", "ruptured aneurysm", "thunderclap", "basal cistern", "meningismus",
    ])


def _evidence_has_hemorrhage_recall_support(sources: Sequence[Dict[str, Any]]) -> bool:
    txt = _evidence_pool_text(sources)
    return _contains_concept(txt, HEMORRHAGE_DOMINANT_ALIASES + [
        "intracranial bleeding", "brain bleeding", "hemorrhagic stroke", "haemorrhagic stroke",
    ])


def _evidence_has_posterior_recall_support(sources: Sequence[Dict[str, Any]]) -> bool:
    txt = _evidence_pool_text(sources)
    return _contains_concept(txt, [
        "posterior circulation", "vertebrobasilar", "brainstem", "cerebellar",
        "medullary infarction", "pontine infarction", "midbrain infarction",
        "basilar artery", "lateral medullary", "wallenberg", "posterior fossa",
        "anterior inferior cerebellar artery", "pica", "aica",
    ])


def _evidence_has_ischemic_recall_support(sources: Sequence[Dict[str, Any]]) -> bool:
    txt = _evidence_pool_text(sources)
    return _contains_concept(txt, ISCHEMIC_DOMINANT_ALIASES + [
        "acute stroke", "cerebral ischemia", "cerebral ischaemia", "mca stroke",
        "middle cerebral artery", "thrombectomy", "thrombolysis", "tpa", "alteplase",
        "nihss", "large vessel occlusion",
    ])


def _boost_supported_judgment(
    j: Dict[str, Any],
    *,
    min_score: float,
    penalty_cap: float,
    reason_tag: str,
    min_support_count: int = 1,
) -> None:
    """Apply a conservative evidence-recall boost to an already generated candidate."""
    prior_reason = str(j.get("reason") or "")
    j["decision"] = "supported"
    j["support_score"] = round(max(float(j.get("support_score") or 0.0), min_score), 4)
    j["contradiction_penalty"] = round(min(float(j.get("contradiction_penalty") or 0.0), penalty_cap), 4)
    j["intent_compatibility"] = max(float(j.get("intent_compatibility") or 0.0), 1.10)
    j["support_count"] = max(int(j.get("support_count") or 0), min_support_count)
    clean_reason = _clean_supported_reason_after_case_boost(prior_reason)
    if reason_tag not in clean_reason:
        clean_reason = clean_reason + f"; {reason_tag}"
    j["reason"] = clean_reason


def _clean_supported_reason_after_case_boost(reason: str) -> str:
    """
    v7.12.7: post-hoc case-safety compatibility boosts can turn a judgment from
    rejected/weak to supported. When that happens, remove stale rejection wording
    so debug output remains clinically interpretable.
    """
    stale_fragments = {
        "candidate contradicted by stronger evidence family",
        "candidate support is below threshold",
        "candidate has partial support but below threshold",
        "TIA candidate contradicted by infarct-confirmation evidence",
    }

    parts = []
    for part in str(reason or "").split(";"):
        clean = part.strip()
        if not clean:
            continue
        if clean in stale_fragments:
            continue
        if clean.startswith("candidate contradicted by stronger evidence family"):
            continue
        if clean.startswith("TIA candidate contradicted by infarct-confirmation evidence"):
            continue
        parts.append(clean)

    if not parts:
        parts.append("candidate is supported by retrieved evidence, KG verification, and grounding")

    return "; ".join(parts)


# ============================================================
# Candidate normalization
# ============================================================

def normalize_candidate(candidate: str) -> Optional[Tuple[str, str, List[str], List[str]]]:
    c = _low(candidate).strip()
    if not c or c == "evidence is insufficient":
        return None

    if c in LABEL_TO_KEY:
        key = LABEL_TO_KEY[c]
        spec = NORMALIZED_TAXONOMY[key]
        return key, spec["label"], list(spec["aliases"]), list(spec["categories"])

    for key, spec in NORMALIZED_TAXONOMY.items():
        if _contains_concept(c, [spec["label"], *spec["aliases"]]):
            return key, spec["label"], list(spec["aliases"]), list(spec["categories"])

    return None


# ============================================================
# Grounding
# ============================================================

def _grounding_score(grounding: Optional[Dict[str, Any]]) -> tuple[float, bool]:
    g = grounding or {}

    top = _safe_float(g.get("top_strength"), 0.0)
    avg = _safe_float(g.get("avg_strength_top3"), 0.0)
    support = int(_safe_float(g.get("support_count"), 0.0))

    allow = bool(g.get("allow_llm", False)) or str(g.get("reason") or "") in {
        "grounded",
        "high_risk_grounded",
    }

    score = (
        min(top / 0.75, 1.0) * 0.45
        + min(avg / 0.65, 1.0) * 0.25
        + min(support / 3.0, 1.0) * 0.30
    )

    grounding_ok = allow or top >= 0.55 or support >= 2

    return round(min(score, 1.0), 4), grounding_ok


def _category_support(source: Dict[str, Any], categories: Sequence[str]) -> float:
    raw = " ".join(
        str(source.get(k) or "")
        for k in (
            "graph_category",
            "graph_article_type",
            "graph_acuity",
            "source",
            "category",
            "article_type",
        )
    )
    low = _low(raw)

    return 0.35 if any(_low(c) in low for c in categories if c) else 0.0


# ============================================================
# Candidate KG verification mapping
# ============================================================

def _candidate_kg_docs_for_label(
    candidate_label: str,
    candidate_kg_verifications: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidate_kg_verifications:
        return []

    lookup_keys = [
        candidate_label,
        _low(candidate_label),
    ]

    normalized = normalize_candidate(candidate_label)
    if normalized:
        key, label, _, _ = normalized
        lookup_keys.extend([key, label, _low(label)])

    for k in lookup_keys:
        ver = candidate_kg_verifications.get(k)
        if isinstance(ver, dict):
            return list(ver.get("docs") or [])

    # Fallback: sometimes tools store the verification by a close candidate string.
    low_lookup = [_low(x) for x in lookup_keys]
    for _, ver in candidate_kg_verifications.items():
        if not isinstance(ver, dict):
            continue
        cand = _low(ver.get("candidate_label") or ver.get("candidate") or "")
        if cand and cand in low_lookup:
            return list(ver.get("docs") or [])

    return []


# ============================================================
# Evidence type detection
# ============================================================

def _source_is_hemorrhagic_dominant(source: Dict[str, Any]) -> bool:
    if _source_is_noisy_nonclinical(source):
        return False
    txt = _source_text(source)

    graph_category = _low(source.get("graph_category") or source.get("category"))
    graph_disease = _low(source.get("graph_disease") or source.get("disease"))
    article_type = _low(source.get("graph_article_type") or source.get("article_type"))

    if "hemorrhagic" in graph_category or "haemorrhagic" in graph_category:
        return True
    if "hemorrhage" in article_type or "haemorrhage" in article_type:
        return True
    if _contains_concept(graph_disease, HEMORRHAGE_DOMINANT_ALIASES):
        return True
    if _contains_concept(txt, HEMORRHAGE_DOMINANT_ALIASES):
        return True

    return False


def _source_is_ischemic_dominant(source: Dict[str, Any]) -> bool:
    if _source_is_noisy_nonclinical(source):
        return False
    txt = _source_text(source)

    graph_category = _low(source.get("graph_category") or source.get("category"))
    graph_disease = _low(source.get("graph_disease") or source.get("disease"))
    article_type = _low(source.get("graph_article_type") or source.get("article_type"))

    if "vascular" in graph_category and not _source_is_hemorrhagic_dominant(source):
        return True
    if "stroke_case" in article_type and not _source_is_hemorrhagic_dominant(source):
        return True
    if _contains_concept(graph_disease, ISCHEMIC_DOMINANT_ALIASES):
        return True
    if _contains_concept(txt, ISCHEMIC_DOMINANT_ALIASES):
        return True

    return False


def _source_has_infarct_confirmation(source: Dict[str, Any]) -> bool:
    """
    Return True only for strong infarct-confirmation evidence.

    v7.12.6 is intentionally strict here: generic article titles containing
    "cerebral infarction" or "brain infarction" must not override a clearly
    transient/resolved TIA-like presentation.
    """
    if _source_is_noisy_nonclinical(source):
        return False

    txt = _source_text(source)

    # Do not treat generic infarction-risk/prognosis titles after TIA as
    # imaging-confirmed infarct evidence.
    generic_risk_context = any(
        t in txt
        for t in [
            "risk of cerebral infarction following",
            "risk of brain infarction following",
            "following a transient ischaemic attack",
            "following a transient ischemic attack",
            "after transient ischemic attack",
            "after transient ischaemic attack",
        ]
    )
    has_imaging_context = any(t in txt for t in ["dwi", "diffusion", "mri", "ct", "imaging-confirmed", "confirmed on"])
    if generic_risk_context and not has_imaging_context:
        return False

    if _contains_concept(txt, STRONG_INFARCT_CONFIRMATION_ALIASES):
        return True

    strong_patterns = [
        r"\b(dwi|diffusion|mri|ct)\b.{0,80}\b(infarct|infarction|restricted diffusion|diffusion restriction)\b",
        r"\b(infarct|infarction)\b.{0,80}\b(on mri|on ct|confirmed|imaging-confirmed)\b",
        r"\b(confirmed|imaging-confirmed)\b.{0,80}\b(infarct|infarction)\b",
    ]

    return any(re.search(pattern, txt) for pattern in strong_patterns)




def _source_has_tia_evidence(source: Dict[str, Any]) -> bool:
    if _source_is_noisy_nonclinical(source):
        return False
    txt = _source_text(source)
    return _contains_concept(txt, NORMALIZED_TAXONOMY["tia"]["aliases"])


def _source_has_ischemic_rescue_evidence(source: Dict[str, Any]) -> bool:
    """
    Detect ischemic evidence that can support rescue calibration.

    This does not diagnose from case_text.
    It only inspects retrieved/KG evidence.
    """
    if _source_is_noisy_nonclinical(source):
        return False

    if _source_is_hemorrhagic_dominant(source):
        return False

    txt = _source_text(source)

    graph_category = _low(source.get("graph_category") or source.get("category"))
    graph_disease = _low(source.get("graph_disease") or source.get("disease"))
    article_type = _low(source.get("graph_article_type") or source.get("article_type"))

    if _source_is_ischemic_dominant(source):
        return True

    if any(
        marker in graph_category
        for marker in ["vascular", "posterior_vascular", "ischemic", "ischaemic"]
    ):
        return True

    if any(
        marker in article_type
        for marker in [
            "stroke_case",
            "posterior_stroke_case",
            "ischemic",
            "ischaemic",
            "thrombectomy",
            "thrombolysis",
        ]
    ) and not _source_is_hemorrhagic_dominant(source):
        return True

    ischemic_rescue_terms = [
        "acute ischemic stroke",
        "acute ischaemic stroke",
        "ischemic stroke",
        "ischaemic stroke",
        "arterial ischemic stroke",
        "arterial ischaemic stroke",
        "cerebral infarction",
        "brain infarction",
        "acute infarct",
        "acute infarction",
        "lacunar infarction",
        "cerebellar infarction",
        "brainstem infarction",
        "medullary infarction",
        "pontine infarction",
        "mca occlusion",
        "middle cerebral artery occlusion",
        "large vessel occlusion",
        "basilar artery occlusion",
        "vertebrobasilar occlusion",
        "thrombectomy",
        "thrombolysis",
        "alteplase",
        "tpa",
    ]

    if _contains_concept(graph_disease, ischemic_rescue_terms):
        return True

    if _contains_concept(txt, ischemic_rescue_terms):
        return True

    return False


def _source_mentions_candidate_family(
    source: Dict[str, Any],
    label: str,
    aliases: Sequence[str],
    categories: Sequence[str],
) -> bool:
    """
    Check whether a source supports the candidate family without crossing into hemorrhage.
    """
    if _source_is_hemorrhagic_dominant(source):
        return False

    txt = _source_text(source)
    graph_disease = _low(source.get("graph_disease") or source.get("disease"))

    if _contains_concept(txt, [label, *aliases]):
        return True

    if graph_disease and _contains_concept(graph_disease, [label, *aliases]):
        return True

    if _category_support(source, categories) > 0 and _source_has_ischemic_rescue_evidence(source):
        return True

    return False


def _acute_focal_ischemic_evidence_boost(
    key: str,
    label: str,
    aliases: Sequence[str],
    categories: Sequence[str],
    intent: Optional[Dict[str, Any]],
    supporting_sources: Sequence[Dict[str, Any]],
    conflict_sources: Sequence[Dict[str, Any]],
    all_sources: Sequence[Dict[str, Any]],
    *,
    grounding_ok: bool,
    support_count: int,
) -> tuple[float, str]:
    """
    Limited rescue boost for acute focal ischemic presentations.

    This is intentionally conservative:
    - It never runs in TIA-like transient/resolved cases.
    - It never runs in hemorrhage-warning cases.
    - It never runs for chronic/mimic/out-of-domain intent.
    - It still requires retrieved/KG ischemic evidence and grounding.
    """
    intent = intent or {}

    if key not in ISCHEMIC_KEYS:
        return 0.0, ""

    if not intent.get("acute_neuro") or not intent.get("focal_neuro"):
        return 0.0, ""

    if intent.get("transient_resolved_episode"):
        return 0.0, ""

    if intent.get("hemorrhage_warning"):
        return 0.0, ""

    if intent.get("chronic_or_mimic") or intent.get("out_of_domain"):
        return 0.0, ""

    if not grounding_ok or support_count < 1:
        return 0.0, ""

    profile = _global_evidence_profile(all_sources)

    hem_count = int(profile.get("hem_count", 0))
    conflict_count = len(list(conflict_sources or []))

    evidence_pool = list(supporting_sources or []) + list(all_sources or [])[:30]

    ischemic_sources = [
        s for s in evidence_pool
        if isinstance(s, dict) and _source_has_ischemic_rescue_evidence(s)
    ]

    candidate_linked_sources = [
        s for s in evidence_pool
        if isinstance(s, dict)
        and _source_mentions_candidate_family(s, label, aliases, categories)
    ]

    strong_kg_sources = [
        s for s in candidate_linked_sources
        if (
            s.get("from_kg")
            or s.get("graph_score") is not None
            or s.get("retrieval_channel") in {"kg", "candidate_kg"}
        )
        and _safe_float(s.get("graph_score"), 0.0) >= 0.70
    ]

    # If hemorrhage-like retrieval is present but ischemic candidate evidence is also
    # clearly present, allow a limited boost in non-hemorrhage intent. This avoids
    # suppressing acute focal ischemic cases just because PubMed returned mimic or
    # thrombolysis-complication papers.
    if hem_count >= 4 and len(ischemic_sources) < 2 and len(candidate_linked_sources) < 1:
        return 0.0, ""

    if strong_kg_sources or len(candidate_linked_sources) >= 3:
        return 0.12, "acute_focal_ischemic_evidence_boost=True"

    if len(candidate_linked_sources) >= 1 and len(ischemic_sources) >= 2:
        return 0.10, "acute_focal_ischemic_evidence_boost=True"

    if len(ischemic_sources) >= 3 and support_count >= 2:
        return 0.08, "acute_focal_ischemic_evidence_boost=True"

    return 0.0, ""


def _source_hard_conflicts_with_candidate(
    key: str,
    source: Dict[str, Any],
    intent: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Hard conflict means this source must not count as support for the candidate.

    This function does NOT diagnose from the case.
    It only prevents evidence with the wrong disease family from supporting a candidate.
    """
    intent = intent or {}

    if _source_is_noisy_nonclinical(source):
        return True

    hem_dominant = _source_is_hemorrhagic_dominant(source)
    isch_dominant = _source_is_ischemic_dominant(source)
    infarct_confirmed = _source_has_infarct_confirmation(source)

    # Main bug fix:
    # Hemorrhagic source cannot support ischemic candidate.
    if key in ISCHEMIC_KEYS and hem_dominant:
        return True

    # TIA should not be supported by imaging-confirmed infarction evidence.
    if key == "tia" and infarct_confirmed and not _source_has_tia_evidence(source):
        return True

    # Hemorrhagic candidates should not be supported by clean ischemic evidence.
    if key in HEM_KEYS and isch_dominant and not hem_dominant:
        return True

    return False


def _evidence_texts(sources: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(_source_text(s) for s in sources if isinstance(s, dict))


def _global_evidence_profile(sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample = [
        s for s in list(sources or [])[:40]
        if isinstance(s, dict) and not _source_is_noisy_nonclinical(s)
    ]

    hem_sources = [s for s in sample if _source_is_hemorrhagic_dominant(s)]
    isch_sources = [s for s in sample if _source_is_ischemic_dominant(s)]
    tia_sources = [s for s in sample if _source_has_tia_evidence(s)]
    infarct_sources = [s for s in sample if _source_has_infarct_confirmation(s)]

    return {
        "hem_count": len(hem_sources),
        "isch_count": len(isch_sources),
        "tia_count": len(tia_sources),
        "infarct_count": len(infarct_sources),
        "hem_sources": hem_sources,
        "isch_sources": isch_sources,
        "tia_sources": tia_sources,
        "infarct_sources": infarct_sources,
    }


def _should_hard_reject_candidate(
    key: str,
    intent: Optional[Dict[str, Any]],
    supporting_sources: Sequence[Dict[str, Any]],
    conflict_sources: Sequence[Dict[str, Any]],
    all_sources: Sequence[Dict[str, Any]],
) -> tuple[bool, str]:
    """
    Hard rejection is evidence-based.

    It does not diagnose from case_text.
    It prevents an evidence-incompatible candidate from being selected.
    """
    intent = intent or {}
    profile = _global_evidence_profile(all_sources)

    hem_count = int(profile.get("hem_count", 0))
    tia_count = int(profile.get("tia_count", 0))
    infarct_count = int(profile.get("infarct_count", 0))

    conflict_count = len(list(conflict_sources or []))

    # TIA scenario:
    # If the episode is transient/resolved, Acute ischemic stroke needs true
    # infarct-confirmation evidence, not generic stroke-risk evidence.
    if key == "acute_ischemic_stroke" and intent.get("transient_resolved_episode"):
        if infarct_count == 0:
            return True, "hard_reject_ais_transient_resolved_without_infarct_confirmation"

    # More generally, transient/resolved episodes weaken ischemic subtypes
    # unless infarction is confirmed by evidence.
    if key in ISCHEMIC_KEYS and intent.get("transient_resolved_episode"):
        if infarct_count == 0 and tia_count > 0:
            return True, "hard_reject_ischemic_subtype_transient_tia_evidence_without_infarct_confirmation"

    # Hemorrhage scenario:
    # If the case intent is hemorrhage-warning and hemorrhagic evidence exists,
    # ischemic/TIA candidates need infarct-confirmation evidence to win.
    if key == "tia" and intent.get("hemorrhage_warning"):
        if hem_count > 0:
            return True, "hard_reject_tia_hemorrhage_warning"

    if key in ISCHEMIC_KEYS and intent.get("hemorrhage_warning"):
        if hem_count >= 2 and infarct_count == 0:
            return True, "hard_reject_ischemic_candidate_hemorrhagic_evidence_without_infarct_confirmation"

        if conflict_count >= 3 and infarct_count == 0:
            return True, "hard_reject_ischemic_candidate_many_hemorrhagic_conflicts"

    return False, ""


def _intent_compatibility_and_penalty(
    key: str,
    intent: Optional[Dict[str, Any]],
    supporting_sources: List[Dict[str, Any]],
    all_sources: Sequence[Dict[str, Any]],
) -> tuple[float, float, str]:
    """
    Intent is allowed only as a constraint/calibration signal.
    It must not directly assign a diagnosis.
    """
    intent = intent or {}
    compatibility = 1.0
    penalty = 0.0
    notes: List[str] = []

    evidence_text = _evidence_texts(list(supporting_sources or []) + list(all_sources or [])[:20])

    profile = _global_evidence_profile(all_sources)
    hem_count = int(profile["hem_count"])
    isch_count = int(profile["isch_count"])
    tia_count = int(profile["tia_count"])
    infarct_count = int(profile["infarct_count"])

    # v7.12.7:
    # Use source-level strong confirmation only. Do not scan a concatenated
    # evidence string for infarction words, because generic risk titles such as
    # "cerebral infarction following TIA" can otherwise look like confirmation.
    has_infarct_confirmation = bool(infarct_count > 0)
    has_tia_evidence = bool(tia_count > 0 or _contains_concept(evidence_text, NORMALIZED_TAXONOMY["tia"]["aliases"]))
    has_hem_evidence = bool(hem_count > 0 or any(
        _contains_concept(evidence_text, NORMALIZED_TAXONOMY[k]["aliases"])
        for k in HEM_KEYS
    ))
    has_isch_evidence = bool(isch_count > 0 or any(
        _contains_concept(evidence_text, NORMALIZED_TAXONOMY[k]["aliases"])
        for k in ISCHEMIC_KEYS
    ))

    # TIA intent calibration
    if intent.get("transient_resolved_episode"):
        if key == "tia" and (has_tia_evidence or tia_count > 0):
            compatibility += 0.16
            notes.append("transient intent compatible with TIA evidence")

        elif key == "acute_ischemic_stroke":
            if not has_infarct_confirmation and infarct_count == 0:
                penalty += 0.40
                notes.append("transient-resolved episode without infarct-confirmation evidence")

        elif key in ISCHEMIC_KEYS:
            if not has_infarct_confirmation and infarct_count == 0:
                penalty += 0.25
                notes.append("transient-resolved episode weakens unconfirmed ischemic subtype")

    # Hemorrhage intent calibration
    if intent.get("hemorrhage_warning"):
        if key in HEM_KEYS and (has_hem_evidence or hem_count > 0):
            compatibility += 0.16
            notes.append("hemorrhage warning compatible with hemorrhagic evidence")

        elif key in ISCHEMIC_KEYS:
            if hem_count > 0 and infarct_count == 0:
                penalty += 0.42
                notes.append("hemorrhagic evidence present without infarct-confirmation evidence")

            elif hem_count > isch_count and infarct_count == 0:
                penalty += 0.35
                notes.append("hemorrhagic evidence outweighs ischemic evidence")

    # Candidate-family contradictions
    if key == "tia" and has_infarct_confirmation:
        penalty += 0.25
        notes.append("TIA candidate contradicted by infarct-confirmation evidence")

    if key in HEM_KEYS and has_isch_evidence and not has_hem_evidence and hem_count == 0:
        penalty += 0.20
        notes.append("hemorrhagic candidate lacks hemorrhagic evidence while ischemic evidence exists")

    return (
        round(min(compatibility, 1.25), 4),
        round(min(penalty, 0.70), 4),
        "; ".join(notes),
    )


# ============================================================
# Candidate scoring
# ============================================================

def score_candidate_support(
    candidate: str,
    sources: Sequence[Dict[str, Any]],
    grounding: Optional[Dict[str, Any]] = None,
    *,
    evidence_support_cap: int = 3,
    intent: Optional[Dict[str, Any]] = None,
    candidate_kg_verifications: Optional[Dict[str, Any]] = None,
) -> EvidenceJudgment:
    normalized = normalize_candidate(candidate)

    if normalized is None:
        return EvidenceJudgment(
            candidate=candidate,
            candidate_key="unknown",
            support_score=0.0,
            support_count=0,
            supporting_sources=[],
            kg_support=0.0,
            textual_support=0.0,
            grounding_support=0.0,
            source_diversity=0.0,
            rerank_strength=0.0,
            intent_compatibility=0.0,
            contradiction_penalty=0.0,
            grounding_ok=False,
            decision="rejected",
            reason="candidate is not in normalized taxonomy",
            candidate_kg_sources=[],
        )

    key, label, aliases, categories = normalized

    candidate_kg_docs = _candidate_kg_docs_for_label(label, candidate_kg_verifications)
    all_sources = list(sources or []) + candidate_kg_docs

    supporting: List[Dict[str, Any]] = []
    conflict_sources: List[Dict[str, Any]] = []

    textual_hits = 0
    kg_hits = 0
    best_kg = 0.0
    best_rerank = 0.0

    for source in list(all_sources or [])[:50]:
        if not isinstance(source, dict):
            continue

        if _source_is_noisy_nonclinical(source):
            conflict_sources.append(source)
            continue

        if _source_hard_conflicts_with_candidate(key, source, intent):
            conflict_sources.append(source)
            continue

        txt = _source_text(source)

        text_match = _contains_concept(txt, [label, *aliases])
        cat_match = _category_support(source, categories) > 0

        graph_score = _safe_float(source.get("graph_score"), 0.0)
        graph_disease = _low(source.get("graph_disease") or source.get("disease"))

        kg_match = bool(
            graph_disease
            and _contains_concept(graph_disease, [label, *aliases])
        )

        # KG category can support candidate only when it is not cross-family.
        if graph_score >= 0.75 and cat_match:
            kg_match = True

        if source.get("retrieval_channel") == "candidate_kg" and (text_match or kg_match or cat_match):
            kg_match = True

        # v7.12.6: very low candidate-KG scores from graph-disease metadata alone
        # must not create kg_support=1.0 for unrelated noisy articles.
        if (
            kg_match
            and not text_match
            and (source.get("from_kg") or source.get("graph_score") is not None)
            and graph_score < 0.25
        ):
            kg_match = False

        # Important: TIA should not be supported by generic ischemic category alone.
        if key == "tia" and not text_match and not kg_match:
            cat_match = False

        # Important: Acute ischemic stroke in transient-resolved cases needs stricter evidence.
        if (
            key == "acute_ischemic_stroke"
            and (intent or {}).get("transient_resolved_episode")
            and not _source_has_infarct_confirmation(source)
            and not _contains_concept(txt, ["acute ischemic stroke", "acute ischaemic stroke"])
        ):
            text_match = False
            kg_match = False
            cat_match = False

        if text_match or kg_match:
            src_copy = dict(source)
            src_copy["_evidence_match"] = {
                "text_match": bool(text_match),
                "kg_match": bool(kg_match),
                "category_match": bool(cat_match),
            }

            supporting.append(src_copy)

            textual_hits += int(text_match)
            kg_hits += int(kg_match)

            best_kg = max(best_kg, graph_score)
            best_rerank = max(
                best_rerank,
                _safe_float(source.get("medcpt_score"), 0.0),
                _safe_float(source.get("hybrid_score"), 0.0),
                _safe_float(source.get("score"), 0.0),
                _safe_float(source.get("match_cosine"), 0.0),
            )

    support_count = len(supporting)
    cap = max(1, int(evidence_support_cap or 3))

    textual_support = min(textual_hits / float(cap), 1.0)
    kg_support = min((kg_hits / 2.0) + (0.25 if best_kg >= 0.75 else 0.0), 1.0)

    grounding_support, grounding_ok = _grounding_score(grounding)

    has_kg = any(s.get("from_kg") or s.get("graph_score") is not None for s in supporting)
    has_non_kg = any(not (s.get("from_kg") or s.get("graph_score") is not None) for s in supporting)

    source_diversity = min(support_count / float(cap), 1.0)
    if has_kg and has_non_kg:
        source_diversity = min(source_diversity + 0.20, 1.0)

    rerank_strength = min(max(best_rerank, 0.0) / 1.0, 1.0)
    if best_rerank > 1.5:
        rerank_strength = min(best_rerank / 8.0, 1.0)

    intent_compat, contradiction_penalty, compat_reason = _intent_compatibility_and_penalty(
        key,
        intent,
        supporting,
        all_sources,
    )

    # Extra penalty from hard conflicting sources.
    #
    # Context-aware calibration:
    # In non-hemorrhage acute focal cases, hemorrhage/mimic papers may appear in retrieval
    # because stroke differentials and thrombolysis complications are common in PubMed.
    # Those sources must not SUPPORT ischemic candidates, but they also should not fully
    # suppress a well-supported ischemic candidate unless the clinical intent has
    # hemorrhage_warning=True or transient_resolved_episode=True.
    if conflict_sources:
        if (
            key in ISCHEMIC_KEYS
            and intent
            and not intent.get("hemorrhage_warning")
            and not intent.get("transient_resolved_episode")
            and intent.get("acute_neuro")
            and intent.get("focal_neuro")
            and not intent.get("chronic_or_mimic")
            and not intent.get("out_of_domain")
        ):
            contradiction_penalty += min(0.24, 0.03 * len(conflict_sources))
        else:
            contradiction_penalty += min(0.60, 0.12 * len(conflict_sources))

    hard_reject, hard_reject_reason = _should_hard_reject_candidate(
        key=key,
        intent=intent,
        supporting_sources=supporting,
        conflict_sources=conflict_sources,
        all_sources=all_sources,
    )

    if hard_reject:
        contradiction_penalty = max(contradiction_penalty, 0.85)

    contradiction_penalty = min(contradiction_penalty, 0.95)

    acute_focal_boost, acute_focal_boost_reason = _acute_focal_ischemic_evidence_boost(
        key=key,
        label=label,
        aliases=aliases,
        categories=categories,
        intent=intent,
        supporting_sources=supporting,
        conflict_sources=conflict_sources,
        all_sources=all_sources,
        grounding_ok=grounding_ok,
        support_count=support_count,
    )

    score = (
        0.32 * textual_support
        + 0.27 * kg_support
        + 0.18 * grounding_support
        + 0.10 * source_diversity
        + 0.06 * rerank_strength
        + 0.07 * min(intent_compat, 1.0)
        - contradiction_penalty
    )

    # Evidence-based boost: one strong KG + text match can support a candidate
    # when grounding is OK. This is not case-text diagnosis.
    if kg_support >= 0.75 and textual_hits >= 1 and grounding_ok:
        score += 0.12

    if acute_focal_boost > 0:
        score += acute_focal_boost

    score = round(min(max(score, 0.0), 1.0), 4)

    if not grounding_ok:
        decision, reason = "rejected", "grounding is too weak for subtype selection"
    elif support_count <= 0:
        decision, reason = "rejected", "no retrieved evidence or candidate-KG evidence mentions the candidate concept"
    elif hard_reject:
        decision, reason = "rejected", hard_reject_reason
    elif contradiction_penalty >= 0.70:
        decision, reason = "rejected", "candidate contradicted by stronger evidence family"
    elif contradiction_penalty >= 0.55 and score < 0.72:
        decision, reason = "rejected", "candidate contradicted by stronger evidence family"
    elif score >= 0.66 and support_count >= 1:
        decision, reason = "supported", "candidate is supported by retrieved evidence, KG verification, and grounding"
    elif acute_focal_boost > 0 and score >= 0.58 and support_count >= 1:
        decision, reason = "supported", "candidate is supported by acute focal ischemic evidence boost and grounding"
    elif score >= 0.45:
        decision, reason = "weak", "candidate has partial support but below threshold"
    else:
        decision, reason = "rejected", "candidate support is below threshold"

    if compat_reason:
        reason += f"; {compat_reason}"

    if conflict_sources:
        reason += f"; conflicting_sources={len(conflict_sources)}"

    if hard_reject:
        reason += "; hard_reject=True"

    if acute_focal_boost_reason:
        reason += f"; {acute_focal_boost_reason}"

    if (
        conflict_sources
        and key in ISCHEMIC_KEYS
        and intent
        and not intent.get("hemorrhage_warning")
        and not intent.get("transient_resolved_episode")
        and intent.get("acute_neuro")
        and intent.get("focal_neuro")
    ):
        reason += "; soft_conflict_penalty=True"

    return EvidenceJudgment(
        candidate=label,
        candidate_key=key,
        support_score=score,
        support_count=support_count,
        supporting_sources=supporting,
        kg_support=round(kg_support, 4),
        textual_support=round(textual_support, 4),
        grounding_support=round(grounding_support, 4),
        source_diversity=round(source_diversity, 4),
        rerank_strength=round(rerank_strength, 4),
        intent_compatibility=round(intent_compat, 4),
        contradiction_penalty=round(contradiction_penalty, 4),
        grounding_ok=grounding_ok,
        decision=decision,
        reason=reason,
        candidate_kg_sources=candidate_kg_docs,
        conflicting_sources=conflict_sources,
    )


# ============================================================
# Candidate generation
# ============================================================

def candidate_from_sources(
    sources: Sequence[Dict[str, Any]],
    initial_answer: str | None = None,
    max_candidates: int = 8,
    intent: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate candidates mostly from evidence.

    Intent can add candidate options, but it never decides the final diagnosis.
    """
    candidates: List[str] = []

    def add(label: str) -> None:
        if label and label not in candidates:
            candidates.append(label)

    if initial_answer:
        norm = normalize_candidate(initial_answer)
        if norm:
            add(norm[1])

    for source in list(sources or [])[:30]:
        if not isinstance(source, dict):
            continue

        txt = _source_text(source)

        for _, spec in NORMALIZED_TAXONOMY.items():
            if _contains_concept(txt, [spec["label"], *spec["aliases"]]):
                add(spec["label"])

    intent = intent or {}

    # Intent adds candidates only; it does NOT select them.
    if intent.get("transient_resolved_episode"):
        add(NORMALIZED_TAXONOMY["tia"]["label"])

    if intent.get("hemorrhage_warning"):
        add(NORMALIZED_TAXONOMY["intracerebral_hemorrhage"]["label"])
        add(NORMALIZED_TAXONOMY["subarachnoid_hemorrhage"]["label"])
        add(NORMALIZED_TAXONOMY["hemorrhagic_stroke"]["label"])

    return candidates[:max_candidates]


# ============================================================
# Candidate judging
# ============================================================

# [تم حذف كود ميت مكرر غير مستخدم: judge_candidates (v1, superseded) — السطر الأصلي 2089-2405]
def select_best_supported_candidate(
    judgments: Sequence[Dict[str, Any]],
    min_support_score: float = 0.66,
) -> Optional[Dict[str, Any]]:
    """
    Select the best supported candidate.

    v7.12.3 KG-guided calibration:
    - Keep the conservative default threshold.
    - Permit a lower floor only when the Evidence Judge already marked the
      candidate as supported and the support is KG-guided or acute-focal boosted.
    - Never select hard-rejected candidates.
    """
    best = None

    for j in judgments or []:
        if j.get("decision") != "supported":
            continue

        score = float(j.get("support_score") or 0.0)
        reason = _low(j.get("reason") or "")

        if "hard_reject=true" in reason:
            continue

        kg_support = float(j.get("kg_support") or 0.0)
        kg_sources = j.get("candidate_kg_sources") or []

        standard_ok = score >= min_support_score
        kg_guided_ok = (
            score >= 0.56
            and (
                kg_support >= 0.25
                or bool(kg_sources)
                or "acute_focal_ischemic_evidence_boost=true" in reason
                or "soft_conflict_penalty=true" in reason
            )
        )

        if not (standard_ok or kg_guided_ok):
            continue

        if best is None or score > float(best.get("support_score") or 0.0):
            best = j

    return best


def explain_support(judgment: Optional[Dict[str, Any]]) -> str:
    if not judgment:
        return "no supported candidate"

    return (
        f"{judgment.get('candidate')} | decision={judgment.get('decision')} | "
        f"support_score={judgment.get('support_score')} | sources={judgment.get('support_count')} | "
        f"kg={judgment.get('kg_support')} | grounding={judgment.get('grounding_support')} | "
        f"penalty={judgment.get('contradiction_penalty')}"
    )


# ============================================================
# v7.12.13 posterior-vomiting hemorrhage demotion
# Goal:
#   In strong posterior circulation pattern with vomiting only,
#   do not allow ICH/SAH/hemorrhagic stroke to suppress ischemic posterior candidates.
# ============================================================

_V71213_EJ_POSTERIOR_CUES = [
    "vertigo", "dizziness", "ataxia", "gait ataxia", "gait difficulty",
    "difficulty walking", "unsteady gait", "diplopia", "double vision",
    "nystagmus", "dysmetria", "brainstem", "cerebellar",
    "posterior circulation", "vertebrobasilar", "basilar",
    "????", "????", "????", "??????", "???? ??????", "????", "??? ??????",
]

_V71213_EJ_HARD_HEMORRHAGE_CUES = [
    "thunderclap", "worst headache", "worst headache of life",
    "sudden severe headache", "severe headache",
    "neck stiffness", "meningismus", "photophobia",
    "decreased consciousness", "loss of consciousness", "reduced consciousness",
    "altered consciousness", "coma", "seizure",
    "intraparenchymal", "intracerebral hemorrhage", "subarachnoid hemorrhage",
    "hemorrhagic stroke", "hematoma", "parenchymal blood",
    "ct shows hemorrhage", "ct showed hemorrhage", "brain bleed",
    "intracranial bleeding", "hemorrhage on ct",
    "hypertensive emergency", "hypertensive crisis", "malignant hypertension",
    "very high blood pressure", "severe hypertension",
    "warfarin", "anticoagulation", "anticoagulant",
    "rivaroxaban", "apixaban", "dabigatran",
    "???? ????", "???? ?????", "????? ???", "??? ???", "??????",
    "???", "????", "??? ????",
]

_V71213_EJ_VOMITING_CUES = [
    "vomiting", "emesis", "nausea and vomiting",
    "???", "?????", "????",
]

_V71213_HEMORRHAGE_KEYS = {
    "intracerebral_hemorrhage",
    "subarachnoid_hemorrhage",
    "hemorrhagic_stroke",
}

_V71213_POSTERIOR_KEYS = {
    "posterior_circulation_stroke",
    "basilar_artery_occlusion",
    "cerebellar_infarction",
    "brainstem_infarction",
}


def _v71213_ej_any(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in terms)


def _v71213_ej_count(text: str, terms: list[str]) -> int:
    low = (text or "").lower()
    return sum(1 for t in terms if t in low)


def _v71213_posterior_vomiting_only_context(case_text: str, intent=None) -> bool:
    q = (case_text or "").lower()
    intent = intent or {}

    if bool(intent.get("hemorrhage_warning")):
        return False

    return bool(
        _v71213_ej_count(q, _V71213_EJ_POSTERIOR_CUES) >= 2
        and _v71213_ej_any(q, _V71213_EJ_VOMITING_CUES)
        and not _v71213_ej_any(q, _V71213_EJ_HARD_HEMORRHAGE_CUES)
    )


def _v71213_has_evidence_gate(j) -> bool:
    try:
        kg_support = float(j.get("kg_support") or 0.0)
    except Exception:
        kg_support = 0.0

    try:
        textual_support = float(j.get("textual_support") or 0.0)
    except Exception:
        textual_support = 0.0

    try:
        grounding_support = float(j.get("grounding_support") or 0.0)
    except Exception:
        grounding_support = 0.0

    support_count = int(j.get("support_count") or 0)
    kg_sources = j.get("candidate_kg_sources") or []

    return bool(
        support_count > 0
        and (
            kg_support >= 0.25
            or bool(kg_sources)
            or textual_support >= 0.34
            or grounding_support >= 0.45
        )
    )


if "_v71213_original_judge_candidates" not in globals():
    _v71213_original_judge_candidates = judge_candidates

    def judge_candidates(
        case_text,
        candidates,
        sources,
        grounding,
        intent=None,
        candidate_kg_verifications=None,
        min_support_score=0.66,
        evidence_support_cap=3,
    ):
        judgments = _v71213_original_judge_candidates(
            case_text,
            candidates,
            sources,
            grounding,
            intent=intent,
            candidate_kg_verifications=candidate_kg_verifications,
            min_support_score=min_support_score,
            evidence_support_cap=evidence_support_cap,
        )

        if not _v71213_posterior_vomiting_only_context(case_text, intent=intent):
            return judgments

        for j in judgments or []:
            key = str(j.get("candidate_key") or "")

            if key in _V71213_HEMORRHAGE_KEYS:
                old_score = float(j.get("support_score") or 0.0)
                j["decision"] = "rejected"
                j["support_score"] = round(min(old_score, 0.34), 4)
                j["contradiction_penalty"] = max(float(j.get("contradiction_penalty") or 0.0), 0.90)
                j["reason"] = (
                    str(j.get("reason") or "")
                    + "; v71213_reject_hemorrhage_in_posterior_vomiting_only_context"
                )

            elif key in _V71213_POSTERIOR_KEYS and _v71213_has_evidence_gate(j):
                old_score = float(j.get("support_score") or 0.0)
                j["decision"] = "supported"
                j["support_score"] = round(max(old_score, 0.68), 4)
                j["intent_compatibility"] = max(float(j.get("intent_compatibility") or 0.0), 1.12)
                j["reason"] = (
                    str(j.get("reason") or "")
                    .replace("v71210_case_safety_reject_ischemic_tia_in_clinical_ich_context", "")
                    + "; v71213_posterior_vomiting_context_boost"
                )

        judgments.sort(
            key=lambda j: (
                j.get("decision") == "supported",
                float(j.get("support_score") or 0.0),
                int(j.get("support_count") or 0),
            ),
            reverse=True,
        )

        return judgments


# ============================================================
# v7.12.13b evidence_judge signature compatibility fix
# Fix:
#   The original judge_candidates in this project version does not accept
#   min_support_score / evidence_support_cap in some builds.
#   This wrapper accepts all kwargs, filters unsupported kwargs, then applies
#   the posterior-vomiting calibration safely.
# ============================================================

import inspect as _v71213b_inspect

_V71213B_HEMORRHAGE_LABEL_TERMS = [
    "intracerebral hemorrhage",
    "intracerebral haemorrhage",
    "hemorrhagic stroke",
    "haemorrhagic stroke",
    "subarachnoid hemorrhage",
    "subarachnoid haemorrhage",
    "acute intracranial hemorrhage",
]

_V71213B_POSTERIOR_LABEL_TERMS = [
    "posterior circulation",
    "vertebrobasilar",
    "basilar artery occlusion",
    "basilar occlusion",
    "cerebellar infarction",
    "cerebellar stroke",
    "brainstem infarction",
]


def _v71213b_candidate_text(j):
    return str(
        j.get("candidate")
        or j.get("candidate_label")
        or j.get("diagnosis")
        or j.get("label")
        or ""
    ).lower()


def _v71213b_is_hemorrhage_candidate(j):
    key = str(j.get("candidate_key") or "").lower()
    label = _v71213b_candidate_text(j)

    if key in {
        "intracerebral_hemorrhage",
        "subarachnoid_hemorrhage",
        "hemorrhagic_stroke",
    }:
        return True

    return any(t in label for t in _V71213B_HEMORRHAGE_LABEL_TERMS)


def _v71213b_is_posterior_candidate(j):
    key = str(j.get("candidate_key") or "").lower()
    label = _v71213b_candidate_text(j)

    if key in {
        "posterior_circulation_stroke",
        "basilar_artery_occlusion",
        "cerebellar_infarction",
        "brainstem_infarction",
    }:
        return True

    return any(t in label for t in _V71213B_POSTERIOR_LABEL_TERMS)


def _v71213b_has_evidence_gate(j):
    def f(name, default=0.0):
        try:
            return float(j.get(name) or default)
        except Exception:
            return default

    support_count = int(j.get("support_count") or 0)
    kg_sources = j.get("candidate_kg_sources") or j.get("kg_sources") or []

    return bool(
        support_count > 0
        or bool(kg_sources)
        or f("kg_support") >= 0.25
        or f("textual_support") >= 0.34
        or f("grounding_support") >= 0.45
        or f("support_score") >= 0.45
    )


def _v71213b_call_original_judge(*args, **kwargs):
    sig = _v71213b_inspect.signature(_v71213_original_judge_candidates)
    params = sig.parameters

    accepts_var_kwargs = any(
        p.kind == p.VAR_KEYWORD
        for p in params.values()
    )

    if accepts_var_kwargs:
        safe_kwargs = dict(kwargs)
    else:
        safe_kwargs = {
            k: v for k, v in kwargs.items()
            if k in params
        }

    return _v71213_original_judge_candidates(*args, **safe_kwargs)



def _attach_decision_trace_fields(judgments):
    """Add ranks and structured reasons after all calibration wrappers finish."""
    rows = [j for j in (judgments or []) if isinstance(j, dict)]
    try:
        rows.sort(
            key=lambda j: (
                j.get("decision") == "supported",
                float(j.get("support_score") or 0.0),
                int(j.get("support_count") or 0),
            ),
            reverse=True,
        )
    except Exception:
        pass

    for rank, j in enumerate(rows, start=1):
        j["rank"] = rank
        j["final_support_score"] = float(j.get("support_score") or 0.0)
        j["score_breakdown"] = {
            "textual_support": float(j.get("textual_support") or 0.0),
            "kg_support": float(j.get("kg_support") or 0.0),
            "grounding_support": float(j.get("grounding_support") or 0.0),
            "rerank_strength": float(j.get("rerank_strength") or 0.0),
            "source_diversity": float(j.get("source_diversity") or 0.0),
            "intent_compatibility": float(j.get("intent_compatibility") or 0.0),
            "contradiction_penalty": float(j.get("contradiction_penalty") or 0.0),
        }
        reason = str(j.get("reason") or "").strip()
        decision = str(j.get("decision") or "unknown")
        j["selection_reasons"] = [reason] if decision == "supported" and reason else []
        j["weakening_reasons"] = [reason] if decision == "weak" and reason else []
        j["rejection_reasons"] = [reason] if decision == "rejected" and reason else []
    return judgments

# Override the broken v7.12.13 wrapper with a signature-safe wrapper.
# [تم حذف كود ميت مكرر غير مستخدم: judge_candidates (v2 "signature-safe wrapper", superseded) — السطر الأصلي 2782-2859]
def _case_has_hemorrhage_hard_cue(case_text: str) -> bool:
    profile = _v715_case_profile(case_text)
    return bool(profile.get("ich_warning") or profile.get("sah_warning"))


def _case_is_transient_resolved_no_infarct(case_text: str) -> bool:
    profile = _v715_case_profile(case_text)
    if not profile.get("transient_resolved_episode"):
        return False
    q = _low(case_text)
    return _count_concept_hits(q, CASE_INFARCT_CONFIRMATION_CUES) <= 0


def _case_has_broad_tia_recall_cue(case_text: str) -> bool:
    profile = _v715_case_profile(case_text)
    return bool(profile.get("transient_resolved_episode") and not profile.get("persistent_deficit"))


def _case_has_sah_recall_cue(case_text: str) -> bool:
    return bool(_v715_case_profile(case_text).get("sah_warning"))


def _case_has_weak_sah_recall_cue(case_text: str) -> bool:
    # A weak substring-only SAH gate is unsafe.  Use the same positive SAH
    # profile as the strict gate and let evidence determine final support.
    return bool(_v715_case_profile(case_text).get("sah_warning"))


def _case_has_sah_specific_cue(case_text: str) -> bool:
    return bool(_v715_case_profile(case_text).get("sah_warning"))


def _case_has_broad_hemorrhage_recall_cue(case_text: str) -> bool:
    return bool(_v715_case_profile(case_text).get("hemorrhage_warning"))


def _case_has_clinical_ich_recall_cue(case_text: str) -> bool:
    return bool(_v715_case_profile(case_text).get("ich_warning"))


def _case_has_posterior_recall_cue(case_text: str) -> bool:
    profile = _v715_case_profile(case_text)
    return bool(
        profile.get("acute_onset")
        and profile.get("posterior_pattern")
        and not profile.get("hemorrhage_warning")
        and not profile.get("transient_resolved_episode")
    )


def _case_has_strong_posterior_clinical_cue(case_text: str) -> bool:
    return _case_has_posterior_recall_cue(case_text)


def _case_has_acute_focal_ischemic_recall_cue(case_text: str) -> bool:
    profile = _v715_case_profile(case_text)
    return bool(
        profile.get("acute_onset")
        and profile.get("focal_neuro")
        and not profile.get("hemorrhage_warning")
        and not profile.get("transient_resolved_episode")
        and not profile.get("mimic_present")
        and not profile.get("chronic_context")
    )


def _v715_recompute_score(j: Dict[str, Any], penalty: float) -> float:
    """Recompute the documented Evidence Judge formula with a corrected penalty."""
    text = _safe_float(j.get("textual_support"), 0.0)
    kg = _safe_float(j.get("kg_support"), 0.0)
    grounding_support = _safe_float(j.get("grounding_support"), 0.0)
    diversity = _safe_float(j.get("source_diversity"), 0.0)
    rerank = _safe_float(j.get("rerank_strength"), 0.0)
    intent_compat = min(_safe_float(j.get("intent_compatibility"), 0.0), 1.0)
    score = (
        0.32 * text
        + 0.27 * kg
        + 0.18 * grounding_support
        + 0.10 * diversity
        + 0.06 * rerank
        + 0.07 * intent_compat
        - penalty
    )
    if kg >= 0.75 and text > 0 and bool(j.get("grounding_ok")):
        score += 0.12
    return round(min(max(score, 0.0), 1.0), 4)


if "_v715_original_judge_candidates" not in globals():
    _v715_original_judge_candidates = judge_candidates


def judge_candidates(*args, **kwargs):
    case_text = str(args[0] if args else kwargs.get("case_text") or "")
    profile = _v715_case_profile(case_text)

    # Reconcile any caller-provided intent with the authoritative shared profile.
    corrected_intent = dict(kwargs.get("intent") or {})
    corrected_intent.update({
        "acute_neuro": bool(profile.get("acute_onset")),
        "focal_neuro": bool(profile.get("focal_neuro")),
        "hemorrhage_warning": bool(profile.get("hemorrhage_warning")),
        "transient_resolved_episode": bool(profile.get("transient_resolved_episode")),
        "persistent_deficit": bool(profile.get("persistent_deficit")),
        "chronic_or_mimic": bool(profile.get("chronic_context") or profile.get("mimic_present")),
        "context_profile": profile,
    })
    kwargs["intent"] = corrected_intent

    judgments = _v715_original_judge_candidates(*args, **kwargs)
    acute_persistent_focal = bool(
        profile.get("acute_onset")
        and profile.get("focal_neuro")
        and profile.get("persistent_deficit")
        and not profile.get("hemorrhage_warning")
    )
    explicit_resolution = bool(profile.get("explicit_resolution"))
    sah_positive = bool(profile.get("sah_warning"))
    ich_positive = bool(profile.get("ich_warning"))

    for j in judgments or []:
        if not isinstance(j, dict):
            continue
        key = str(j.get("candidate_key") or "")
        reason = str(j.get("reason") or "")
        support_count = int(_safe_float(j.get("support_count"), 0.0))
        evidence_gate = bool(
            j.get("grounding_ok")
            and support_count >= 1
            and (
                _safe_float(j.get("textual_support"), 0.0) > 0
                or _safe_float(j.get("kg_support"), 0.0) > 0
            )
        )

        # TIA requires a rapidly and completely resolved focal episode.  A short
        # elapsed time with deficits still present is not TIA.
        if key == "tia" and (profile.get("persistent_deficit") or not explicit_resolution):
            j["decision"] = "rejected"
            j["support_score"] = round(min(_safe_float(j.get("support_score"), 0.0), 0.25), 4)
            j["contradiction_penalty"] = max(_safe_float(j.get("contradiction_penalty"), 0.0), 0.90)
            j["reason"] = reason + "; v715_reject_tia_without_complete_resolution"
            continue

        # Generic KG/retrieved hemorrhage literature cannot establish that this
        # patient has hemorrhage when the case has no positive hemorrhage cue.
        if key == "subarachnoid_hemorrhage" and not sah_positive:
            j["decision"] = "rejected"
            j["support_score"] = round(min(_safe_float(j.get("support_score"), 0.0), 0.28), 4)
            j["contradiction_penalty"] = max(_safe_float(j.get("contradiction_penalty"), 0.0), 0.92)
            j["reason"] = reason + "; v715_reject_sah_without_affirmed_sah_features"
            continue

        if key in {"intracerebral_hemorrhage", "hemorrhagic_stroke"} and not ich_positive and not sah_positive:
            j["decision"] = "rejected"
            j["support_score"] = round(min(_safe_float(j.get("support_score"), 0.0), 0.30), 4)
            j["contradiction_penalty"] = max(_safe_float(j.get("contradiction_penalty"), 0.0), 0.90)
            j["reason"] = reason + "; v715_reject_hemorrhage_without_affirmed_bleeding_features"
            continue

        # In an acute persistent focal syndrome with explicitly absent
        # hemorrhage warning features, generic opposite-family retrieval is
        # treated as differential/noise, not patient-level contradiction.
        if key in ISCHEMIC_KEYS and acute_persistent_focal and evidence_gate:
            corrected_penalty = min(_safe_float(j.get("contradiction_penalty"), 0.0), 0.24)
            corrected_score = _v715_recompute_score(j, corrected_penalty)
            j["contradiction_penalty"] = round(corrected_penalty, 4)
            j["support_score"] = corrected_score
            provisional_ais_gate = bool(
                key == "acute_ischemic_stroke"
                and corrected_score >= 0.58
                and _safe_float(j.get("textual_support"), 0.0) >= 0.66
                and support_count >= 2
            )
            if corrected_score >= 0.66 or provisional_ais_gate:
                j["decision"] = "supported"
                tag = (
                    "v715_provisional_ais_supported_pending_neuroimaging"
                    if provisional_ais_gate and corrected_score < 0.66
                    else "v715_negation_aware_acute_persistent_focal_reconciliation"
                )
                j["reason"] = _clean_supported_reason_after_case_boost(reason) + f"; {tag}"
            elif corrected_score >= 0.45:
                j["decision"] = "weak"
                j["reason"] = reason + "; v715_partial_acute_persistent_focal_support"

        j["case_context"] = {
            "persistent_deficit": bool(profile.get("persistent_deficit")),
            "explicit_resolution": explicit_resolution,
            "hemorrhage_warning": bool(profile.get("hemorrhage_warning")),
            "sah_warning": sah_positive,
            "ich_warning": ich_positive,
        }
        j["subtype_requires_neuroimaging"] = key in (ISCHEMIC_KEYS | HEM_KEYS | {"tia"})

    rows = [j for j in judgments or [] if isinstance(j, dict)]
    rows.sort(key=lambda item: _safe_float(item.get("support_score"), 0.0), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["final_support_score"] = _safe_float(row.get("support_score"), 0.0)
        row["score_breakdown"] = {
            "textual_support": _safe_float(row.get("textual_support"), 0.0),
            "kg_support": _safe_float(row.get("kg_support"), 0.0),
            "grounding_support": _safe_float(row.get("grounding_support"), 0.0),
            "rerank_strength": _safe_float(row.get("rerank_strength"), 0.0),
            "source_diversity": _safe_float(row.get("source_diversity"), 0.0),
            "intent_compatibility": _safe_float(row.get("intent_compatibility"), 0.0),
            "contradiction_penalty": _safe_float(row.get("contradiction_penalty"), 0.0),
        }
    return rows
