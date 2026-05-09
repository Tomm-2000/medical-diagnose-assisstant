# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .build_graph import MedicalKnowledgeGraph


class GraphRetriever:
    def __init__(self, uri="bolt://127.0.0.1:7687", user="neo4j", password="password"):
        self.kg = MedicalKnowledgeGraph(
    uri=uri,
    user=user,
    password=password
)

        self.negation_terms = {
            "no", "not", "without", "denies", "denied", "negative", "negative for",
            "free of", "absence of", "absent"
        }

        self.symptom_synonyms = {
            "weakness": "weakness",
            "unilateral weakness": "unilateral weakness",
            "arm weakness": "arm weakness",
            "leg weakness": "leg weakness",
            "left-sided weakness": "unilateral weakness",
            "right-sided weakness": "unilateral weakness",
            "hemiparesis": "hemiparesis",
            "hemiplegia": "hemiplegia",
            "aphasia": "aphasia",
            "dysarthria": "dysarthria",
            "slurred speech": "dysarthria",
            "speech difficulty": "speech disturbance",
            "speech disturbance": "speech disturbance",
            "facial droop": "facial droop",
            "face droop": "facial droop",
            "numbness": "numbness",
            "paresthesia": "numbness",
            "vertigo": "vertigo",
            "dizziness": "dizziness",
            "diplopia": "diplopia",
            "double vision": "diplopia",
            "ataxia": "ataxia",
            "gait difficulty": "gait difficulty",
            "difficulty walking": "gait difficulty",
            "headache": "headache",
            "severe headache": "headache",
            "sudden severe headache": "headache",
            "thunderclap headache": "headache",
            "vomiting": "vomiting",
            "loss of consciousness": "loss of consciousness",
            "decreased consciousness": "loss of consciousness",
            "worsening consciousness": "loss of consciousness",
            "confusion": "confusion",
        }

        self._ordered_symptoms = sorted(
            self.symptom_synonyms.keys(),
            key=len,
            reverse=True
        )

        self.stroke_priority_terms = [
            "stroke",
            "ischemic stroke",
            "ischaemic stroke",
            "acute ischemic stroke",
            "acute ischaemic stroke",
            "posterior circulation stroke",
            "posterior circulation ischemic stroke",
            "brainstem stroke",
            "brainstem infarction",
            "midbrain infarction",
            "pontine infarction",
            "lateral medullary infarction",
            "medullary infarction",
            "cerebellar stroke",
            "cerebellar infarction",
            "vertebrobasilar disease",
            "vertebrobasilar insufficiency",
            "vertebrobasilar ischemia",
            "vertebrobasilar ischaemia",
            "transient ischemic attack",
            "transient ischaemic attack",
            "tia",
            "cerebral infarction",
            "brain infarction",
            "intracerebral hemorrhage",
            "intracerebral haemorrhage",
            "subarachnoid hemorrhage",
            "subarachnoid haemorrhage",
            "hemorrhagic stroke",
            "haemorrhagic stroke",
            "ischemia",
            "ischaemia",
            "infarction",
        ]

        self.general_vascular_terms = [
            "stroke",
            "acute stroke",
            "ischemic stroke",
            "ischaemic stroke",
            "ischemia",
            "ischaemia",
            "infarction",
            "cerebral infarction",
            "brain infarction",
            "transient ischemic attack",
            "transient ischaemic attack",
            "tia",
        ]

        self.posterior_specific_terms = [
            "posterior circulation",
            "vertebrobasilar",
            "brainstem",
            "midbrain",
            "pontine",
            "pons",
            "medullary",
            "lateral medullary",
            "cerebellar infarction",
            "cerebellar stroke",
            "basilar artery",
            "aica",
            "pica",
        ]

        self.nonacute_penalty_terms = [
            "ataxia telangiectasia",
            "atm",
            "spinocerebellar ataxia",
            "cerebellar ataxia",
            "friedreich",
            "meniere",
            "ménière",
            "bppv",
            "benign paroxysmal positional vertigo",
            "benign positioning vertigo",
            "migraine",
            "migrainous vertigo",
            "vestibular hypofunction",
            "vestibular rehabilitation",
            "schwannoma",
            "chronic dizziness",
            "postconcussion",
            "superficial siderosis",
            "pregnancy",
            "cancer",
            "genetic",
            "familial",
            "autosomal",
            "mutation",
            "syndrome",
        ]

        self.acute_query_terms = [
            "acute", "sudden", "suddenly", "new onset", "acute onset",
            "abrupt", "abruptly", "ongoing", "persistent", "persisting",
            "started", "developed", "began", "onset",
            "thunderclap", "worsening", "emergency",
            "today", "this morning", "this afternoon", "this evening",
            "last night", "same day"
        ]

        self.chronic_query_terms = [
            "chronic", "gradual", "gradually", "progressive", "progressively",
            "long-standing", "longstanding", "recurrent for months",
            "for weeks", "for months", "for years", "over weeks", "over months",
            "over years", "several weeks", "several months", "several years"
        ]

        self.posterior_query_terms = [
            "vertigo", "diplopia", "double vision", "ataxia", "gait difficulty",
            "difficulty walking", "dizziness", "nystagmus", "imbalance",
            "brainstem", "cerebellar", "posterior circulation"
        ]

        self.focal_query_terms = [
            "weakness", "unilateral weakness", "arm weakness", "leg weakness",
            "left-sided weakness", "right-sided weakness",
            "hemiparesis", "hemiplegia", "aphasia", "dysarthria",
            "slurred speech", "facial droop", "numbness"
        ]

        self.hemorrhage_query_terms = [
            "thunderclap", "worst headache", "sudden severe headache",
            "vomiting", "loss of consciousness", "decreased consciousness",
            "worsening consciousness", "subarachnoid", "intracerebral hemorrhage",
            "intracerebral haemorrhage", "hemorrhage", "haemorrhage", "sah", "ich"
        ]

        self.article_priority_terms = [
            "stroke", "posterior circulation", "brainstem", "cerebellar",
            "vertebrobasilar", "ischemia", "ischaemia", "infarction",
            "ischemic", "ischaemic", "transient ischemic attack",
            "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral",
            "acute", "emergency"
        ]

        self.article_penalty_terms = [
            "mutation", "autosomal", "familial", "syndrome", "genetic",
            "rehabilitation", "pregnancy", "cancer", "cell", "molecular",
            "mouse", "rat", "protein", "phosphorylation", "in vitro",
            "mrna", "messenger rna", "kinase", "animal model", "rat model", "mouse model"
        ]

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _canonicalize_symptom(self, symptom: str) -> str:
        s = self._normalize_text(symptom)
        return self.symptom_synonyms.get(s, s)

    def _is_negated(self, text: str, phrase: str, window: int = 5) -> bool:
        txt = self._normalize_text(text)
        phr = self._normalize_text(phrase)

        if not txt or not phr:
            return False

        pattern = re.compile(r"\b" + re.escape(phr) + r"\b")
        for m in pattern.finditer(txt):
            left = txt[:m.start()].strip()
            if not left:
                continue

            tokens = left.split()
            window_tokens = tokens[-window:]
            window_text = " ".join(window_tokens)

            for neg in self.negation_terms:
                if neg in window_text:
                    return True

        return False

    def extract_symptoms_from_query(self, query: str) -> List[str]:
        query_norm = self._normalize_text(query)
        found: List[str] = []

        for symptom in self._ordered_symptoms:
            if symptom in query_norm and not self._is_negated(query_norm, symptom):
                found.append(self._canonicalize_symptom(symptom))

        return list(dict.fromkeys(found))

    @staticmethod
    def _minmax_norm(values: List[float]) -> List[float]:
        if not values:
            return []
        mn, mx = min(values), max(values)
        if mx - mn < 1e-9:
            return [1.0 for _ in values]
        return [(v - mn) / (mx - mn) for v in values]

    def _count_term_hits(self, text: str, terms: List[str]) -> int:
        low = self._normalize_text(text)
        return sum(1 for t in terms if t in low)

    def _extract_time_context(self, query: str) -> Dict[str, Optional[float] | bool | str]:
        """
        Temporal context extraction.

        الهدف:
        - فهم الزمن بشكل عام، وليس فقط 30/60/90 دقيقة.
        - minutes/hours => acute غالباً.
        - days قد تكون acute إذا قليلة، لكن أضعف من minutes/hours.
        - weeks/months/years أو progressive/chronic => non-acute.
        """
        q = self._normalize_text(query)

        ctx = {
            "has_time": False,
            "duration_minutes": None,
            "is_acute_time": False,
            "is_chronic_time": False,
            "time_unit": None,
            "time_phrase": None,
        }

        # chronic/progressive language
        if any(t in q for t in self.chronic_query_terms):
            ctx["has_time"] = True
            ctx["is_chronic_time"] = True
            ctx["is_acute_time"] = False
            ctx["time_phrase"] = "chronic_or_progressive"
            return ctx

        # numeric time patterns: for 3 hours, 90 minutes, 2 hrs, 45 min, over 2 days
        time_patterns = [
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(day|days)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(week|weeks)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(month|months)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(year|years)\b",
        ]

        for pattern in time_patterns:
            m = re.search(pattern, q)
            if not m:
                continue

            value = float(m.group(1))
            unit = m.group(2)

            minutes = None
            if unit in {"minute", "minutes", "min", "mins"}:
                minutes = value
            elif unit in {"hour", "hours", "hr", "hrs"}:
                minutes = value * 60
            elif unit in {"day", "days"}:
                minutes = value * 24 * 60
            elif unit in {"week", "weeks"}:
                minutes = value * 7 * 24 * 60
            elif unit in {"month", "months"}:
                minutes = value * 30 * 24 * 60
            elif unit in {"year", "years"}:
                minutes = value * 365 * 24 * 60

            ctx["has_time"] = True
            ctx["duration_minutes"] = minutes
            ctx["time_unit"] = unit
            ctx["time_phrase"] = m.group(0).strip()

            if unit in {"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs"}:
                # أي دقائق أو ساعات تعتبر acute في triage context
                ctx["is_acute_time"] = True
                ctx["is_chronic_time"] = False
            elif unit in {"day", "days"}:
                # أيام قليلة قد تكون acute/subacute، ونعتبر <= 1 day acute strongly
                ctx["is_acute_time"] = bool(value <= 1)
                ctx["is_chronic_time"] = bool(value >= 14)
            else:
                ctx["is_acute_time"] = False
                ctx["is_chronic_time"] = True

            return ctx

        vague_acute_phrases = [
            "sudden onset", "acute onset", "new onset", "abrupt onset",
            "started today", "began today", "developed today",
            "this morning", "this afternoon", "this evening",
            "last night", "today", "same day",
            "just started", "suddenly developed", "suddenly started"
        ]

        if any(p in q for p in vague_acute_phrases):
            ctx["has_time"] = True
            ctx["is_acute_time"] = True
            ctx["is_chronic_time"] = False
            ctx["time_phrase"] = "vague_acute"
            return ctx

        vague_short_duration = [
            "several minutes", "few minutes", "several hours", "few hours",
            "couple of hours", "a few hours", "a few minutes"
        ]

        if any(p in q for p in vague_short_duration):
            ctx["has_time"] = True
            ctx["is_acute_time"] = True
            ctx["is_chronic_time"] = False
            ctx["time_phrase"] = "vague_short_duration"
            return ctx

        return ctx

    def _query_context_flags(self, query: str) -> Dict[str, bool]:
        q = self._normalize_text(query)

        time_ctx = self._extract_time_context(q)

        acute = any(t in q for t in self.acute_query_terms) or bool(time_ctx["is_acute_time"])
        posterior = any(t in q for t in self.posterior_query_terms)
        focal = any(t in q for t in self.focal_query_terms)
        hemorrhage = any(t in q for t in self.hemorrhage_query_terms)

        if bool(time_ctx["is_chronic_time"]):
            acute = False

        posterior_symptom_hits = self._count_term_hits(q, [
            "vertigo", "diplopia", "double vision", "ataxia",
            "gait difficulty", "difficulty walking", "nystagmus", "imbalance"
        ])

        # posterior pattern يحتاج أكثر من عرض خلفي + سياق حاد أو مفاجئ
        if posterior_symptom_hits >= 2 and (
            bool(time_ctx["is_acute_time"])
            or any(t in q for t in ["sudden", "acute", "ongoing", "persistent", "new onset"])
        ):
            acute = True
            posterior = True

        # hemorrhage red flags are acute unless explicitly chronic context exists
        if hemorrhage and not bool(time_ctx["is_chronic_time"]):
            acute = True

        return {
            "acute": bool(acute),
            "posterior": bool(posterior),
            "focal": bool(focal),
            "hemorrhage": bool(hemorrhage),
        }

    def _query_context_details(self, query: str) -> Dict:
        flags = self._query_context_flags(query)
        time_ctx = self._extract_time_context(query)
        return {
            **flags,
            "time_context": time_ctx,
        }

    def _match_query_symptoms_to_disease(self, disease_name: str, symptoms: List[str], query: str) -> List[str]:
        disease_low = self._normalize_text(disease_name)
        query_low = self._normalize_text(query)

        matched: List[str] = []
        for s in symptoms:
            if not s:
                continue

            if s in disease_low:
                matched.append(s)
                continue

            if s in {"vertigo", "dizziness", "ataxia", "gait difficulty", "diplopia"}:
                if any(t in disease_low for t in [
                    "posterior circulation", "brainstem", "cerebellar",
                    "vertebrobasilar", "medullary", "pontine", "midbrain",
                    "stroke", "infarction", "ischemia", "ischaemia"
                ]):
                    matched.append(s)
                    continue

            if s in {"weakness", "unilateral weakness", "arm weakness", "leg weakness", "hemiparesis", "hemiplegia"}:
                if any(t in disease_low for t in [
                    "stroke", "ischemia", "ischaemia", "infarction", "transient ischemic attack", "tia"
                ]):
                    matched.append(s)
                    continue

            if s in {"aphasia", "dysarthria", "speech disturbance", "facial droop", "numbness"}:
                if any(t in disease_low for t in [
                    "stroke", "ischemia", "ischaemia", "infarction", "transient ischemic attack", "tia"
                ]):
                    matched.append(s)
                    continue

            if s in {"headache", "vomiting", "loss of consciousness", "confusion"}:
                if any(t in disease_low for t in [
                    "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral", "stroke"
                ]):
                    matched.append(s)
                    continue

        flags = self._query_context_flags(query_low)
        if not matched:
            if flags["acute"] and self._count_term_hits(disease_low, self.stroke_priority_terms) > 0:
                matched = symptoms[: min(len(symptoms), 2)]

        return list(dict.fromkeys(matched))

    def _stroke_bias_score(
        self,
        disease_name: str,
        query: str,
        matched_symptoms: List[str],
        disease_row: Dict | None = None,
    ) -> float:
        disease_low = self._normalize_text(disease_name)
        flags = self._query_context_flags(query)
        disease_row = disease_row or {}

        score = 0.0

        stroke_hits = min(self._count_term_hits(disease_low, self.stroke_priority_terms), 3)
        penalty_hits = min(self._count_term_hits(disease_low, self.nonacute_penalty_terms), 3)

        score += 0.45 * stroke_hits
        score -= 0.35 * penalty_hits

        vascular_relevance = float(disease_row.get("vascular_relevance", 0.0) or 0.0)
        posterior_relevance = float(disease_row.get("posterior_relevance", 0.0) or 0.0)
        hemorrhage_relevance = float(disease_row.get("hemorrhage_relevance", 0.0) or 0.0)
        emergency_relevance = float(disease_row.get("emergency_relevance", 0.0) or 0.0)
        chronic_penalty = float(disease_row.get("chronic_penalty", 0.0) or 0.0)
        noise_penalty = float(disease_row.get("noise_penalty", 0.0) or 0.0)

        article_emergency_relevance = float(disease_row.get("article_emergency_relevance", 0.0) or 0.0)
        article_vascular_relevance = float(disease_row.get("article_vascular_relevance", 0.0) or 0.0)
        article_posterior_relevance = float(disease_row.get("article_posterior_relevance", 0.0) or 0.0)
        article_hemorrhage_relevance = float(disease_row.get("article_hemorrhage_relevance", 0.0) or 0.0)
        article_noise_penalty = float(disease_row.get("article_noise_penalty", 0.0) or 0.0)

        category = str(disease_row.get("category", "")).lower()

        if flags["acute"]:
            score += 0.70 * emergency_relevance
            score += 0.60 * vascular_relevance
            score += 0.30 * article_emergency_relevance
            score += 0.25 * article_vascular_relevance
            score -= 0.80 * chronic_penalty
            score -= 0.50 * noise_penalty
            score -= 0.35 * article_noise_penalty

        if flags["posterior"]:
            score += 0.75 * posterior_relevance
            score += 0.35 * vascular_relevance
            score += 0.35 * article_posterior_relevance
            score += 0.20 * article_vascular_relevance
            score -= 0.45 * chronic_penalty

        if flags["focal"]:
            score += 0.45 * vascular_relevance
            score += 0.25 * emergency_relevance

        # مهم:
        # focal بدون posterior يجب أن يفضّل general vascular stroke
        # ولا يجعل posterior_vascular يسيطر على النتائج الأولى.
        if flags["focal"] and not flags["posterior"]:
            if category == "posterior_vascular" or posterior_relevance >= 0.8:
                score -= 0.55

            if category == "vascular" and posterior_relevance < 0.5:
                score += 0.35

            if any(t in disease_low for t in self.general_vascular_terms):
                score += 0.25

            if any(t in disease_low for t in self.posterior_specific_terms):
                score -= 0.20

        if flags["hemorrhage"]:
            score += 0.90 * hemorrhage_relevance
            score += 0.30 * emergency_relevance
            score += 0.30 * article_hemorrhage_relevance
            score -= 0.40 * chronic_penalty

        if flags["posterior"]:
            if any(t in disease_low for t in [
                "posterior circulation", "vertebrobasilar", "brainstem",
                "medullary", "pontine", "midbrain"
            ]):
                score += 0.45

            if any(t in disease_low for t in [
                "bppv", "benign paroxysmal positional vertigo", "meniere",
                "ménière", "vestibular", "migrainous vertigo"
            ]):
                score -= 0.22

        if flags["acute"]:
            if any(t in disease_low for t in [
                "stroke", "infarction", "ischemia", "ischaemia",
                "vertebrobasilar", "hemorrhage", "haemorrhage", "tia"
            ]):
                score += 0.28

            if any(t in disease_low for t in [
                "ataxia telangiectasia", "spinocerebellar ataxia",
                "cerebellar ataxia", "genetic", "familial",
                "autosomal", "mutation", "chronic"
            ]):
                score -= 0.22

        if flags["focal"]:
            if any(t in disease_low for t in [
                "stroke", "infarction", "ischemia", "ischaemia",
                "transient ischemic attack", "tia", "vertebrobasilar"
            ]):
                score += 0.24

            if any(t in disease_low for t in [
                "bppv", "meniere", "ménière", "migraine", "vestibular"
            ]):
                score -= 0.16

        if flags["hemorrhage"]:
            if any(t in disease_low for t in [
                "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral", "sah", "ich"
            ]):
                score += 0.45

        score += 0.08 * min(len(matched_symptoms), 4)
        return score

    def _article_bias_score(self, article_title: str, disease_name: str, query: str, disease_row: Dict | None = None) -> float:
        title_low = self._normalize_text(article_title)
        disease_low = self._normalize_text(disease_name)
        flags = self._query_context_flags(query)
        disease_row = disease_row or {}

        score = 0.0

        score += 0.10 * min(self._count_term_hits(title_low, self.article_priority_terms), 3)
        score -= 0.08 * min(self._count_term_hits(title_low, self.article_penalty_terms), 3)

        vascular_relevance = float(disease_row.get("vascular_relevance", 0.0) or 0.0)
        posterior_relevance = float(disease_row.get("posterior_relevance", 0.0) or 0.0)
        hemorrhage_relevance = float(disease_row.get("hemorrhage_relevance", 0.0) or 0.0)
        emergency_relevance = float(disease_row.get("emergency_relevance", 0.0) or 0.0)
        chronic_penalty = float(disease_row.get("chronic_penalty", 0.0) or 0.0)
        noise_penalty = float(disease_row.get("noise_penalty", 0.0) or 0.0)

        article_emergency_relevance = float(disease_row.get("article_emergency_relevance", 0.0) or 0.0)
        article_vascular_relevance = float(disease_row.get("article_vascular_relevance", 0.0) or 0.0)
        article_posterior_relevance = float(disease_row.get("article_posterior_relevance", 0.0) or 0.0)
        article_hemorrhage_relevance = float(disease_row.get("article_hemorrhage_relevance", 0.0) or 0.0)
        article_noise_penalty = float(disease_row.get("article_noise_penalty", 0.0) or 0.0)

        category = str(disease_row.get("category", "")).lower()
        article_type = str(disease_row.get("article_type", "")).lower()

        if flags["acute"]:
            score += 0.12 * emergency_relevance
            score += 0.10 * vascular_relevance
            score += 0.20 * article_emergency_relevance
            score += 0.18 * article_vascular_relevance
            score -= 0.10 * chronic_penalty
            score -= 0.12 * noise_penalty
            score -= 0.18 * article_noise_penalty

        if flags["posterior"]:
            score += 0.14 * posterior_relevance
            score += 0.08 * vascular_relevance
            score += 0.20 * article_posterior_relevance
            score += 0.12 * article_vascular_relevance

            if any(t in title_low for t in [
                "posterior circulation", "vertebrobasilar", "brainstem",
                "cerebellar infarction", "cerebellar stroke",
                "medullary", "pontine", "midbrain"
            ]):
                score += 0.18

        # focal بدون posterior: خفّض posterior-specific articles وفضّل general vascular evidence
        if flags["focal"] and not flags["posterior"]:
            if category == "posterior_vascular" or "posterior" in article_type:
                score -= 0.25

            if category == "vascular" and "posterior" not in article_type:
                score += 0.20

            if any(t in title_low for t in [
                "acute ischemic stroke", "acute ischaemic stroke",
                "cerebral infarction", "ischemia", "ischaemia",
                "tpa", "thrombolysis", "transient ischemic attack"
            ]):
                score += 0.18

            if any(t in title_low for t in [
                "midbrain", "medullary", "pontine", "vertebrobasilar",
                "cerebellar infarction", "posterior circulation"
            ]):
                score -= 0.12

        if flags["acute"]:
            if any(t in title_low for t in [
                "acute", "stroke", "ischemic", "ischaemic", "infarction",
                "hemorrhage", "haemorrhage", "tia"
            ]):
                score += 0.14

        if flags["hemorrhage"]:
            score += 0.16 * hemorrhage_relevance
            score += 0.20 * article_hemorrhage_relevance
            if any(t in title_low for t in [
                "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral"
            ]):
                score += 0.18

        if self._count_term_hits(disease_low, self.stroke_priority_terms) > 0:
            score += 0.06

        return score

    def _get_item_value(self, item: Dict, *keys, default=None):
        for key in keys:
            if key in item and item.get(key) is not None:
                return item.get(key)
        return default

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Stroke-aware KG retrieval:
        1) extract symptoms from query
        2) infer clinical context flags + temporal context
        3) query disease candidates from graph with clinical constraints
        4) rescore diseases using graph properties + stroke-aware bias
        5) fetch related articles
        6) rescore articles so clinically relevant titles get promoted
        """
        symptoms = self.extract_symptoms_from_query(query)

        if not symptoms:
            return []

        flags = self._query_context_flags(query)
        context_details = self._query_context_details(query)

        disease_limit = max(top_k * 10, 50)

        diseases = self.kg.query_by_symptoms(
            symptoms,
            limit=disease_limit,
            is_acute=flags["acute"],
            is_posterior=flags["posterior"],
            is_focal=flags["focal"],
            is_hemorrhage=flags["hemorrhage"],
        )

        if not diseases:
            return []

        disease_rows: List[Dict] = []
        freqs: List[float] = []
        clinical_scores: List[float] = []
        query_low = self._normalize_text(query)

        for d in diseases:
            disease_name = d.get("disease")
            if not disease_name:
                continue

            try:
                freq = float(d.get("frequency", 0) or 0.0)
            except Exception:
                freq = 0.0

            try:
                clinical_score = float(d.get("clinical_score", 0) or 0.0)
            except Exception:
                clinical_score = 0.0

            matched_symptoms = self._match_query_symptoms_to_disease(
                disease_name=disease_name,
                symptoms=symptoms,
                query=query_low,
            )

            disease_rows.append({
                "disease": disease_name,
                "frequency": freq,
                "clinical_score": clinical_score,
                "matched_symptoms": matched_symptoms,
                "match_count": len(matched_symptoms),
                "category": d.get("category", "unknown"),
                "acuity": d.get("acuity", "unknown"),
                "vascular_relevance": float(d.get("vascular_relevance", 0.0) or 0.0),
                "posterior_relevance": float(d.get("posterior_relevance", 0.0) or 0.0),
                "hemorrhage_relevance": float(d.get("hemorrhage_relevance", 0.0) or 0.0),
                "emergency_relevance": float(d.get("emergency_relevance", 0.0) or 0.0),
                "chronic_penalty": float(d.get("chronic_penalty", 0.0) or 0.0),
                "noise_penalty": float(d.get("noise_penalty", 0.0) or 0.0),
                "article_emergency_relevance": float(d.get("article_emergency_relevance", 0.0) or 0.0),
                "article_vascular_relevance": float(d.get("article_vascular_relevance", 0.0) or 0.0),
                "article_posterior_relevance": float(d.get("article_posterior_relevance", 0.0) or 0.0),
                "article_hemorrhage_relevance": float(d.get("article_hemorrhage_relevance", 0.0) or 0.0),
                "article_noise_penalty": float(d.get("article_noise_penalty", 0.0) or 0.0),
            })

            freqs.append(freq)
            clinical_scores.append(clinical_score)

        if not disease_rows:
            return []

        freq_norms = self._minmax_norm(freqs)
        clinical_norms = self._minmax_norm(clinical_scores)

        rescored_rows: List[Dict] = []
        disease_raw_scores: List[float] = []

        for row, freq_norm, clinical_norm in zip(disease_rows, freq_norms, clinical_norms):
            symptom_ratio = row["match_count"] / max(len(symptoms), 1)

            base_score = (
                0.15 * float(freq_norm)
                + 0.25 * float(symptom_ratio)
                + 0.45 * float(clinical_norm)
            )

            bias_score = self._stroke_bias_score(
                disease_name=row["disease"],
                query=query_low,
                matched_symptoms=row["matched_symptoms"],
                disease_row=row,
            )

            raw_score = base_score + bias_score

            new_row = dict(row)
            new_row["base_score"] = base_score
            new_row["bias_score"] = bias_score
            new_row["raw_disease_score"] = raw_score
            rescored_rows.append(new_row)
            disease_raw_scores.append(raw_score)

        disease_norms = self._minmax_norm(disease_raw_scores)
        for row, norm_score in zip(rescored_rows, disease_norms):
            row["graph_score"] = float(norm_score)

        filtered_rows = [r for r in rescored_rows if float(r.get("graph_score", 0.0)) >= 0.20]

        if not filtered_rows:
            filtered_rows = sorted(
                rescored_rows,
                key=lambda x: float(x.get("graph_score", 0.0) or 0.0),
                reverse=True
            )[:max(top_k * 4, 12)]
        else:
            filtered_rows.sort(
                key=lambda x: float(x.get("graph_score", 0.0) or 0.0),
                reverse=True
            )

        article_rows: List[Dict] = []
        article_raw_scores: List[float] = []

        for row in filtered_rows[: max(top_k * 4, 15)]:
            related = self.kg.get_related_articles(disease=row["disease"], limit=4)

            for item in related:
                pmid = self._get_item_value(item, "pmid", "a.pmid")
                title = self._get_item_value(item, "title", "a.title")
                chunk_id = self._get_item_value(item, "chunk_id", "a.chunk_id")

                if not pmid or not title:
                    continue

                article_context = dict(row)
                article_context.update({
                    "article_type": self._get_item_value(item, "article_type", "a.article_type", default=row.get("article_type", "unknown")),
                    "article_emergency_relevance": float(self._get_item_value(item, "article_emergency_relevance", "a.article_emergency_relevance", default=row.get("article_emergency_relevance", 0.0)) or 0.0),
                    "article_vascular_relevance": float(self._get_item_value(item, "article_vascular_relevance", "a.article_vascular_relevance", default=row.get("article_vascular_relevance", 0.0)) or 0.0),
                    "article_posterior_relevance": float(self._get_item_value(item, "article_posterior_relevance", "a.article_posterior_relevance", default=row.get("article_posterior_relevance", 0.0)) or 0.0),
                    "article_hemorrhage_relevance": float(self._get_item_value(item, "article_hemorrhage_relevance", "a.article_hemorrhage_relevance", default=row.get("article_hemorrhage_relevance", 0.0)) or 0.0),
                    "article_noise_penalty": float(self._get_item_value(item, "article_noise_penalty", "a.article_noise_penalty", default=row.get("article_noise_penalty", 0.0)) or 0.0),
                    "category": self._get_item_value(item, "category", "d.category", default=row.get("category", "unknown")),
                    "acuity": self._get_item_value(item, "acuity", "d.acuity", default=row.get("acuity", "unknown")),
                    "vascular_relevance": float(self._get_item_value(item, "vascular_relevance", "d.vascular_relevance", default=row.get("vascular_relevance", 0.0)) or 0.0),
                    "posterior_relevance": float(self._get_item_value(item, "posterior_relevance", "d.posterior_relevance", default=row.get("posterior_relevance", 0.0)) or 0.0),
                    "hemorrhage_relevance": float(self._get_item_value(item, "hemorrhage_relevance", "d.hemorrhage_relevance", default=row.get("hemorrhage_relevance", 0.0)) or 0.0),
                    "emergency_relevance": float(self._get_item_value(item, "emergency_relevance", "d.emergency_relevance", default=row.get("emergency_relevance", 0.0)) or 0.0),
                    "chronic_penalty": float(self._get_item_value(item, "chronic_penalty", "d.chronic_penalty", default=row.get("chronic_penalty", 0.0)) or 0.0),
                    "noise_penalty": float(self._get_item_value(item, "noise_penalty", "d.noise_penalty", default=row.get("noise_penalty", 0.0)) or 0.0),
                })

                article_bias = self._article_bias_score(
                    article_title=title,
                    disease_name=row["disease"],
                    query=query_low,
                    disease_row=article_context,
                )

                raw_article_score = float(row["graph_score"]) + article_bias

                article = {
                    "pmid": pmid,
                    "title": title,
                    "chunk_id": chunk_id,
                    "source": "GraphKB",
                    "disease": row["disease"],
                    "category": article_context.get("category", "unknown"),
                    "acuity": article_context.get("acuity", "unknown"),
                    "article_type": article_context.get("article_type", "unknown"),
                    "matched_symptoms": row["matched_symptoms"],
                    "match_count": row["match_count"],
                    "disease_frequency": row["frequency"],
                    "clinical_score": row.get("clinical_score", 0.0),
                    "base_score": row["base_score"],
                    "bias_score": row["bias_score"],
                    "article_bias_score": article_bias,
                    "raw_article_score": raw_article_score,
                    "vascular_relevance": article_context.get("vascular_relevance", 0.0),
                    "posterior_relevance": article_context.get("posterior_relevance", 0.0),
                    "hemorrhage_relevance": article_context.get("hemorrhage_relevance", 0.0),
                    "emergency_relevance": article_context.get("emergency_relevance", 0.0),
                    "chronic_penalty": article_context.get("chronic_penalty", 0.0),
                    "noise_penalty": article_context.get("noise_penalty", 0.0),
                    "article_emergency_relevance": article_context.get("article_emergency_relevance", 0.0),
                    "article_vascular_relevance": article_context.get("article_vascular_relevance", 0.0),
                    "article_posterior_relevance": article_context.get("article_posterior_relevance", 0.0),
                    "article_hemorrhage_relevance": article_context.get("article_hemorrhage_relevance", 0.0),
                    "article_noise_penalty": article_context.get("article_noise_penalty", 0.0),
                    "query_flags": flags,
                    "context_details": context_details,
                }

                article_rows.append(article)
                article_raw_scores.append(raw_article_score)

        if not article_rows:
            return []

        article_norms = self._minmax_norm(article_raw_scores)
        for row, norm_score in zip(article_rows, article_norms):
            row["graph_score"] = float(norm_score)

        best_by_key: Dict[str, Dict] = {}

        for a in article_rows:
            dedup_key = str(a.get("chunk_id") or a.get("pmid"))

            prev = best_by_key.get(dedup_key)
            if prev is None:
                best_by_key[dedup_key] = a
                continue

            prev_score = float(prev.get("graph_score", 0.0) or 0.0)
            new_score = float(a.get("graph_score", 0.0) or 0.0)

            if new_score > prev_score:
                best_by_key[dedup_key] = a

        ordered = sorted(
            best_by_key.values(),
            key=lambda x: float(x.get("graph_score", 0.0) or 0.0),
            reverse=True
        )

        return ordered[:top_k]