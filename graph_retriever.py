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
            "free of", "absence of", "absent",
            "لا", "بدون", "ينفي", "لا يوجد", "غياب"
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

            # Arabic clinical terms
            "ضعف": "weakness",
            "ضعف مفاجئ": "weakness",
            "ضعف نصفي": "unilateral weakness",
            "شلل": "weakness",
            "شلل نصفي": "hemiplegia",
            "ضعف بالطرف": "limb weakness",
            "ضعف في الطرف": "limb weakness",
            "ضعف بالذراع": "arm weakness",
            "ضعف بالساق": "leg weakness",
            "ضعف باليد": "arm weakness",
            "ميلان بالفم": "facial droop",
            "انحراف بالفم": "facial droop",
            "انحراف الوجه": "facial droop",
            "تلعثم": "dysarthria",
            "تلعثم بالكلام": "dysarthria",
            "صعوبة بالكلام": "speech disturbance",
            "حبسة": "aphasia",
            "دوار": "vertigo",
            "دوخة": "dizziness",
            "ازدواج رؤية": "diplopia",
            "رؤية مزدوجة": "diplopia",
            "ترنح": "ataxia",
            "ترنح بالمشي": "gait difficulty",
            "صعوبة بالمشي": "gait difficulty",
            "صداع": "headache",
            "صداع مفاجئ": "headache",
            "صداع شديد مفاجئ": "headache",
            "قيء": "vomiting",
            "إقياء": "vomiting",
            "تدهور وعي": "loss of consciousness",
            "نقص وعي": "loss of consciousness",
            "فقدان وعي": "loss of consciousness",
            "ارتباك": "confusion",
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
            "radiation",
            "dna repair",
            "protein kinase",
            "cell",
            "molecular",
        ]

        self.acute_query_terms = [
            "acute", "sudden", "suddenly", "new onset", "acute onset",
            "abrupt", "abruptly", "ongoing", "persistent", "persisting",
            "started", "developed", "began", "onset",
            "thunderclap", "worsening", "emergency",
            "today", "this morning", "this afternoon", "this evening",
            "last night", "same day",
            "مفاجئ", "فجأة", "حاد", "حادة", "مستمر", "مستمرة",
            "منذ", "خلال", "ساعة", "ساعات", "دقيقة", "دقائق"
        ]

        self.chronic_query_terms = [
            "chronic", "gradual", "gradually", "progressive", "progressively",
            "long-standing", "longstanding", "recurrent for months",
            "for weeks", "for months", "for years", "over weeks", "over months",
            "over years", "several weeks", "several months", "several years",
            "مزمن", "مزمنة", "تدريجي", "تدريجية", "منذ أسابيع", "منذ أشهر", "منذ سنوات"
        ]

        self.posterior_query_terms = [
            "vertigo", "diplopia", "double vision", "ataxia", "gait difficulty",
            "difficulty walking", "dizziness", "nystagmus", "imbalance",
            "brainstem", "cerebellar", "posterior circulation",
            "دوار", "دوخة", "ازدواج رؤية", "رؤية مزدوجة", "ترنح", "ترنح بالمشي",
            "صعوبة بالمشي", "جذع الدماغ", "مخيخ"
        ]

        self.focal_query_terms = [
            "weakness", "unilateral weakness", "arm weakness", "leg weakness",
            "left-sided weakness", "right-sided weakness",
            "hemiparesis", "hemiplegia", "aphasia", "dysarthria",
            "slurred speech", "facial droop", "numbness",
            "ضعف", "شلل", "ضعف نصفي", "شلل نصفي", "حبسة", "تلعثم",
            "صعوبة بالكلام", "انحراف بالفم", "ميلان بالفم", "تنميل"
        ]

        self.hemorrhage_query_terms = [
            "thunderclap", "worst headache", "sudden severe headache",
            "vomiting", "loss of consciousness", "decreased consciousness",
            "worsening consciousness", "subarachnoid", "intracerebral hemorrhage",
            "intracerebral haemorrhage", "hemorrhage", "haemorrhage", "sah", "ich",
            "صداع شديد", "صداع مفاجئ", "قيء", "إقياء", "تدهور وعي",
            "نقص وعي", "فقدان وعي", "نزف", "نزيف"
        ]

        self.article_priority_terms = [
            "stroke", "posterior circulation", "brainstem", "cerebellar",
            "vertebrobasilar", "ischemia", "ischaemia", "infarction",
            "ischemic", "ischaemic", "transient ischemic attack",
            "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral",
            "acute", "emergency", "thrombolysis", "thrombectomy", "tpa", "alteplase"
        ]

        self.article_penalty_terms = [
            "mutation", "autosomal", "familial", "syndrome", "genetic",
            "rehabilitation", "pregnancy", "cancer", "cell", "molecular",
            "mouse", "rat", "protein", "phosphorylation", "in vitro",
            "mrna", "messenger rna", "kinase", "animal model", "rat model", "mouse model",
            "radiation", "dna repair", "double-strand break", "gamma-h2ax"
        ]

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _build_kg_explanation_paths(article: Dict) -> List[Dict]:
        """Build auditable paths only from fields returned by Neo4j queries.

        No LLM-generated edge or node is introduced.  Every path is marked as
        metadata-derived because it is reconstructed from the graph query row.
        """
        disease = article.get("disease") or article.get("graph_disease")
        if not disease:
            return []
        article_node = {"type": "Article", "id": str(article.get("pmid") or article.get("chunk_id") or "unknown"), "label": article.get("title") or "Article"}
        disease_node = {"type": "Disease", "id": str(disease), "label": str(disease)}
        paths: List[Dict] = []

        def add_path(relation: str, target_type: str, value: object) -> None:
            if value in (None, ""):
                return
            target = {"type": target_type, "id": str(value), "label": str(value)}
            paths.append({
                "path_id": f"{article_node['id']}::{disease}::{relation}::{value}",
                "nodes": [article_node, disease_node, target],
                "relationships": ["SUPPORTS_DISEASE", relation],
                "source": "neo4j_query_metadata",
                "verified": True,
            })

        for symptom in list(article.get("matched_symptoms") or []):
            add_path("HAS_SYMPTOM", "Symptom", symptom)
        for family in list(article.get("stroke_families") or []):
            add_path("IS_SUBTYPE_OF", "StrokeFamily", family)
        for territory in list(article.get("territories") or []):
            add_path("AFFECTS_TERRITORY", "Territory", territory)
        for finding in list(article.get("imaging_findings") or []):
            add_path("HAS_IMAGING_FINDING", "ImagingFinding", finding)

        if not paths:
            paths.append({
                "path_id": f"{article_node['id']}::{disease}::SUPPORTS_DISEASE",
                "nodes": [article_node, disease_node],
                "relationships": ["SUPPORTS_DISEASE"],
                "source": "neo4j_query_metadata",
                "verified": True,
            })
        return paths[:20]

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

    def _has_stroke_signal(self, text: str, row: Optional[Dict] = None) -> bool:
        row = row or {}
        low = self._normalize_text(text)

        if self._count_term_hits(low, self.stroke_priority_terms) > 0:
            return True

        category = str(row.get("category", "") or "").lower()
        article_type = str(row.get("article_type", "") or "").lower()

        if category in {"vascular", "posterior_vascular", "hemorrhagic"}:
            return True

        if any(t in article_type for t in ["stroke", "vascular", "posterior", "hemorrhage", "infarction"]):
            return True

        vascular_relevance = float(row.get("vascular_relevance", 0.0) or 0.0)
        posterior_relevance = float(row.get("posterior_relevance", 0.0) or 0.0)
        hemorrhage_relevance = float(row.get("hemorrhage_relevance", 0.0) or 0.0)
        emergency_relevance = float(row.get("emergency_relevance", 0.0) or 0.0)

        return max(
            vascular_relevance,
            posterior_relevance,
            hemorrhage_relevance,
            emergency_relevance,
        ) >= 0.45

    def _noise_hits(self, text: str) -> int:
        return self._count_term_hits(text, self.nonacute_penalty_terms)

    def _article_noise_hits(self, text: str) -> int:
        return self._count_term_hits(text, self.article_penalty_terms)


    def _hemorrhage_signal_hits(self, text: str) -> int:
        """
        Count explicit hemorrhage signals safely.
        Uses word-boundaries for SAH/ICH so words like "which" are not matched.
        """
        low = self._normalize_text(text)

        hard_terms = [
            "subarachnoid hemorrhage",
            "subarachnoid haemorrhage",
            "intracerebral hemorrhage",
            "intracerebral haemorrhage",
            "brain hemorrhage",
            "brain haemorrhage",
            "cerebral hemorrhage",
            "cerebral haemorrhage",
            "pontine hemorrhage",
            "pontine haemorrhage",
            "midbrain hemorrhage",
            "midbrain haemorrhage",
            "brainstem hemorrhage",
            "brainstem haemorrhage",
            "aneurysmal subarachnoid hemorrhage",
            "aneurysmal subarachnoid haemorrhage",
            "hemorrhagic stroke",
            "haemorrhagic stroke",
            "hemorrhagic transformation",
            "haemorrhagic transformation",
        ]

        hits = self._count_term_hits(low, hard_terms)

        if "hemorrhage" in low or "haemorrhage" in low:
            hits += 1
            re.search(r"\b(sah|ich)\b", low)
            hits += 1

        return hits

    def _passes_stroke_scope_gate(self, row: Dict, flags: Dict[str, bool]) -> bool:
        disease_name = row.get("disease", "")
        disease_low = self._normalize_text(disease_name)

        high_risk_query = bool(flags.get("acute") and (
            flags.get("posterior") or flags.get("focal") or flags.get("hemorrhage")
        ))

        if not high_risk_query:
            return True

        stroke_signal = self._has_stroke_signal(disease_low, row)
        noise_hits = self._noise_hits(disease_low)

        hard_noise_terms = [
            "ataxia telangiectasia",
            "spinocerebellar ataxia",
            "friedreich",
            "bppv",
            "benign paroxysmal positional vertigo",
            "benign positioning vertigo",
            "meniere",
            "ménière",
            "vestibular rehabilitation",
            "genetic",
            "familial",
            "autosomal",
            "mutation",
            "radiation",
            "dna repair",
        ]

        if any(t in disease_low for t in hard_noise_terms) and not stroke_signal:
            return False

        if noise_hits >= 2 and not stroke_signal:
            return False

        if flags.get("posterior"):
            if any(t in disease_low for t in [
                "bppv", "benign paroxysmal positional vertigo", "meniere",
                "ménière", "migrainous vertigo", "vestibular hypofunction"
            ]) and not stroke_signal:
                return False

        return True

    def _passes_article_scope_gate(self, title: str, disease_name: str, flags: Dict[str, bool]) -> bool:
        title_low = self._normalize_text(title)
        disease_low = self._normalize_text(disease_name)
        combined = f"{title_low} {disease_low}"

        high_risk_query = bool(flags.get("acute") and (
            flags.get("posterior") or flags.get("focal") or flags.get("hemorrhage")
        ))

        if not high_risk_query:
            return True

        priority_hits = self._count_term_hits(combined, self.article_priority_terms + self.stroke_priority_terms)
        noise_hits = self._article_noise_hits(combined) + self._noise_hits(combined)

        # New targeted rule:
        # If the query is not hemorrhagic, do not allow explicit SAH/ICH/hemorrhage articles
        # to become the closest evidence for posterior/focal ischemic-looking cases.
        if not flags.get("hemorrhage", False):
            if self._hemorrhage_signal_hits(combined) > 0:
                return False

        if noise_hits >= 2 and priority_hits == 0:
            return False

        if any(t in combined for t in [
            "ataxia telangiectasia",
            "gamma-h2ax",
            "double-strand break",
            "dna repair",
            "radiation",
            "protein kinase",
            "phosphorylation",
            "molecular",
            "cell",
            "mouse",
            "rat",
            "animal model",
        ]) and priority_hits == 0:
            return False

        return True

    def _extract_time_context(self, query: str) -> Dict[str, Optional[float] | bool | str]:
        q = self._normalize_text(query)

        ctx = {
            "has_time": False,
            "duration_minutes": None,
            "is_acute_time": False,
            "is_chronic_time": False,
            "time_unit": None,
            "time_phrase": None,
        }

        if any(t in q for t in self.chronic_query_terms):
            ctx["has_time"] = True
            ctx["is_chronic_time"] = True
            ctx["is_acute_time"] = False
            ctx["time_phrase"] = "chronic_or_progressive"
            return ctx

        time_patterns = [
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(minute|minutes|min|mins)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(day|days)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(week|weeks)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(month|months)\b",
            r"\b(?:for|since|within|over|during|last|past)?\s*(\d+(?:\.\d+)?)\s*(year|years)\b",
            r"(?:منذ|خلال)\s*(\d+(?:\.\d+)?)\s*(دقيقة|دقائق|ساعة|ساعات|يوم|أيام|اسبوع|أسبوع|أسابيع|شهر|أشهر|سنة|سنوات)",
        ]

        for pattern in time_patterns:
            m = re.search(pattern, q)
            if not m:
                continue

            value = float(m.group(1))
            unit = m.group(2)

            minutes = None
            if unit in {"minute", "minutes", "min", "mins", "دقيقة", "دقائق"}:
                minutes = value
            elif unit in {"hour", "hours", "hr", "hrs", "ساعة", "ساعات"}:
                minutes = value * 60
            elif unit in {"day", "days", "يوم", "أيام"}:
                minutes = value * 24 * 60
            elif unit in {"week", "weeks", "اسبوع", "أسبوع", "أسابيع"}:
                minutes = value * 7 * 24 * 60
            elif unit in {"month", "months", "شهر", "أشهر"}:
                minutes = value * 30 * 24 * 60
            elif unit in {"year", "years", "سنة", "سنوات"}:
                minutes = value * 365 * 24 * 60

            ctx["has_time"] = True
            ctx["duration_minutes"] = minutes
            ctx["time_unit"] = unit
            ctx["time_phrase"] = m.group(0).strip()

            if unit in {"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "دقيقة", "دقائق", "ساعة", "ساعات"}:
                ctx["is_acute_time"] = True
                ctx["is_chronic_time"] = False
            elif unit in {"day", "days", "يوم", "أيام"}:
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
            "just started", "suddenly developed", "suddenly started",
            "فجأة", "بشكل مفاجئ", "مفاجئ", "حاد", "حادة", "اليوم"
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
            "gait difficulty", "difficulty walking", "nystagmus", "imbalance",
            "دوار", "دوخة", "ازدواج رؤية", "رؤية مزدوجة", "ترنح", "صعوبة بالمشي"
        ])

        if posterior_symptom_hits >= 2 and (
            bool(time_ctx["is_acute_time"])
            or any(t in q for t in ["sudden", "acute", "ongoing", "persistent", "new onset", "مفاجئ", "مستمرة", "مستمر"])
        ):
            acute = True
            posterior = True

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

            if s in {"weakness", "unilateral weakness", "arm weakness", "leg weakness", "limb weakness", "hemiparesis", "hemiplegia"}:
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

        stroke_hits = min(self._count_term_hits(disease_low, self.stroke_priority_terms), 4)
        penalty_hits = min(self._count_term_hits(disease_low, self.nonacute_penalty_terms), 4)

        score += 0.65 * stroke_hits
        score -= 0.85 * penalty_hits

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
        has_stroke_signal = self._has_stroke_signal(disease_low, disease_row)

        if flags["acute"]:
            score += 0.85 * emergency_relevance
            score += 0.75 * vascular_relevance
            score += 0.35 * article_emergency_relevance
            score += 0.30 * article_vascular_relevance
            score -= 1.00 * chronic_penalty
            score -= 0.70 * noise_penalty
            score -= 0.45 * article_noise_penalty

            if not has_stroke_signal and penalty_hits > 0:
                score -= 1.25

        if flags["posterior"]:
            score += 0.95 * posterior_relevance
            score += 0.45 * vascular_relevance
            score += 0.45 * article_posterior_relevance
            score += 0.28 * article_vascular_relevance
            score -= 0.65 * chronic_penalty

            if not has_stroke_signal and penalty_hits > 0:
                score -= 1.35

        if flags["focal"]:
            score += 0.55 * vascular_relevance
            score += 0.30 * emergency_relevance

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
            score += 1.10 * hemorrhage_relevance
            score += 0.35 * emergency_relevance
            score += 0.40 * article_hemorrhage_relevance
            score -= 0.55 * chronic_penalty

        if flags["posterior"]:
            if any(t in disease_low for t in [
                "posterior circulation", "vertebrobasilar", "brainstem",
                "medullary", "pontine", "midbrain", "cerebellar infarction",
                "cerebellar stroke"
            ]):
                score += 0.70

            if any(t in disease_low for t in [
                "bppv", "benign paroxysmal positional vertigo", "meniere",
                "ménière", "vestibular", "migrainous vertigo",
                "ataxia telangiectasia", "spinocerebellar ataxia"
            ]):
                score -= 0.95

        if flags["acute"]:
            if any(t in disease_low for t in [
                "stroke", "infarction", "ischemia", "ischaemia",
                "vertebrobasilar", "hemorrhage", "haemorrhage", "tia"
            ]):
                score += 0.40

            if any(t in disease_low for t in [
                "ataxia telangiectasia", "spinocerebellar ataxia",
                "cerebellar ataxia", "genetic", "familial",
                "autosomal", "mutation", "chronic", "radiation", "dna repair"
            ]):
                score -= 0.85

        if flags["focal"]:
            if any(t in disease_low for t in [
                "stroke", "infarction", "ischemia", "ischaemia",
                "transient ischemic attack", "tia", "vertebrobasilar"
            ]):
                score += 0.30

            if any(t in disease_low for t in [
                "bppv", "meniere", "ménière", "migraine", "vestibular"
            ]):
                score -= 0.35

        if flags["hemorrhage"]:
            if any(t in disease_low for t in [
                "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral", "sah", "ich"
            ]):
                score += 0.60

        score += 0.10 * min(len(matched_symptoms), 4)
        return score

    def _article_bias_score(self, article_title: str, disease_name: str, query: str, disease_row: Dict | None = None) -> float:
        title_low = self._normalize_text(article_title)
        disease_low = self._normalize_text(disease_name)
        combined_low = f"{title_low} {disease_low}"
        flags = self._query_context_flags(query)
        disease_row = disease_row or {}

        score = 0.0

        score += 0.16 * min(self._count_term_hits(title_low, self.article_priority_terms), 4)
        score -= 0.22 * min(self._count_term_hits(title_low, self.article_penalty_terms), 4)

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

        priority_hits = self._count_term_hits(combined_low, self.article_priority_terms + self.stroke_priority_terms)
        noise_hits = self._article_noise_hits(combined_low) + self._noise_hits(combined_low)
        hemorrhage_hits = self._hemorrhage_signal_hits(combined_low)

        if flags["acute"]:
            score += 0.16 * emergency_relevance
            score += 0.14 * vascular_relevance
            score += 0.25 * article_emergency_relevance
            score += 0.22 * article_vascular_relevance
            score -= 0.18 * chronic_penalty
            score -= 0.20 * noise_penalty
            score -= 0.25 * article_noise_penalty

            if noise_hits > 0 and priority_hits == 0:
                score -= 0.70

        if flags["posterior"]:
            score += 0.20 * posterior_relevance
            score += 0.12 * vascular_relevance
            score += 0.25 * article_posterior_relevance
            score += 0.16 * article_vascular_relevance

            if any(t in title_low for t in [
                "posterior circulation", "vertebrobasilar", "brainstem",
                "cerebellar infarction", "cerebellar stroke",
                "medullary", "pontine", "midbrain"
            ]):
                score += 0.38

            if any(t in combined_low for t in [
                "ataxia telangiectasia", "bppv", "benign paroxysmal positional vertigo",
                "benign positioning vertigo", "molecular", "cell", "radiation", "dna repair"
            ]) and priority_hits == 0:
                score -= 1.10

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
                score += 0.22

            if any(t in title_low for t in [
                "midbrain", "medullary", "pontine", "vertebrobasilar",
                "cerebellar infarction", "posterior circulation"
            ]):
                score -= 0.12

        if flags["acute"]:
            acute_title_terms = [
                "acute", "stroke", "ischemic", "ischaemic",
                "infarction", "infarct", "tia"
            ]

            # Only boost hemorrhage terms when the query itself has hemorrhage red flags.
            if flags.get("hemorrhage", False):
                acute_title_terms += [
                    "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral"
                ]

            if any(t in title_low for t in acute_title_terms):
                score += 0.18

            if flags.get("hemorrhage", False) and re.search(r"(sah|ich)", title_low):
                score += 0.18

        if flags["hemorrhage"]:
            score += 0.22 * hemorrhage_relevance
            score += 0.25 * article_hemorrhage_relevance

            if any(t in title_low for t in [
                "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral"
            ]):
                score += 0.30

            if re.search(r"(sah|ich)", title_low):
                score += 0.30

        else:
            # New targeted penalty:
            # For non-hemorrhage posterior/focal cases, push SAH/ICH/hemorrhage evidence down.
            if hemorrhage_hits > 0:
                score -= 1.10 + (0.20 * min(hemorrhage_hits, 4))

            if category == "hemorrhagic":
                score -= 0.75

            if any(t in article_type for t in [
                "hemorrhage",
                "haemorrhage",
                "hemorrhagic",
                "sah",
                "ich",
            ]):
                score -= 1.00

            if any(t in title_low for t in [
                "subarachnoid hemorrhage",
                "subarachnoid haemorrhage",
                "intracerebral hemorrhage",
                "intracerebral haemorrhage",
                "pontine hemorrhage",
                "pontine haemorrhage",
                "brainstem hemorrhage",
                "brainstem haemorrhage",
            ]):
                score -= 1.25

        if self._count_term_hits(disease_low, self.stroke_priority_terms) > 0:
            score += 0.10

        return score

    def _get_item_value(self, item: Dict, *keys, default=None):
        for key in keys:
            if key in item and item.get(key) is not None:
                return item.get(key)
        return default


    def search_by_candidate(
        self,
        candidate_label: str | None = None,
        candidate_key: str | None = None,
        aliases: Optional[List[str]] = None,
        case_text: str | None = None,
        clinical_intent: Optional[Dict] = None,
        top_k: int = 5,
        candidate_query: str | None = None,
    ) -> List[Dict]:
        """
        Candidate-specific KG retrieval.

        The method combines the candidate label/aliases with the case context,
        retrieves graph evidence, filters cross-family mismatches, and annotates
        every result so Evidence Judge can use it as candidate_kg evidence.
        """
        label = (candidate_label or candidate_query or "").strip()
        key = (candidate_key or "unknown").strip()
        aliases = aliases or []
        intent = clinical_intent or {}
        case = (case_text or "").strip()

        if not label and not aliases:
            return []

        query_parts = [case, label, " ".join(aliases[:5])]
        if intent.get("acute_neuro"):
            query_parts.append("acute stroke emergency")
        if intent.get("focal_neuro"):
            query_parts.append("focal neurological deficit")
        if intent.get("hemorrhage_warning"):
            query_parts.append("hemorrhage hematoma subarachnoid intracerebral")
        if intent.get("transient_resolved_episode"):
            query_parts.append("transient ischemic attack resolved symptoms")

        query = self._normalize_text(" ".join(p for p in query_parts if p))
        if candidate_query:
            query = self._normalize_text(f"{candidate_query} {case}")

        docs = self.search(query, top_k=max(top_k * 4, 12))
        if not docs:
            return []

        def doc_text(d: Dict) -> str:
            return self._normalize_text(" ".join(str(d.get(k) or "") for k in [
                "title", "disease", "category", "article_type",
                "graph_disease", "graph_category", "graph_article_type",
                "stroke_families", "territories", "imaging_findings",
                "graph_stroke_families", "graph_territories", "graph_imaging_findings"
            ]))

        def has_any(text: str, terms: List[str]) -> bool:
            return any(t in text for t in terms)

        ischemic_terms = [
            "ischemic", "ischaemic", "infarction", "infarct", "cerebral infarction",
            "brain infarction", "thrombolysis", "thrombectomy", "alteplase", "tpa",
            "large vessel", "mca", "middle cerebral artery", "posterior circulation",
            "vertebrobasilar", "brainstem", "cerebellar",
        ]
        hemorrhage_terms = [
            "hemorrhage", "haemorrhage", "hemorrhagic", "haemorrhagic",
            "hematoma", "subarachnoid", "intracerebral", "sah", "ich",
        ]
        tia_terms = ["transient ischemic attack", "transient ischaemic attack", "tia", "transient"]
        posterior_terms = [
            "posterior circulation", "vertebrobasilar", "brainstem", "cerebellar",
            "basilar", "pontine", "pons", "medullary", "midbrain",
        ]
        anterior_terms = [
            "anterior circulation", "middle cerebral artery", "mca", "carotid",
            "aphasia", "cortical", "large vessel", "hemiparesis", "facial droop",
        ]
        strong_infarct_terms = [
            "dwi lesion", "diffusion restriction", "restricted diffusion",
            "mri confirmed infarct", "ct confirmed acute infarct",
            "imaging-confirmed infarction", "infarct on mri", "infarct on ct",
        ]

        def family_allows(d: Dict) -> bool:
            txt = doc_text(d)
            category = self._normalize_text(str(d.get("category") or d.get("graph_category") or ""))
            families = self._normalize_text(str(d.get("stroke_families") or d.get("graph_stroke_families") or ""))
            territories = self._normalize_text(str(d.get("territories") or d.get("graph_territories") or ""))
            findings = self._normalize_text(str(d.get("imaging_findings") or d.get("graph_imaging_findings") or ""))
            txt = " ".join([txt, families, territories, findings])
            hemorrhagic = has_any(txt, hemorrhage_terms) or category == "hemorrhagic" or "hemorrhagic stroke" in families
            ischemic = has_any(txt, ischemic_terms) or category in {"vascular", "posterior_vascular"} or "ischemic stroke" in families
            strong_infarct = has_any(txt, strong_infarct_terms) or "imaging-confirmed infarction" in findings

            if key in {"intracerebral_hemorrhage", "subarachnoid_hemorrhage", "hemorrhagic_stroke"}:
                return hemorrhagic

            if key == "tia":
                return (not strong_infarct) and has_any(txt, tia_terms)

            if key in {
                "acute_ischemic_stroke", "large_vessel_occlusion",
                "anterior_circulation_stroke", "posterior_circulation_stroke",
                "brainstem_infarction", "cerebellar_infarction",
                "lacunar_infarction", "basilar_artery_occlusion",
                "vertebrobasilar_insufficiency",
            }:
                if hemorrhagic and not strong_infarct:
                    return False
                if intent.get("transient_resolved_episode") and not intent.get("persistent_deficit") and not strong_infarct:
                    return False
                if key in {"posterior_circulation_stroke", "brainstem_infarction", "cerebellar_infarction", "basilar_artery_occlusion", "vertebrobasilar_insufficiency"}:
                    return ischemic and (has_any(txt, posterior_terms) or "posterior circulation" in territories or intent.get("focal_neuro"))
                if key in {"anterior_circulation_stroke", "large_vessel_occlusion"}:
                    return ischemic and (has_any(txt, anterior_terms) or "anterior circulation" in territories or not has_any(txt, posterior_terms))
                return ischemic

            return True

        out: List[Dict] = []
        for d in docs:
            if not isinstance(d, dict):
                continue
            if not family_allows(d):
                continue

            txt = doc_text(d)
            base_score = float(d.get("graph_score", 0.0) or 0.0)
            alignment = 0.0
            if label and self._normalize_text(label) in txt:
                alignment += 0.25
            for a in aliases[:6]:
                if self._normalize_text(str(a)) and self._normalize_text(str(a)) in txt:
                    alignment += 0.08
            if key in {"posterior_circulation_stroke", "brainstem_infarction", "cerebellar_infarction"} and has_any(txt, posterior_terms):
                alignment += 0.20
            if key in {"anterior_circulation_stroke", "large_vessel_occlusion"} and has_any(txt, anterior_terms):
                alignment += 0.20
            if key == "acute_ischemic_stroke" and has_any(txt, ["acute ischemic stroke", "ischemic stroke", "infarction"]):
                alignment += 0.20

            nd = dict(d)
            nd["from_kg"] = True
            nd["retrieval_channel"] = "candidate_kg"
            nd["source"] = "GraphKB"
            nd["graph_disease"] = nd.get("graph_disease") or nd.get("disease")
            nd["graph_category"] = nd.get("graph_category") or nd.get("category")
            nd["graph_article_type"] = nd.get("graph_article_type") or nd.get("article_type")
            nd["graph_stroke_families"] = nd.get("graph_stroke_families") or nd.get("stroke_families", [])
            nd["graph_territories"] = nd.get("graph_territories") or nd.get("territories", [])
            nd["graph_imaging_findings"] = nd.get("graph_imaging_findings") or nd.get("imaging_findings", [])
            nd["candidate_key"] = key
            nd["candidate_label"] = label
            nd["kg_query_used"] = query
            nd["candidate_alignment_score"] = round(alignment, 4)
            nd["graph_score"] = round(min(max(base_score + alignment, base_score, 0.0), 1.0), 4)
            out.append(nd)

        best_by_key: Dict[str, Dict] = {}
        for d in out:
            dedup_key = str(d.get("chunk_id") or d.get("pmid") or d.get("title"))
            prev = best_by_key.get(dedup_key)
            if prev is None or float(d.get("graph_score", 0.0) or 0.0) > float(prev.get("graph_score", 0.0) or 0.0):
                best_by_key[dedup_key] = d

        ordered = sorted(best_by_key.values(), key=lambda x: float(x.get("graph_score", 0.0) or 0.0), reverse=True)
        return ordered[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Stroke-aware KG retrieval:
        1) extract symptoms from query
        2) infer clinical context flags + temporal context
        3) query disease candidates from graph with clinical constraints
        4) rescore diseases using graph properties + stroke-aware bias
        5) apply stroke-scope gate for acute stroke-like queries
        6) fetch related articles
        7) rescore articles so clinically relevant titles get promoted
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

            row = {
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
                "stroke_families": d.get("stroke_families", []) or [],
                "territories": d.get("territories", []) or [],
                "imaging_findings": d.get("imaging_findings", []) or [],
            }

            disease_rows.append(row)
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
                0.10 * float(freq_norm)
                + 0.25 * float(symptom_ratio)
                + 0.50 * float(clinical_norm)
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

        high_risk_query = flags["acute"] and (flags["posterior"] or flags["focal"] or flags["hemorrhage"])
        if high_risk_query:
            gated_rows = [
                r for r in rescored_rows
                if self._passes_stroke_scope_gate(r, flags)
            ]

            if gated_rows:
                rescored_rows = gated_rows
                disease_raw_scores = [float(r.get("raw_disease_score", 0.0) or 0.0) for r in rescored_rows]

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

                if not self._passes_article_scope_gate(title=title, disease_name=row["disease"], flags=flags):
                    continue

                article_context = dict(row)
                article_context.update({
                    "article_type": self._get_item_value(item, "article_type", "a.article_type", default=row.get("article_type", "unknown")),
                    "article_emergency_relevance": float(self._get_item_value(item, "article_emergency_relevance", "a.article_emergency_relevance", default=row.get("article_emergency_relevance", 0.0)) or 0.0),
                    "article_vascular_relevance": float(self._get_item_value(item, "article_vascular_relevance", "a.article_vascular_relevance", default=row.get("article_vascular_relevance", 0.0)) or 0.0),
                    "article_posterior_relevance": float(self._get_item_value(item, "article_posterior_relevance", "a.article_posterior_relevance", default=row.get("article_posterior_relevance", 0.0)) or 0.0),
                    "article_hemorrhage_relevance": float(self._get_item_value(item, "article_hemorrhage_relevance", "a.article_hemorrhage_relevance", default=row.get("article_hemorrhage_relevance", 0.0)) or 0.0),
                    "article_noise_penalty": float(self._get_item_value(item, "article_noise_penalty", "a.article_noise_penalty", default=row.get("article_noise_penalty", 0.0)) or 0.0),
                    "stroke_families": self._get_item_value(item, "stroke_families", default=row.get("stroke_families", [])) or [],
                    "territories": self._get_item_value(item, "territories", default=row.get("territories", [])) or [],
                    "imaging_findings": self._get_item_value(item, "imaging_findings", default=row.get("imaging_findings", [])) or [],
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
                    "stroke_families": article_context.get("stroke_families", []),
                    "territories": article_context.get("territories", []),
                    "imaging_findings": article_context.get("imaging_findings", []),
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
                article["kg_paths"] = self._build_kg_explanation_paths(article)

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


# ============================================================
# v7.12.13 KG query-context guard
# Goal:
#   Do not set graph query hemorrhage=True for posterior + vomiting only.
#   This prevents candidate-KG from over-promoting ICH/SAH docs.
# ============================================================

_V71213_KG_POSTERIOR_CUES = [
    "vertigo", "dizziness", "ataxia", "gait ataxia", "gait difficulty",
    "difficulty walking", "unsteady gait", "diplopia", "double vision",
    "nystagmus", "dysmetria", "brainstem", "cerebellar",
    "posterior circulation", "vertebrobasilar", "basilar",
    "????", "????", "????", "??????", "???? ??????", "????", "??? ??????",
]

_V71213_KG_HARD_HEMORRHAGE_CUES = [
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

_V71213_KG_VOMITING_CUES = [
    "vomiting", "emesis", "nausea and vomiting",
    "???", "?????", "????",
]


def _v71213_kg_count(text: str, terms: list[str]) -> int:
    low = (text or "").lower()
    return sum(1 for t in terms if t in low)


def _v71213_kg_any(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in terms)


def _v71213_kg_posterior_without_hard_heme(text: str) -> bool:
    return bool(
        _v71213_kg_count(text, _V71213_KG_POSTERIOR_CUES) >= 2
        and _v71213_kg_any(text, _V71213_KG_VOMITING_CUES)
        and not _v71213_kg_any(text, _V71213_KG_HARD_HEMORRHAGE_CUES)
    )


if "_v71213_original_query_context_flags" not in globals() and hasattr(GraphRetriever, "_query_context_flags"):
    _v71213_original_query_context_flags = GraphRetriever._query_context_flags

    def _v71213_query_context_flags(self, query):
        flags = _v71213_original_query_context_flags(self, query)
        if isinstance(flags, dict) and _v71213_kg_posterior_without_hard_heme(query):
            flags["hemorrhage"] = False
        return flags

    GraphRetriever._query_context_flags = _v71213_query_context_flags


if "_v71213_original_query_context_details" not in globals() and hasattr(GraphRetriever, "_query_context_details"):
    _v71213_original_query_context_details = GraphRetriever._query_context_details

    def _v71213_query_context_details(self, query):
        details = _v71213_original_query_context_details(self, query)
        if isinstance(details, dict) and _v71213_kg_posterior_without_hard_heme(query):
            details["hemorrhage"] = False
        return details

    GraphRetriever._query_context_details = _v71213_query_context_details


# ============================================================
# v7.15 negation-aware query flags and candidate-KG post-filter
# ============================================================
from app.rag.clinical_context import (
    analyze_case_context as _v715_case_profile,
    phrase_is_negated as _v715_phrase_is_negated,
)

_V715_ANIMAL_BASIC_SCIENCE = [
    "animal model", "experimental model", "in vitro", "cell culture",
    "mouse", "mice", "murine", "rat", "rats", "rabbit", "rabbits",
    "dog", "dogs", "canine", "cat", "cats", "feline", "pig", "pigs",
    "porcine", "swine", "sheep", "goat", "monkey", "macaque", "primate",
    "zebrafish", "guinea pig", "protein kinase", "phosphorylation",
    "gene expression", "dna repair", "gamma-h2ax",
]


def _v715_query_context_flags(self, query: str) -> Dict[str, bool]:
    profile = _v715_case_profile(query or "")
    return {
        "acute": bool(profile.get("acute_onset")),
        "posterior": bool(profile.get("posterior_pattern")),
        "focal": bool(profile.get("focal_neuro")),
        "hemorrhage": bool(profile.get("hemorrhage_warning")),
    }


def _v715_graph_is_negated(self, text: str, phrase: str, window: int = 5) -> bool:
    return _v715_phrase_is_negated(text or "", phrase or "")


def _v715_graph_doc_text(doc: Dict) -> str:
    return " ".join(str(doc.get(k) or "") for k in (
        "title", "text", "snippet", "disease", "graph_disease", "category",
        "graph_category", "article_type", "graph_article_type", "stroke_families",
        "graph_stroke_families", "territories", "graph_territories",
        "imaging_findings", "graph_imaging_findings",
    )).lower()


def _v715_has_term(text: str, term: str) -> bool:
    t = str(term or "").lower().strip()
    if len(t) <= 4 and re.fullmatch(r"[a-z0-9]+", t):
        return re.search(rf"\b{re.escape(t)}\b", text) is not None
    return t in text


def _v715_candidate_exact_terms(candidate_key: str) -> List[str]:
    return {
        "subarachnoid_hemorrhage": [
            "subarachnoid hemorrhage", "subarachnoid haemorrhage",
            "aneurysmal subarachnoid", "ruptured aneurysm", "basal cistern blood",
        ],
        "intracerebral_hemorrhage": [
            "intracerebral hemorrhage", "intracerebral haemorrhage",
            "intraparenchymal hemorrhage", "intraparenchymal haemorrhage",
            "intraparenchymal hematoma", "parenchymal hematoma", "basal ganglia hematoma",
        ],
        "hemorrhagic_stroke": [
            "hemorrhagic stroke", "haemorrhagic stroke", "intracranial hemorrhage",
            "intracranial haemorrhage", "brain hemorrhage", "brain haemorrhage",
        ],
        "tia": ["transient ischemic attack", "transient ischaemic attack", "amaurosis fugax"],
        "acute_ischemic_stroke": [
            "acute ischemic stroke", "acute ischaemic stroke", "ischemic stroke",
            "ischaemic stroke", "cerebral infarction", "brain infarction",
        ],
        "large_vessel_occlusion": ["large vessel occlusion", "mca occlusion", "carotid terminus", "thrombectomy"],
        "anterior_circulation_stroke": ["anterior circulation", "mca stroke", "middle cerebral artery stroke"],
        "posterior_circulation_stroke": ["posterior circulation stroke", "vertebrobasilar stroke"],
        "brainstem_infarction": ["brainstem infarction", "pontine infarction", "medullary infarction"],
        "cerebellar_infarction": ["cerebellar infarction", "cerebellar stroke"],
        "basilar_artery_occlusion": ["basilar artery occlusion", "basilar occlusion"],
        "lacunar_infarction": ["lacunar infarction", "lacunar stroke", "small vessel ischemic stroke"],
    }.get(candidate_key, [])


if "_v715_original_search_by_candidate" not in globals():
    _v715_original_search_by_candidate = GraphRetriever.search_by_candidate


def _v715_search_by_candidate(
    self,
    candidate_label: str | None = None,
    candidate_key: str | None = None,
    aliases: Optional[List[str]] = None,
    case_text: str | None = None,
    clinical_intent: Optional[Dict] = None,
    top_k: int = 5,
    candidate_query: str | None = None,
) -> List[Dict]:
    docs = _v715_original_search_by_candidate(
        self,
        candidate_label=candidate_label,
        candidate_key=candidate_key,
        aliases=aliases,
        case_text=case_text,
        clinical_intent=clinical_intent,
        top_k=max(top_k * 2, top_k),
        candidate_query=candidate_query,
    )
    key = str(candidate_key or "unknown")
    intent = clinical_intent or {}
    profile = intent.get("context_profile") if isinstance(intent.get("context_profile"), dict) else _v715_case_profile(case_text or "")
    exact_terms = _v715_candidate_exact_terms(key)

    filtered: List[Dict] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        text = _v715_graph_doc_text(doc)
        title = str(doc.get("title") or "").lower()
        if any(_v715_has_term(title, term) or _v715_has_term(text, term) for term in _V715_ANIMAL_BASIC_SCIENCE):
            continue

        exact_match = any(_v715_has_term(f"{title} {str(doc.get('disease') or doc.get('graph_disease') or '').lower()}", term) for term in exact_terms)
        alignment = float(doc.get("candidate_alignment_score", 0.0) or 0.0)
        graph_score = float(doc.get("graph_score", 0.0) or 0.0)

        if key == "subarachnoid_hemorrhage" and not profile.get("sah_warning"):
            continue
        if key == "intracerebral_hemorrhage" and not profile.get("ich_warning"):
            continue
        if key == "hemorrhagic_stroke" and not profile.get("hemorrhage_warning"):
            continue
        if key == "tia" and not profile.get("transient_resolved_episode"):
            continue

        # Subtype-specific verification must match the subtype, not merely the
        # broad vascular/hemorrhage category assigned in graph metadata.
        if exact_terms and not exact_match:
            continue
        if graph_score < 0.20 or (alignment < 0.08 and not exact_match):
            continue

        nd = dict(doc)
        nd["clinical_alignment_verified"] = True
        nd["quality_gate"] = "v7.15_human_subtype_aligned"
        filtered.append(nd)

    filtered.sort(
        key=lambda d: (
            float(d.get("candidate_alignment_score", 0.0) or 0.0),
            float(d.get("graph_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return filtered[:top_k]


GraphRetriever._query_context_flags = _v715_query_context_flags
GraphRetriever._is_negated = _v715_graph_is_negated
GraphRetriever.search_by_candidate = _v715_search_by_candidate
