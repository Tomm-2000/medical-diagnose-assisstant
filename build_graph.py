# -*- coding: utf-8 -*-
import json
from pathlib import Path
from py2neo import Graph, Node, Relationship
import re
from tqdm import tqdm
import scispacy
import spacy
import time

try:
    from app.rag.clinical_signals import detect_clinical_signals
except Exception:
    def detect_clinical_signals(text, use_model=False):
        return {}

# تحميل نموذج scispacy (يتطلب تثبيت النموذج مسبقاً)
nlp = spacy.load("en_ner_bc5cdr_md")


class MedicalKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.graph = Graph(uri, auth=(user, password))

        # =========================
        # Canonical vocab + synonyms
        # =========================
        self.negation_terms = {
            "no", "not", "without", "denies", "denied", "negative", "negative for",
            "free of", "absence of", "absent"
        }

        self.disease_synonyms = {
            "stroke": "stroke",
            "cva": "stroke",
            "brain attack": "stroke",
            "ischemic stroke": "ischemic stroke",
            "ischaemic stroke": "ischemic stroke",
            "ischemic": "ischemic stroke",
            "ischaemic": "ischemic stroke",
            "hemorrhagic stroke": "hemorrhagic stroke",
            "haemorrhagic stroke": "hemorrhagic stroke",
            "intracerebral hemorrhage": "intracerebral hemorrhage",
            "intracerebral haemorrhage": "intracerebral hemorrhage",
            "ich": "intracerebral hemorrhage",
            "subarachnoid hemorrhage": "subarachnoid hemorrhage",
            "subarachnoid haemorrhage": "subarachnoid hemorrhage",
            "sah": "subarachnoid hemorrhage",
            "tia": "transient ischemic attack",
            "transient ischemic attack": "transient ischemic attack",
            "transient ischaemic attack": "transient ischemic attack",
            "infarction": "cerebral infarction",
            "cerebral infarction": "cerebral infarction",
            "brain infarction": "cerebral infarction",

            # Specific posterior-circulation diagnoses.
            "posterior circulation stroke": "posterior circulation stroke",
            "posterior circulation ischemic stroke": "posterior circulation stroke",
            "posterior circulation ischaemic stroke": "posterior circulation stroke",
            "vertebrobasilar stroke": "posterior circulation stroke",

            "brainstem infarction": "brainstem infarction",
            "brainstem stroke": "brainstem infarction",

            "cerebellar infarction": "cerebellar infarction",
            "cerebellar stroke": "cerebellar infarction",

            "hypertension": "hypertension",
            "htn": "hypertension",
            "diabetes": "diabetes mellitus",
            "dm": "diabetes mellitus",
            "afib": "atrial fibrillation",
            "a fib": "atrial fibrillation",
            "atrial fibrillation": "atrial fibrillation",
            "hyperlipidemia": "hyperlipidemia",
            "hyperlipidaemia": "hyperlipidemia",
        }

        self.symptom_synonyms = {
            "weakness": "weakness",
            "unilateral weakness": "unilateral weakness",
            "arm weakness": "arm weakness",
            "leg weakness": "leg weakness",
            "hemiparesis": "hemiparesis",
            "hemiplegia": "hemiplegia",
            "aphasia": "aphasia",
            "dysarthria": "dysarthria",
            "slurred speech": "dysarthria",
            "speech difficulty": "speech disturbance",
            "speech disturbance": "speech disturbance",
            "vertigo": "vertigo",
            "diplopia": "diplopia",
            "double vision": "diplopia",
            "headache": "headache",
            "vomiting": "vomiting",
            "numbness": "numbness",
            "ataxia": "ataxia",
            "facial droop": "facial droop",
            "difficulty walking": "gait difficulty",
            "gait difficulty": "gait difficulty",
            "dizziness": "dizziness",
            "loss of consciousness": "loss of consciousness",
            "confusion": "confusion",
        }

        self.drug_synonyms = {
            "aspirin": "aspirin",
            "clopidogrel": "clopidogrel",
            "warfarin": "warfarin",
            "rivaroxaban": "rivaroxaban",
            "alteplase": "alteplase",
            "tpa": "alteplase",
            "rt-pa": "alteplase",
        }

        self.risk_factor_synonyms = {
            "smoking": "smoking",
            "smoker": "smoking",
            "obesity": "obesity",
            "age": "advanced age",
            "old age": "advanced age",
            "sedentary": "sedentary lifestyle",
            "sedentary lifestyle": "sedentary lifestyle",
            "hypertension": "hypertension",
            "diabetes": "diabetes mellitus",
            "afib": "atrial fibrillation",
            "a fib": "atrial fibrillation",
            "atrial fibrillation": "atrial fibrillation",
            "diabetes mellitus": "diabetes mellitus",
            "hyperlipidemia": "hyperlipidemia",
            "hyperlipidaemia": "hyperlipidemia",
            "htn": "hypertension",
        }

        # These concepts are medically valid diseases, but in this
        # stroke decision-support KG they must not be treated as
        # competing neurological diagnoses. They are stored as
        # risk factors / comorbidities instead.
        self.non_diagnostic_risk_concepts = {
            "hypertension",
            "diabetes mellitus",
            "atrial fibrillation",
            "hyperlipidemia",
        }

        # =========================
        # Clinical classification vocab
        # =========================

        self.vascular_terms = [
            "stroke",
            "ischemic stroke",
            "ischaemic stroke",
            "ischemia",
            "ischaemia",
            "ischemic",
            "ischaemic",
            "infarction",
            "infarct",
            "cerebral infarction",
            "brain infarction",
            "transient ischemic attack",
            "transient ischaemic attack",
            "tia",
            "cerebrovascular",
            "arterial occlusion",
            "large vessel occlusion",
            "thrombosis",
            "embolism",
        ]

        self.posterior_vascular_terms = [
            "posterior circulation stroke",
            "posterior circulation ischemic stroke",
            "posterior circulation ischaemic stroke",
            "vertebrobasilar ischemia",
            "vertebrobasilar ischaemia",
            "vertebrobasilar insufficiency",
            "vertebrobasilar disease",
            "basilar artery",
            "brainstem infarction",
            "brainstem stroke",
            "midbrain infarction",
            "pontine infarction",
            "pons infarction",
            "medullary infarction",
            "lateral medullary infarction",
            "wallenberg",
            "cerebellar infarction",
            "cerebellar stroke",
            "anterior inferior cerebellar artery",
            "posterior inferior cerebellar artery",
            "superior cerebellar artery",
            "aica infarction",
            "pica infarction",
            "sca infarction",
        ]

        self.posterior_anatomy_terms = [
            "posterior circulation",
            "vertebrobasilar",
            "brainstem",
            "midbrain",
            "pontine",
            "pons",
            "medullary",
            "lateral medullary",
            "cerebellar",
            "cerebellum",
            "basilar artery",
            "aica",
            "pica",
            "superior cerebellar artery",
        ]

        self.hemorrhage_terms = [
            "hemorrhage",
            "haemorrhage",
            "intracerebral hemorrhage",
            "intracerebral haemorrhage",
            "subarachnoid hemorrhage",
            "subarachnoid haemorrhage",
            "hemorrhagic stroke",
            "haemorrhagic stroke",
            "ich",
            "sah",
            "aneurysmal rupture",
            "ruptured aneurysm",
        ]

        self.peripheral_vestibular_terms = [
            "bppv",
            "benign paroxysmal positional vertigo",
            "benign positional vertigo",
            "benign positioning vertigo",
            "meniere",
            "ménière",
            "vestibular neuritis",
            "vestibular hypofunction",
            "vestibular rehabilitation",
            "labyrinthitis",
            "peripheral vertigo",
            "intralabyrinthine schwannoma",
            "acoustic neuroma",
            "vestibular schwannoma",
        ]

        self.genetic_chronic_neuro_terms = [
            "spinocerebellar ataxia",
            "ataxia telangiectasia",
            "friedreich",
            "hereditary",
            "autosomal",
            "genetic",
            "mutation",
            "syndrome",
            "cerebellar ataxia",
            "olivopontocerebellar atrophy",
            "machado-joseph",
            "sca1",
            "sca2",
            "sca3",
            "sca6",
            "familial",
            "degeneration",
            "degenerative",
        ]

        self.mimic_terms = [
            "migraine",
            "migrainous vertigo",
            "seizure",
            "epilepsy",
            "functional dizziness",
            "conversion",
            "panic disorder",
            "psychiatric",
        ]

        self.infectious_inflammatory_terms = [
            "meningitis",
            "encephalitis",
            "multiple sclerosis",
            "demyelinating",
            "inflammation",
            "infectious",
            "abscess",
            "neurosyphilis",
            "tuberculosis",
            "vasculitis",
        ]

        self.tumor_terms = [
            "tumor",
            "tumour",
            "metastasis",
            "cancer",
            "glioma",
            "meningioma",
            "schwannoma",
            "neoplasm",
            "mass lesion",
        ]

        self.basic_science_noise_terms = [
            "cell",
            "cells",
            "protein",
            "phosphorylation",
            "mouse",
            "rat",
            "mice",
            "gene",
            "kinase",
            "receptor",
            "in vitro",
            "culture",
            "molecular",
            "dna",
            "rna",
            "pathway",
            "enzyme",
            "knockout",
        ]

        self.animal_model_noise_terms = [
            "rat model",
            "mouse model",
            "mice model",
            "animal model",
            "experimental model",
            "murine",
            "rodent",
            "rats",
            "mice",
            "rabbit model",
            "canine model",
            "swine model",
            "primate model",
        ]

        self.molecular_noise_terms = [
            "messenger rna",
            "mrna",
            "upregulation",
            "downregulation",
            "gene expression",
            "protein expression",
            "signal transduction",
            "receptor expression",
            "enzyme activity",
            "kinase activity",
            "western blot",
            "polymerase chain reaction",
            "pcr",
            "immunohistochemistry",
            "oxidative stress",
            "apoptosis",
            "inflammatory mediator",
        ]

        self.rehabilitation_noise_terms = [
            "rehabilitation",
            "training",
            "exercise",
            "therapy improves",
            "vestibular rehabilitation",
            "balance training",
        ]

        self.human_clinical_terms = [
            "case report",
            "case series",
            "clinical",
            "patient",
            "patients",
            "presented with",
            "admitted",
            "emergency",
            "acute onset",
            "sudden",
            "diagnosis",
            "treatment",
            "outcome",
            "outcomes",
            "computed tomography",
            "ct",
            "magnetic resonance imaging",
            "mri",
            "angiography",
            "randomized",
            "trial",
            "cohort",
            "prospective",
            "retrospective",
        ]

        self.clinical_case_terms = self.human_clinical_terms

    def _normalize_text(self, text):
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _canonicalize_term(self, term, mapping):
        term_norm = self._normalize_text(term)
        return mapping.get(term_norm, term_norm)

    def _compile_term_pattern(self, term):
        """
        Build a safe whole-term regular expression.

        This prevents abbreviations such as:
        - TIA from matching "initial"
        - ICH from matching "which"
        - DM from matching "admission"
        """
        term_norm = self._normalize_text(term)

        if not term_norm:
            return None

        pattern_body = re.escape(term_norm)

        # Permit flexible whitespace inside multi-word terms.
        pattern_body = pattern_body.replace(
            r"\ ",
            r"\s+"
        )

        return re.compile(
            r"(?<!\w)"
            + pattern_body
            + r"(?!\w)",
            flags=re.IGNORECASE
        )

    def _term_occurs(self, text, term):
        """
        Return True only when the complete term occurs.

        Unlike: term in text
        this method does not match abbreviations inside larger words.
        """
        text_norm = self._normalize_text(text)
        pattern = self._compile_term_pattern(term)

        if not text_norm or pattern is None:
            return False

        return bool(pattern.search(text_norm))

    def _is_contextual_cva_stroke(self, text):
        """
        Disambiguate the acronym CVA.

        CVA is accepted as cerebrovascular accident only when:
        1. It is written as uppercase CVA.
        2. Neurological or cerebrovascular context is present.
        3. No strong competing scientific meaning is present.

        This deliberately favors precision over recall.
        """
        raw_text = str(text or "")

        cva_matches = list(
            re.finditer(
                r"(?<!\w)CVA(?!\w)",
                raw_text
            )
        )

        # Mixed-case cVA is commonly used for non-stroke meanings,
        # such as the Drosophila pheromone.
        if not cva_matches:
            return False

        exclusion_terms = [
            "cva/ampc",
            "clavulanic acid",
            "clavulanate",
            "amoxicillin",
            "cell viability assay",
            "drosophila",
            "pheromone",
            "craniovertebral angle",
            "cranio-vertebral angle",
            "crypt-villus axis",
            "coefficient of additive genetic variance",
            "sevoflurane",
        ]

        if any(
            self._term_occurs(raw_text, term)
            for term in exclusion_terms
        ):
            return False

        neurological_context_terms = [
            "stroke",
            "cerebrovascular",
            "brain attack",
            "ischemic",
            "ischaemic",
            "infarct",
            "infarction",
            "hemiplegia",
            "hemiparesis",
            "aphasia",
            "dysarthria",
            "facial weakness",
            "neurological deficit",
            "cerebral",
            "thrombolysis",
            "thrombectomy",
        ]

        # Check a local context window around every CVA occurrence.
        for match in cva_matches:
            start = max(
                0,
                match.start() - 250
            )

            end = min(
                len(raw_text),
                match.end() + 250
            )

            context_window = raw_text[start:end]

            if any(
                self._term_occurs(
                    context_window,
                    term
                )
                for term
                in neurological_context_terms
            ):
                return True

        return False

    def _count_term_hits(self, text, terms):
        """
        Count complete term occurrences at the vocabulary level.

        This replaces unsafe substring matching.
        """
        return sum(
            1
            for term in terms
            if self._term_occurs(text, term)
        )

    def _infer_posterior_diagnoses(self, text):
        """
        Extract only explicit posterior-circulation diagnoses.

        Basilar artery occlusion and vertebrobasilar occlusion are
        vascular findings and must never be converted automatically
        into posterior circulation stroke.
        """
        text = str(text or "")
        normalized_text = self._normalize_text(text)

        explicit_diagnoses = {
            "posterior circulation stroke":
                "posterior circulation stroke",

            "posterior circulation ischemic stroke":
                "posterior circulation stroke",

            "posterior circulation ischaemic stroke":
                "posterior circulation stroke",

            "vertebrobasilar stroke":
                "posterior circulation stroke",

            "brainstem infarction":
                "brainstem infarction",

            "brainstem stroke":
                "brainstem infarction",

            "cerebellar infarction":
                "cerebellar infarction",

            "cerebellar stroke":
                "cerebellar infarction",
        }

        diagnoses = []

        for phrase, canonical in explicit_diagnoses.items():
            if (
                self._term_occurs(
                    text,
                    phrase
                )
                and not self._is_negated(
                    normalized_text,
                    phrase
                )
            ):
                diagnoses.append(
                    canonical
                )

        return list(
            dict.fromkeys(diagnoses)
        )

    def extract_vascular_findings(self, text):
        """
        Extract vascular findings independently from diseases.

        This relation represents that an article mentions the finding;
        it does not claim that the finding is the final diagnosis or
        that the article supports a diagnostic conclusion.

        Excluded, therapeutic, animal, and clinical mentions remain
        vascular-finding mentions and are distinguished later using
        article metadata and evidence context.
        """
        text = str(text or "")

        vascular_synonyms = {
            "basilar artery occlusion":
                "basilar artery occlusion",

            "basilar occlusion":
                "basilar artery occlusion",

            "vertebrobasilar occlusion":
                "vertebrobasilar occlusion",
        }

        findings = []

        for phrase, canonical in vascular_synonyms.items():
            if self._term_occurs(
                text,
                phrase
            ):
                findings.append(
                    canonical
                )

        return list(
            dict.fromkeys(findings)
        )

    def _apply_taxonomy_policy(self, entities):
        """
        Enforce the stroke-KG taxonomy after any extraction method.

        Some medically valid diseases, such as hypertension and
        diabetes mellitus, are risk factors or comorbidities in this
        diagnostic system. They must not become competing neurological
        diagnosis candidates.

        This method also handles entities returned by SciSpacy.
        """
        entities = dict(entities or {})

        diseases = list(
            entities.get("diseases", []) or []
        )

        risk_factors = list(
            entities.get("risk_factors", []) or []
        )

        configured_risk_concepts = getattr(
            self,
            "non_diagnostic_risk_concepts",
            set()
        ) or set()

        risk_concepts = {
            self._normalize_text(concept)
            for concept in configured_risk_concepts
            if self._normalize_text(concept)
        }

        diagnostic_diseases = []

        for disease in diseases:
            disease_normalized = self._normalize_text(
                disease
            )

            if disease_normalized in risk_concepts:
                risk_factors.append(disease)
            else:
                diagnostic_diseases.append(disease)

        entities["diseases"] = list(
            dict.fromkeys(diagnostic_diseases)
        )

        entities["risk_factors"] = list(
            dict.fromkeys(risk_factors)
        )

        entities.setdefault("symptoms", [])
        entities.setdefault("drugs", [])
        entities.setdefault("vascular_findings", [])

        return entities

    def _is_negated(self, text, phrase, window=5):
        """
        فحص بسيط: إذا وجدنا أداة نفي قبل العبارة ضمن نافذة كلمات قصيرة،
        نعتبر العبارة منفية ولا نضيفها.
        """
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

    def classify_disease(self, disease_name):
        """
        تصنيف سريري عام للمرض قبل تخزينه في Neo4j.
        """
        name = self._normalize_text(disease_name)

        result = {
            "category": "unknown",
            "acuity": "unknown",
            "vascular_relevance": 0.0,
            "posterior_relevance": 0.0,
            "hemorrhage_relevance": 0.0,
            "emergency_relevance": 0.0,
            "chronic_penalty": 0.0,
            "noise_penalty": 0.0,
        }

        vascular_hits = self._count_term_hits(name, self.vascular_terms)
        posterior_vascular_hits = self._count_term_hits(name, self.posterior_vascular_terms)
        posterior_anatomy_hits = self._count_term_hits(name, self.posterior_anatomy_terms)
        hemorrhage_hits = self._count_term_hits(name, self.hemorrhage_terms)
        peripheral_hits = self._count_term_hits(name, self.peripheral_vestibular_terms)
        genetic_hits = self._count_term_hits(name, self.genetic_chronic_neuro_terms)
        mimic_hits = self._count_term_hits(name, self.mimic_terms)
        infectious_hits = self._count_term_hits(name, self.infectious_inflammatory_terms)
        tumor_hits = self._count_term_hits(name, self.tumor_terms)
        noise_hits = self._count_term_hits(name, self.basic_science_noise_terms)
        rehab_hits = self._count_term_hits(name, self.rehabilitation_noise_terms)

        # Clinical signal layer: complements the old term-hit rules.
        # Model use stays disabled during KG build for speed/reproducibility;
        # the rule layer remains the deterministic fallback.
        try:
            signals = detect_clinical_signals(name, use_model=False)
        except Exception:
            signals = {}

        if signals.get("is_hemorrhage") and hemorrhage_hits == 0:
            hemorrhage_hits = 1
        if signals.get("is_posterior") and posterior_vascular_hits == 0 and posterior_anatomy_hits == 0:
            posterior_anatomy_hits = 1
        if signals.get("is_chronic_or_general"):
            result["chronic_penalty"] = max(result["chronic_penalty"], 0.45)
        if signals.get("is_mimic_or_nonstroke"):
            mimic_hits = max(mimic_hits, 1)

        if hemorrhage_hits > 0:
            result.update({
                "category": "hemorrhagic",
                "acuity": "acute",
                "hemorrhage_relevance": 1.0,
                "vascular_relevance": max(result["vascular_relevance"], 0.8),
                "emergency_relevance": 1.0,
                "chronic_penalty": 0.0,
            })

        if posterior_vascular_hits > 0:
            result.update({
                "category": "posterior_vascular",
                "acuity": "acute",
                "vascular_relevance": 1.0,
                "posterior_relevance": 1.0,
                "emergency_relevance": 1.0,
                "chronic_penalty": 0.0,
            })

        if vascular_hits > 0 and posterior_vascular_hits == 0 and hemorrhage_hits == 0:
            result.update({
                "category": "vascular",
                "acuity": "acute",
                "vascular_relevance": max(result["vascular_relevance"], 1.0),
                "emergency_relevance": max(result["emergency_relevance"], 1.0),
                "chronic_penalty": 0.0,
            })

        if posterior_anatomy_hits > 0 and posterior_vascular_hits == 0:
            result["posterior_relevance"] = max(result["posterior_relevance"], 0.35)
            result["emergency_relevance"] = max(result["emergency_relevance"], 0.20)

        if peripheral_hits > 0:
            result.update({
                "category": "peripheral_vestibular",
                "acuity": "episodic",
                "vascular_relevance": min(result["vascular_relevance"], 0.2),
                "posterior_relevance": max(result["posterior_relevance"], 0.2),
                "emergency_relevance": max(result["emergency_relevance"], 0.2),
                "chronic_penalty": max(result["chronic_penalty"], 0.4),
            })

        if genetic_hits > 0:
            result.update({
                "category": "genetic_chronic_neuro",
                "acuity": "chronic",
                "vascular_relevance": min(result["vascular_relevance"], 0.1),
                "posterior_relevance": min(max(result["posterior_relevance"], 0.15), 0.25),
                "emergency_relevance": min(max(result["emergency_relevance"], 0.1), 0.2),
                "chronic_penalty": max(result["chronic_penalty"], 1.0),
            })

        if mimic_hits > 0:
            result.update({
                "category": "migraine_seizure_mimic",
                "acuity": "episodic",
                "vascular_relevance": min(result["vascular_relevance"], 0.2),
                "emergency_relevance": max(result["emergency_relevance"], 0.3),
                "chronic_penalty": max(result["chronic_penalty"], 0.3),
            })

        if infectious_hits > 0:
            result.update({
                "category": "inflammatory_infectious",
                "acuity": "subacute",
                "emergency_relevance": max(result["emergency_relevance"], 0.5),
                "chronic_penalty": max(result["chronic_penalty"], 0.2),
            })

        if tumor_hits > 0:
            result.update({
                "category": "tumor",
                "acuity": "subacute",
                "emergency_relevance": max(result["emergency_relevance"], 0.3),
                "chronic_penalty": max(result["chronic_penalty"], 0.5),
            })

        if noise_hits > 0:
            result["noise_penalty"] = max(result["noise_penalty"], 0.8)
            result["chronic_penalty"] = max(result["chronic_penalty"], 0.5)

        if rehab_hits > 0:
            result["noise_penalty"] = max(result["noise_penalty"], 0.5)
            result["chronic_penalty"] = max(result["chronic_penalty"], 0.4)

        return result

    def classify_article(self, title, text):
        """
        تصنيف سريري للمقال/chunk نفسه.

        الهدف:
        - رفع المقالات السريرية المتعلقة بـ stroke / hemorrhage / posterior circulation.
        - خفض مقالات basic science / animal models / molecular studies حتى لو ذكرت stroke أو hemorrhage.
        - جعل التصنيف قابل للتعميم على حالات السكتة، النزف، المقلدات، الاضطرابات الدهليزية، والحالات المزمنة.
        """
        title_low = self._normalize_text(title)
        full = self._normalize_text((title or "") + " " + (text or ""))

        result = {
            "article_type": "unknown",
            "article_emergency_relevance": 0.0,
            "article_vascular_relevance": 0.0,
            "article_posterior_relevance": 0.0,
            "article_hemorrhage_relevance": 0.0,
            "article_noise_penalty": 0.0,
        }

        vascular_hits = self._count_term_hits(full, self.vascular_terms)
        posterior_vascular_hits = self._count_term_hits(full, self.posterior_vascular_terms)
        hemorrhage_hits = self._count_term_hits(full, self.hemorrhage_terms)
        peripheral_hits = self._count_term_hits(full, self.peripheral_vestibular_terms)
        genetic_hits = self._count_term_hits(full, self.genetic_chronic_neuro_terms)
        mimic_hits = self._count_term_hits(full, self.mimic_terms)

        basic_noise_hits = self._count_term_hits(full, self.basic_science_noise_terms)
        animal_noise_hits = self._count_term_hits(full, self.animal_model_noise_terms)
        molecular_noise_hits = self._count_term_hits(full, self.molecular_noise_terms)
        rehab_hits = self._count_term_hits(full, self.rehabilitation_noise_terms)
        human_clinical_hits = self._count_term_hits(full, self.human_clinical_terms)

        title_vascular_hits = self._count_term_hits(title_low, self.vascular_terms)
        title_posterior_hits = self._count_term_hits(title_low, self.posterior_vascular_terms)
        title_hemorrhage_hits = self._count_term_hits(title_low, self.hemorrhage_terms)

        # Clinical signal layer over the article/chunk text.
        # This is not a replacement for the curated rules; it adds a centralized
        # signal detector that can later be backed by a transformer model.
        try:
            signals = detect_clinical_signals(full, use_model=False)
        except Exception:
            signals = {}

        if signals.get("is_hemorrhage") and hemorrhage_hits == 0:
            hemorrhage_hits = 1
        if signals.get("is_posterior") and posterior_vascular_hits == 0:
            posterior_vascular_hits = 1
        if signals.get("is_chronic_or_general"):
            rehab_hits = max(rehab_hits, 1)
        if signals.get("is_mimic_or_nonstroke"):
            mimic_hits = max(mimic_hits, 1)

        strong_basic_noise = (
            animal_noise_hits > 0
            or molecular_noise_hits >= 2
            or basic_noise_hits >= 3
        )

        moderate_basic_noise = (
            basic_noise_hits > 0
            or molecular_noise_hits > 0
            or animal_noise_hits > 0
        )

        has_human_clinical_context = human_clinical_hits > 0

        # =========================
        # Clinical vascular / hemorrhage relevance
        # =========================
        if hemorrhage_hits > 0:
            result.update({
                "article_type": "hemorrhage_case_or_review",
                "article_emergency_relevance": 1.0,
                "article_vascular_relevance": 0.8,
                "article_hemorrhage_relevance": 1.0,
            })

        if posterior_vascular_hits > 0:
            result.update({
                "article_type": "posterior_stroke_case_or_review",
                "article_emergency_relevance": 1.0,
                "article_vascular_relevance": 1.0,
                "article_posterior_relevance": 1.0,
            })

        elif vascular_hits > 0:
            result.update({
                "article_type": "stroke_case_or_review",
                "article_emergency_relevance": max(result["article_emergency_relevance"], 0.9),
                "article_vascular_relevance": max(result["article_vascular_relevance"], 1.0),
            })

        # =========================
        # Non-stroke / chronic / mimic classes
        # =========================
        if peripheral_hits > 0 and vascular_hits == 0 and hemorrhage_hits == 0:
            result.update({
                "article_type": "vestibular_disorder",
                "article_emergency_relevance": max(result["article_emergency_relevance"], 0.2),
                "article_noise_penalty": max(result["article_noise_penalty"], 0.2),
            })

        if genetic_hits > 0 and vascular_hits == 0 and hemorrhage_hits == 0:
            result.update({
                "article_type": "genetic_or_chronic_ataxia",
                "article_emergency_relevance": min(max(result["article_emergency_relevance"], 0.1), 0.2),
                "article_noise_penalty": max(result["article_noise_penalty"], 0.7),
            })

        if mimic_hits > 0 and vascular_hits == 0 and hemorrhage_hits == 0:
            result.update({
                "article_type": "stroke_mimic_or_nonvascular",
                "article_emergency_relevance": max(result["article_emergency_relevance"], 0.3),
                "article_noise_penalty": max(result["article_noise_penalty"], 0.3),
            })

        if rehab_hits > 0 and vascular_hits == 0:
            result.update({
                "article_type": "rehabilitation_or_training",
                "article_noise_penalty": max(result["article_noise_penalty"], 0.6),
                "article_emergency_relevance": min(result["article_emergency_relevance"], 0.3),
            })

        # =========================
        # Generalizable noise handling
        # =========================

        # أي basic science noise يجب أن يخفض المقال، حتى لو فيه hemorrhage/stroke.
        if moderate_basic_noise:
            result["article_noise_penalty"] = max(result["article_noise_penalty"], 0.45)

        # animal/molecular studies ليست دليلاً سريرياً مباشرًا لحالة إسعافية.
        if strong_basic_noise:
            result["article_type"] = "basic_science"
            result["article_noise_penalty"] = max(result["article_noise_penalty"], 0.95)

            result["article_emergency_relevance"] = min(result["article_emergency_relevance"], 0.25)
            result["article_vascular_relevance"] = min(result["article_vascular_relevance"], 0.35)
            result["article_posterior_relevance"] = min(result["article_posterior_relevance"], 0.35)
            result["article_hemorrhage_relevance"] = min(result["article_hemorrhage_relevance"], 0.45)

        # إذا المقال noise بسيط لكنه واضح سريرياً، لا نسحقه تماماً.
        if moderate_basic_noise and has_human_clinical_context and not strong_basic_noise:
            result["article_noise_penalty"] = min(result["article_noise_penalty"], 0.45)

        # إذا لا يوجد سياق سريري بشري، والعلم الأساسي واضح، خفض أقوى.
        if moderate_basic_noise and not has_human_clinical_context:
            result["article_type"] = "basic_science"
            result["article_noise_penalty"] = max(result["article_noise_penalty"], 0.85)
            result["article_emergency_relevance"] = min(result["article_emergency_relevance"], 0.25)

        # =========================
        # Title-level override
        # العنوان أقوى من النص، لكن لا يتجاوز strong basic science.
        # =========================
        if title_posterior_hits > 0 and not strong_basic_noise:
            result.update({
                "article_type": "posterior_stroke_case_or_review",
                "article_emergency_relevance": 1.0,
                "article_vascular_relevance": 1.0,
                "article_posterior_relevance": 1.0,
                "article_noise_penalty": min(result["article_noise_penalty"], 0.1),
            })

        if title_hemorrhage_hits > 0 and not strong_basic_noise:
            result.update({
                "article_type": "hemorrhage_case_or_review",
                "article_emergency_relevance": 1.0,
                "article_vascular_relevance": max(result["article_vascular_relevance"], 0.8),
                "article_hemorrhage_relevance": 1.0,
                "article_noise_penalty": min(result["article_noise_penalty"], 0.1),
            })

        if title_vascular_hits > 0 and not strong_basic_noise:
            result["article_emergency_relevance"] = max(result["article_emergency_relevance"], 0.9)
            result["article_vascular_relevance"] = max(result["article_vascular_relevance"], 1.0)

        return result


    # ============================================================
    # v7.12.4 ontology KG relations
    # ============================================================

    def _extract_chunk_order(self, chunk_id):
        """
        Extract the numeric order from identifiers such as:
        pmid_123_chunk_1

        Unknown formats are placed after ordered chunks.
        """
        chunk_id_text = str(chunk_id or "")

        match = re.search(
            r"chunk[_-]?(\d+)$",
            chunk_id_text,
            flags=re.IGNORECASE
        )

        if match:
            return int(match.group(1))

        return 10 ** 9

    def _longest_exact_chunk_overlap(
        self,
        previous_text,
        current_text,
        maximum=500
    ):
        """
        Return the longest exact suffix-prefix overlap.

        Example:
            previous = "Alpha beta gamma delta"
            current  = "gamma delta epsilon"

        overlap:
            "gamma delta"
        """
        previous = str(previous_text or "")
        current = str(current_text or "")

        maximum = min(
            len(previous),
            len(current),
            int(maximum)
        )

        for length in range(maximum, 0, -1):
            if previous[-length:] == current[:length]:
                return length

        return 0

    def _reconstruct_article_chunks(self, chunks):
        """
        Reconstruct one article/abstract from its ordered chunks while
        removing the exact overlap between adjacent chunks.
        """
        ordered_chunks = sorted(
            list(chunks or []),
            key=lambda row: (
                self._extract_chunk_order(
                    row.get("chunk_id")
                ),
                str(row.get("chunk_id") or "")
            )
        )

        if not ordered_chunks:
            return "", []

        reconstructed = str(
            ordered_chunks[0].get("text") or ""
        )

        ordered_chunk_ids = [
            str(
                ordered_chunks[0].get(
                    "chunk_id"
                ) or ""
            )
        ]

        for row in ordered_chunks[1:]:
            current_text = str(
                row.get("text") or ""
            )

            ordered_chunk_ids.append(
                str(row.get("chunk_id") or "")
            )

            overlap = (
                self._longest_exact_chunk_overlap(
                    reconstructed,
                    current_text,
                    maximum=500
                )
            )

            if overlap > 0:
                reconstructed += current_text[
                    overlap:
                ]
            else:
                # Preserve readable separation when no exact overlap
                # can be detected.
                if (
                    reconstructed
                    and current_text
                    and not reconstructed[-1].isspace()
                    and not current_text[0].isspace()
                ):
                    reconstructed += " "

                reconstructed += current_text

        return reconstructed, ordered_chunk_ids

    def _prepare_article_level_records(self, rows):
        """
        Add article-level text and metadata to every chunk record.

        Chunks sharing the same PMID receive the same:
        - article_text
        - article_full_text
        - article_chunk_count
        - article_chunk_ids

        Missing PMIDs are intentionally treated as independent articles
        using their chunk_id, so unrelated records are never mixed.
        """
        source_rows = [
            dict(row or {})
            for row in (rows or [])
        ]

        grouped_rows = {}

        for position, row in enumerate(source_rows):
            pmid = str(
                row.get("pmid") or ""
            ).strip()

            chunk_id = str(
                row.get("chunk_id")
                or f"missing_chunk_{position}"
            )

            if pmid:
                group_key = (
                    "pmid",
                    pmid
                )
            else:
                group_key = (
                    "chunk",
                    chunk_id,
                    position
                )

            grouped_rows.setdefault(
                group_key,
                []
            ).append(row)

        article_metadata = {}

        for group_key, group in grouped_rows.items():
            article_text, article_chunk_ids = (
                self._reconstruct_article_chunks(
                    group
                )
            )

            ordered_group = sorted(
                group,
                key=lambda row: (
                    self._extract_chunk_order(
                        row.get("chunk_id")
                    ),
                    str(row.get("chunk_id") or "")
                )
            )

            article_title = next(
                (
                    str(row.get("title") or "").strip()
                    for row in ordered_group
                    if str(
                        row.get("title") or ""
                    ).strip()
                ),
                ""
            )

            if article_title:
                article_full_text = (
                    article_title
                    + "\n"
                    + article_text
                )
            else:
                article_full_text = article_text

            article_metadata[group_key] = {
                "article_title":
                    article_title,

                "article_text":
                    article_text,

                "article_full_text":
                    article_full_text,

                "article_chunk_count":
                    len(group),

                "article_chunk_ids":
                    article_chunk_ids,
            }

        prepared_rows = []

        for position, row in enumerate(source_rows):
            pmid = str(
                row.get("pmid") or ""
            ).strip()

            chunk_id = str(
                row.get("chunk_id")
                or f"missing_chunk_{position}"
            )

            if pmid:
                group_key = (
                    "pmid",
                    pmid
                )
            else:
                group_key = (
                    "chunk",
                    chunk_id,
                    position
                )

            prepared_row = dict(row)

            prepared_row.update(
                article_metadata[group_key]
            )

            prepared_rows.append(
                prepared_row
            )

        return prepared_rows

    def infer_ontology_links(self, disease_name, title="", text="", disease_props=None, article_props=None):
        """
        Infer stable ontology-style links from a disease/article context.

        These links are not used to create a diagnosis from a patient case.
        They are built offline from article evidence to make Neo4j more clinically
        structured than generic DISCUSSES / MENTIONS edges.
        """
        disease_props = disease_props or self.classify_disease(disease_name)
        article_props = article_props or self.classify_article(title, text)
        full = self._normalize_text(" ".join([str(disease_name or ""), str(title or ""), str(text or "")]))

        families = set()
        territories = set()
        imaging_findings = set()

        category = str(disease_props.get("category") or "").lower()
        disease_low = self._normalize_text(disease_name)

        if category == "hemorrhagic" or self._count_term_hits(full, self.hemorrhage_terms) > 0:
            families.add("hemorrhagic stroke")
        if category in {"vascular", "posterior_vascular"} or self._count_term_hits(full, self.vascular_terms) > 0:
            # Keep TIA separate from tissue infarction.
            if "transient ischemic attack" in full or re.search(r"\btia\b", full):
                families.add("transient ischemic attack")
            elif self._count_term_hits(full, ["ischemic", "ischaemic", "infarction", "infarct", "ischemia", "ischaemia"]) > 0:
                families.add("ischemic stroke")
            else:
                families.add("stroke")

        if category == "posterior_vascular" or self._count_term_hits(full, self.posterior_anatomy_terms + self.posterior_vascular_terms) > 0:
            families.add("posterior circulation stroke")
            territories.add("posterior circulation")
        if any(t in full for t in ["vertebrobasilar", "basilar artery", "basilar"]):
            territories.add("vertebrobasilar")
        if any(t in full for t in ["brainstem", "midbrain", "pontine", "pons", "medullary", "lateral medullary"]):
            territories.add("brainstem")
        if any(t in full for t in ["cerebellar", "cerebellum", "pica", "aica", "superior cerebellar artery"]):
            territories.add("cerebellum")
        if any(t in full for t in ["middle cerebral artery", " mca ", "anterior circulation", "carotid", "cortical aphasia"]):
            territories.add("anterior circulation")

        if any(t in full for t in ["intracerebral hemorrhage", "intracerebral haemorrhage", "intraparenchymal hemorrhage", "intraparenchymal haemorrhage", "intraparenchymal hematoma", "parenchymal hematoma", "basal ganglia hemorrhage", "putaminal hemorrhage", "thalamic hemorrhage"]):
            families.add("intracerebral hemorrhage")
            imaging_findings.add("intraparenchymal hematoma")
        if any(t in full for t in ["subarachnoid hemorrhage", "subarachnoid haemorrhage", "aneurysmal subarachnoid", "ruptured aneurysm", "basal cistern blood", "convexal subarachnoid"]):
            families.add("subarachnoid hemorrhage")
            imaging_findings.add("subarachnoid blood")
        if any(t in full for t in ["dwi lesion", "diffusion restriction", "restricted diffusion", "infarct on mri", "mri confirmed infarct", "ct confirmed acute infarct", "imaging-confirmed infarction"]):
            imaging_findings.add("imaging-confirmed infarction")
        if any(t in full for t in ["large vessel occlusion", "arterial occlusion", "basilar artery occlusion", "mca occlusion", "middle cerebral artery occlusion"]):
            imaging_findings.add("arterial occlusion")

        # Article-level labels can contribute when disease names are too sparse.
        if article_props.get("article_posterior_relevance", 0.0) >= 0.8:
            territories.add("posterior circulation")
        if article_props.get("article_hemorrhage_relevance", 0.0) >= 0.8:
            families.add("hemorrhagic stroke")
        if article_props.get("article_vascular_relevance", 0.0) >= 0.8 and not families:
            families.add("stroke")

        return {
            "families": sorted(families),
            "territories": sorted(territories),
            "imaging_findings": sorted(imaging_findings),
        }

    def _merge_ontology_links(self, tx, article_node, disease_node, disease, entities, title, text, disease_props, article_props):
        """Create ontology-style disease links while preserving legacy relations."""
        links = self.infer_ontology_links(disease, title=title, text=text, disease_props=disease_props, article_props=article_props)

        # Do not create SUPPORTS_DISEASE automatically.
        #
        # Entity detection proves only that an article mentions or discusses
        # a disease. A support relationship requires separate claim-level
        # evidence, provenance, and confidence.
        for symptom in set(entities.get("symptoms", [])):
            symptom_node = Node("Symptom", name=symptom)
            tx.merge(symptom_node, "Symptom", "name")
            tx.merge(Relationship(disease_node, "HAS_SYMPTOM", symptom_node))
            tx.merge(Relationship(article_node, "MENTIONS_SYMPTOM", symptom_node))

        for risk in set(entities.get("risk_factors", [])):
            risk_node = Node("RiskFactor", name=risk)
            tx.merge(risk_node, "RiskFactor", "name")
            tx.merge(Relationship(disease_node, "HAS_RISK_FACTOR", risk_node))

        for drug in set(entities.get("drugs", [])):
            drug_node = Node("Drug", name=drug)
            tx.merge(drug_node, "Drug", "name")
            tx.merge(Relationship(disease_node, "ASSOCIATED_WITH_DRUG", drug_node))

        for family in links["families"]:
            family_node = Node("StrokeFamily", name=family)
            tx.merge(family_node, "StrokeFamily", "name")
            tx.merge(Relationship(disease_node, "IS_SUBTYPE_OF", family_node))

        for territory in links["territories"]:
            territory_node = Node("Territory", name=territory)
            tx.merge(territory_node, "Territory", "name")
            tx.merge(Relationship(disease_node, "AFFECTS_TERRITORY", territory_node))

        for finding in links["imaging_findings"]:
            finding_node = Node("ImagingFinding", name=finding)
            tx.merge(finding_node, "ImagingFinding", "name")
            tx.merge(Relationship(disease_node, "HAS_IMAGING_FINDING", finding_node))

    def clear_graph(self, batch_size=500, sleep_seconds=0.1):
        """مسح الرسم البياني على دفعات لتجنب MemoryPoolOutOfMemoryError."""
        total_deleted = 0

        while True:
            result = self.graph.run(
                """
                MATCH (n)
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                batch_size=batch_size,
            ).data()

            deleted = result[0]["deleted"] if result else 0
            total_deleted += deleted

            print(f"تم حذف دفعة: {deleted} | المجموع: {total_deleted}")

            if deleted == 0:
                break

            time.sleep(sleep_seconds)

        print("تم مسح الرسم البياني القديم بالكامل")

    def drop_indexes(self):
        """تعطيل الفهارس الخاصة بالمشروع لتسريع الإدراج بطريقة متوافقة مع Neo4j 5+."""
        target_labels = {"Disease", "Symptom", "Drug", "RiskFactor", "Article", "StrokeFamily", "Territory", "ImagingFinding"}

        try:
            result = self.graph.run("SHOW INDEXES")
            dropped = 0

            for record in result:
                try:
                    name = record.get("name")
                    labels_or_types = record.get("labelsOrTypes") or []
                    entity_type = record.get("entityType")

                    if not name:
                        continue
                    if entity_type != "NODE":
                        continue
                    if not any(lbl in target_labels for lbl in labels_or_types):
                        continue

                    self.graph.run(f"DROP INDEX `{name}` IF EXISTS")
                    dropped += 1
                except Exception as inner_e:
                    print(f"تحذير: فشل حذف index واحد - {inner_e}")

            print(f"تم تعطيل الفهارس مؤقتاً ({dropped} indexes)")
        except Exception as e:
            print(f"تحذير: لم يتم تعطيل الفهارس - {e}")

    def recreate_indexes(self):
        """إعادة إنشاء الفهارس بعد الإدراج بصيغة Neo4j الحديثة."""
        statements = [
            "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX disease_category_idx IF NOT EXISTS FOR (d:Disease) ON (d.category)",
            "CREATE INDEX disease_acuity_idx IF NOT EXISTS FOR (d:Disease) ON (d.acuity)",
            "CREATE INDEX disease_vascular_idx IF NOT EXISTS FOR (d:Disease) ON (d.vascular_relevance)",
            "CREATE INDEX disease_posterior_idx IF NOT EXISTS FOR (d:Disease) ON (d.posterior_relevance)",
            "CREATE INDEX disease_emergency_idx IF NOT EXISTS FOR (d:Disease) ON (d.emergency_relevance)",
            "CREATE INDEX symptom_name_idx IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
            "CREATE INDEX drug_name_idx IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX riskfactor_name_idx IF NOT EXISTS FOR (r:RiskFactor) ON (r.name)",
            "CREATE INDEX article_pmid_idx IF NOT EXISTS FOR (a:Article) ON (a.pmid)",
            "CREATE INDEX article_chunk_idx IF NOT EXISTS FOR (a:Article) ON (a.chunk_id)",
            "CREATE INDEX article_type_idx IF NOT EXISTS FOR (a:Article) ON (a.article_type)",
            "CREATE INDEX strokefamily_name_idx IF NOT EXISTS FOR (f:StrokeFamily) ON (f.name)",
            "CREATE INDEX territory_name_idx IF NOT EXISTS FOR (t:Territory) ON (t.name)",
            "CREATE INDEX imagingfinding_name_idx IF NOT EXISTS FOR (i:ImagingFinding) ON (i.name)",
        ]

        try:
            for stmt in statements:
                self.graph.run(stmt)
            print("تم إعادة إنشاء الفهارس")
        except Exception as e:
            print(f"تحذير: فشل إنشاء الفهارس - {e}")

    def extract_entities(self, text):
        """
        Extract medical entities using:
        - whole-term matching,
        - acronym disambiguation,
        - canonicalization,
        - basic negation handling.
        """
        entities = {
            "diseases": [],
            "symptoms": [],
            "drugs": [],
            "risk_factors": []
        }

        text_normalized = self._normalize_text(text)

        disease_terms = list(
            self.disease_synonyms.keys()
        )

        symptom_terms = list(
            self.symptom_synonyms.keys()
        )

        drug_terms = list(
            self.drug_synonyms.keys()
        )

        risk_factor_terms = list(
            self.risk_factor_synonyms.keys()
        )

        for disease_term in disease_terms:
            normalized_term = self._normalize_text(
                disease_term
            )

            if normalized_term == "cva":
                matched = (
                    self._is_contextual_cva_stroke(text)
                )
            else:
                matched = self._term_occurs(
                    text_normalized,
                    disease_term
                )

            if (
                matched
                and not self._is_negated(
                    text_normalized,
                    disease_term
                )
            ):
                canonical_disease = (
                    self._canonicalize_term(
                        disease_term,
                        self.disease_synonyms
                    )
                )

                entities["diseases"].append(
                    canonical_disease
                )

        for symptom_term in symptom_terms:
            if (
                self._term_occurs(
                    text_normalized,
                    symptom_term
                )
                and not self._is_negated(
                    text_normalized,
                    symptom_term
                )
            ):
                entities["symptoms"].append(
                    self._canonicalize_term(
                        symptom_term,
                        self.symptom_synonyms
                    )
                )

        for drug_term in drug_terms:
            if (
                self._term_occurs(
                    text_normalized,
                    drug_term
                )
                and not self._is_negated(
                    text_normalized,
                    drug_term
                )
            ):
                entities["drugs"].append(
                    self._canonicalize_term(
                        drug_term,
                        self.drug_synonyms
                    )
                )

        for risk_term in risk_factor_terms:
            if (
                self._term_occurs(
                    text_normalized,
                    risk_term
                )
                and not self._is_negated(
                    text_normalized,
                    risk_term
                )
            ):
                entities["risk_factors"].append(
                    self._canonicalize_term(
                        risk_term,
                        self.risk_factor_synonyms
                    )
                )

        entities["diseases"] = list(
            dict.fromkeys(entities["diseases"])
        )

        entities["symptoms"] = list(
            dict.fromkeys(entities["symptoms"])
        )

        entities["drugs"] = list(
            dict.fromkeys(entities["drugs"])
        )

        entities["risk_factors"] = list(
            dict.fromkeys(
                entities["risk_factors"]
            )
        )

        posterior_diseases = (
            self._infer_posterior_diagnoses(
                text
            )
        )

        entities["diseases"].extend(
            posterior_diseases
        )

        entities["diseases"] = list(
            dict.fromkeys(
                entities["diseases"]
            )
        )

        entities["vascular_findings"] = (
            self.extract_vascular_findings(
                text
            )
        )

        return self._apply_taxonomy_policy(entities)

    def extract_entities_scispacy(self, text):
        """استخراج الكيانات باستخدام scispacy (لنص واحد) + canonicalization"""
        doc = nlp(text)
        entities = {
            "diseases": [],
            "drugs": []
        }

        for ent in doc.ents:
            ent_text = self._normalize_text(ent.text)

            if ent.label_ == "DISEASE":
                if not self._is_negated(text, ent_text):
                    entities["diseases"].append(self._canonicalize_term(ent_text, self.disease_synonyms))
            elif ent.label_ == "CHEMICAL":
                if not self._is_negated(text, ent_text):
                    entities["drugs"].append(self._canonicalize_term(ent_text, self.drug_synonyms))

        entities["diseases"] = list(dict.fromkeys(entities["diseases"]))
        entities["drugs"] = list(dict.fromkeys(entities["drugs"]))

        return entities

    def extract_entities_combined(self, text):
        """دمج الطريقة القديمة (للأعراض وعوامل الخطورة) مع scispacy لنص واحد"""
        entities_scispacy = self.extract_entities_scispacy(text)
        entities_old = self.extract_entities(text)

        combined = {
            "diseases": list(set(entities_scispacy.get("diseases", []) + entities_old.get("diseases", []))),
            "symptoms": entities_old.get("symptoms", []),
            "drugs": list(set(entities_scispacy.get("drugs", []) + entities_old.get("drugs", []))),
            "risk_factors": entities_old.get("risk_factors", [])
        }

        combined["vascular_findings"] = (
            self.extract_vascular_findings(
                text
            )
        )

        return self._apply_taxonomy_policy(combined)

    def build_from_chunks(
        self,
        chunks_dir,
        limit=None,
        batch_size=64,
        disable_indexes=True
    ):
        """
        Build the KG while extracting and classifying entities at
        complete-article level.

        A separate Article node is retained for every chunk because
        retrieval uses chunk_id, but chunks sharing one PMID receive
        the same article-level entities and classification.
        """
        chunk_files = sorted(
            Path(chunks_dir).glob("*.json")
        )

        if limit:
            chunk_files = chunk_files[:limit]

        total_chunks = len(chunk_files)

        print(
            "Starting graph build from "
            f"{total_chunks} chunks."
        )

        raw_rows = []

        for chunk_file in tqdm(
            chunk_files,
            desc="Reading chunk files"
        ):
            with open(
                chunk_file,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            raw_rows.append({
                "pmid":
                    data.get("pmid"),

                "title":
                    data.get("title", "") or "",

                "text":
                    data.get("text", "") or "",

                "chunk_id":
                    str(
                        data.get("chunk_id")
                        or chunk_file.stem
                    ),
            })

        # Critical policy:
        # aggregate the complete dataset before entity extraction.
        articles_data = (
            self._prepare_article_level_records(
                raw_rows
            )
        )

        if not articles_data:
            raise RuntimeError(
                "No valid chunk records were loaded."
            )

        unique_pmids = {
            str(row.get("pmid") or "").strip()
            for row in articles_data
            if str(
                row.get("pmid") or ""
            ).strip()
        }

        print(
            "Article-level preparation complete | "
            f"chunks={len(articles_data)} | "
            f"unique_pmids={len(unique_pmids)}"
        )

        # Do not alter Neo4j until the source files have been read and
        # article-level aggregation has completed successfully.
        if disable_indexes:
            self.drop_indexes()

        print(
            "Extracting article-level entities "
            "with SciSpacy and rule policy..."
        )

        all_entities = []

        for start in range(
            0,
            len(articles_data),
            batch_size
        ):
            batch = articles_data[
                start:start + batch_size
            ]

            batch_texts = [
                row["article_full_text"]
                for row in batch
            ]

            docs = list(
                nlp.pipe(
                    batch_texts,
                    batch_size=batch_size
                )
            )

            for position, doc in enumerate(docs):
                article_full_text = batch_texts[
                    position
                ]

                entities_scispacy = {
                    "diseases": [],
                    "drugs": [],
                }

                for entity in doc.ents:
                    entity_text = self._normalize_text(
                        entity.text
                    )

                    if entity.label_ == "DISEASE":
                        if not self._is_negated(
                            article_full_text,
                            entity_text
                        ):
                            entities_scispacy[
                                "diseases"
                            ].append(
                                self._canonicalize_term(
                                    entity_text,
                                    self.disease_synonyms
                                )
                            )

                    elif entity.label_ == "CHEMICAL":
                        if not self._is_negated(
                            article_full_text,
                            entity_text
                        ):
                            entities_scispacy[
                                "drugs"
                            ].append(
                                self._canonicalize_term(
                                    entity_text,
                                    self.drug_synonyms
                                )
                            )

                rule_entities = self.extract_entities(
                    article_full_text
                )

                combined = {
                    "diseases": list(
                        set(
                            entities_scispacy.get(
                                "diseases",
                                []
                            )
                            + rule_entities.get(
                                "diseases",
                                []
                            )
                        )
                    ),

                    "symptoms":
                        rule_entities.get(
                            "symptoms",
                            []
                        ),

                    "drugs": list(
                        set(
                            entities_scispacy.get(
                                "drugs",
                                []
                            )
                            + rule_entities.get(
                                "drugs",
                                []
                            )
                        )
                    ),

                    "risk_factors":
                        rule_entities.get(
                            "risk_factors",
                            []
                        ),

                    "vascular_findings":
                        self.extract_vascular_findings(
                            article_full_text
                        ),
                }

                # SciSpacy output must obey the same stroke taxonomy.
                combined = self._apply_taxonomy_policy(
                    combined
                )

                all_entities.append(combined)

        print(
            "Creating chunk nodes and article-level "
            "relationships in Neo4j..."
        )

        batch_insert_size = 500

        for start_idx in tqdm(
            range(
                0,
                len(articles_data),
                batch_insert_size
            ),
            desc="Inserting graph batches"
        ):
            end_idx = min(
                start_idx + batch_insert_size,
                len(articles_data)
            )

            tx = self.graph.begin()

            try:
                for idx in range(
                    start_idx,
                    end_idx
                ):
                    row = articles_data[idx]
                    entities = all_entities[idx]

                    pmid = row.get("pmid")

                    title = (
                        row.get("article_title")
                        or row.get("title")
                        or ""
                    )

                    article_text = (
                        row.get("article_text")
                        or row.get("text")
                        or ""
                    )

                    chunk_id = str(
                        row.get("chunk_id") or ""
                    )

                    article_chunk_count = int(
                        row.get(
                            "article_chunk_count",
                            1
                        )
                        or 1
                    )

                    article_props = self.classify_article(
                        title,
                        article_text
                    )

                    article_node = Node(
                        "Article",
                        pmid=pmid,
                        title=title,
                        chunk_id=chunk_id,
                        source="PubMed",
                        article_chunk_count=
                            article_chunk_count,
                        article_type=
                            article_props[
                                "article_type"
                            ],
                        article_emergency_relevance=
                            float(
                                article_props[
                                    "article_emergency_relevance"
                                ]
                            ),
                        article_vascular_relevance=
                            float(
                                article_props[
                                    "article_vascular_relevance"
                                ]
                            ),
                        article_posterior_relevance=
                            float(
                                article_props[
                                    "article_posterior_relevance"
                                ]
                            ),
                        article_hemorrhage_relevance=
                            float(
                                article_props[
                                    "article_hemorrhage_relevance"
                                ]
                            ),
                        article_noise_penalty=
                            float(
                                article_props[
                                    "article_noise_penalty"
                                ]
                            ),
                    )

                    tx.merge(
                        article_node,
                        "Article",
                        "chunk_id"
                    )

                    for disease in set(
                        entities["diseases"]
                    ):
                        disease_props = (
                            self.classify_disease(
                                disease
                            )
                        )

                        disease_node = Node(
                            "Disease",
                            name=disease,
                            category=
                                disease_props[
                                    "category"
                                ],
                            acuity=
                                disease_props[
                                    "acuity"
                                ],
                            vascular_relevance=
                                float(
                                    disease_props[
                                        "vascular_relevance"
                                    ]
                                ),
                            posterior_relevance=
                                float(
                                    disease_props[
                                        "posterior_relevance"
                                    ]
                                ),
                            hemorrhage_relevance=
                                float(
                                    disease_props[
                                        "hemorrhage_relevance"
                                    ]
                                ),
                            emergency_relevance=
                                float(
                                    disease_props[
                                        "emergency_relevance"
                                    ]
                                ),
                            chronic_penalty=
                                float(
                                    disease_props[
                                        "chronic_penalty"
                                    ]
                                ),
                            noise_penalty=
                                float(
                                    disease_props[
                                        "noise_penalty"
                                    ]
                                ),
                        )

                        tx.merge(
                            disease_node,
                            "Disease",
                            "name"
                        )

                        tx.merge(
                            Relationship(
                                article_node,
                                "DISCUSSES",
                                disease_node
                            )
                        )

                        self._merge_ontology_links(
                            tx,
                            article_node,
                            disease_node,
                            disease,
                            entities,
                            title,
                            article_text,
                            disease_props,
                            article_props,
                        )

                    for symptom in set(
                        entities["symptoms"]
                    ):
                        symptom_node = Node(
                            "Symptom",
                            name=symptom
                        )

                        tx.merge(
                            symptom_node,
                            "Symptom",
                            "name"
                        )

                        tx.merge(
                            Relationship(
                                article_node,
                                "MENTIONS",
                                symptom_node
                            )
                        )

                    for drug in set(
                        entities["drugs"]
                    ):
                        drug_node = Node(
                            "Drug",
                            name=drug
                        )

                        tx.merge(
                            drug_node,
                            "Drug",
                            "name"
                        )

                        tx.merge(
                            Relationship(
                                article_node,
                                "REFERENCES",
                                drug_node
                            )
                        )

                    for risk in set(
                        entities["risk_factors"]
                    ):
                        risk_node = Node(
                            "RiskFactor",
                            name=risk
                        )

                        tx.merge(
                            risk_node,
                            "RiskFactor",
                            "name"
                        )

                        tx.merge(
                            Relationship(
                                article_node,
                                "ASSOCIATED_WITH",
                                risk_node
                            )
                        )

                    for finding in set(
                        entities.get(
                            "vascular_findings",
                            []
                        )
                        or []
                    ):
                        if not finding:
                            continue

                        finding_node = Node(
                            "VascularFinding",
                            name=finding
                        )

                        tx.merge(
                            finding_node,
                            "VascularFinding",
                            "name"
                        )

                        tx.merge(
                            Relationship(
                                article_node,
                                "MENTIONS_VASCULAR_FINDING",
                                finding_node
                            )
                        )

                tx.commit()

            except Exception as error:
                tx.rollback()

                print(
                    "Batch error "
                    f"{start_idx}-{end_idx}: "
                    f"{error}"
                )

                raise

        if disable_indexes:
            self.recreate_indexes()

        print(
            "Graph build completed | "
            f"chunks={len(articles_data)} | "
            f"unique_pmids={len(unique_pmids)}"
        )

    def _score_symptom_query_row(
        self,
        row,
        is_acute=False,
        is_posterior=False,
        is_focal=False,
        is_hemorrhage=False,
    ):
        frequency = float(row.get("frequency", 0) or 0.0)
        matched_symptoms_count = float(row.get("matched_symptoms_count", 0) or 0.0)

        vascular_relevance = float(row.get("vascular_relevance", 0) or 0.0)
        posterior_relevance = float(row.get("posterior_relevance", 0) or 0.0)
        hemorrhage_relevance = float(row.get("hemorrhage_relevance", 0) or 0.0)
        emergency_relevance = float(row.get("emergency_relevance", 0) or 0.0)
        chronic_penalty = float(row.get("chronic_penalty", 0) or 0.0)
        disease_noise_penalty = float(row.get("noise_penalty", 0) or 0.0)

        article_emergency_relevance = float(row.get("article_emergency_relevance", 0) or 0.0)
        article_vascular_relevance = float(row.get("article_vascular_relevance", 0) or 0.0)
        article_posterior_relevance = float(row.get("article_posterior_relevance", 0) or 0.0)
        article_hemorrhage_relevance = float(row.get("article_hemorrhage_relevance", 0) or 0.0)
        article_noise_penalty = float(row.get("article_noise_penalty", 0) or 0.0)

        score = 0.18 * frequency
        score += 0.75 * matched_symptoms_count

        if is_acute:
            score += 3.00 * emergency_relevance
            score += 2.50 * vascular_relevance
            score -= 3.50 * chronic_penalty
            score -= 2.50 * disease_noise_penalty

        if is_posterior:
            score += 3.50 * posterior_relevance
            score += 1.50 * vascular_relevance
            score -= 2.50 * chronic_penalty

        if is_focal:
            score += 2.00 * vascular_relevance
            score += 1.00 * emergency_relevance

        if is_hemorrhage:
            score += 4.00 * hemorrhage_relevance
            score += 1.00 * emergency_relevance
            score -= 1.50 * chronic_penalty

        if is_acute:
            score += 1.30 * article_emergency_relevance
            score += 1.20 * article_vascular_relevance
            score -= 1.50 * article_noise_penalty

        if is_posterior:
            score += 1.50 * article_posterior_relevance
            score += 0.80 * article_vascular_relevance

        if is_hemorrhage:
            score += 1.50 * article_hemorrhage_relevance

        row["clinical_score"] = score
        return row

    def query_by_symptoms(
        self,
        symptoms,
        limit=5,
        is_acute=False,
        is_posterior=False,
        is_focal=False,
        is_hemorrhage=False,
    ):
        query = """
        CALL {
            MATCH (s:Symptom)<-[:MENTIONS|MENTIONS_SYMPTOM]-(a:Article)-[:DISCUSSES|SUPPORTS_DISEASE]->(d:Disease)
            WHERE s.name IN $symptoms
            RETURN d, a, s
            UNION
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name IN $symptoms
            OPTIONAL MATCH (d)<-[:DISCUSSES|SUPPORTS_DISEASE]-(a:Article)
            RETURN d, a, s
        }
        OPTIONAL MATCH (d)-[:IS_SUBTYPE_OF]->(sf:StrokeFamily)
        OPTIONAL MATCH (d)-[:AFFECTS_TERRITORY]->(t:Territory)
        OPTIONAL MATCH (d)-[:HAS_IMAGING_FINDING]->(im:ImagingFinding)
        WITH d,
             COUNT(DISTINCT a) as frequency,
             COUNT(DISTINCT s) as matched_symptoms_count,
             COLLECT(DISTINCT a.pmid) as articles,
             COLLECT(DISTINCT sf.name) as stroke_families,
             COLLECT(DISTINCT t.name) as territories,
             COLLECT(DISTINCT im.name) as imaging_findings,
             avg(coalesce(a.article_emergency_relevance, 0.0)) as article_emergency_relevance,
             avg(coalesce(a.article_vascular_relevance, 0.0)) as article_vascular_relevance,
             avg(coalesce(a.article_posterior_relevance, 0.0)) as article_posterior_relevance,
             avg(coalesce(a.article_hemorrhage_relevance, 0.0)) as article_hemorrhage_relevance,
             avg(coalesce(a.article_noise_penalty, 0.0)) as article_noise_penalty
        RETURN d.name as disease,
               frequency,
               matched_symptoms_count,
               articles,
               stroke_families,
               territories,
               imaging_findings,
               coalesce(d.category, 'unknown') as category,
               coalesce(d.acuity, 'unknown') as acuity,
               coalesce(d.vascular_relevance, 0.0) as vascular_relevance,
               coalesce(d.posterior_relevance, 0.0) as posterior_relevance,
               coalesce(d.hemorrhage_relevance, 0.0) as hemorrhage_relevance,
               coalesce(d.emergency_relevance, 0.0) as emergency_relevance,
               coalesce(d.chronic_penalty, 0.0) as chronic_penalty,
               coalesce(d.noise_penalty, 0.0) as noise_penalty,
               article_emergency_relevance,
               article_vascular_relevance,
               article_posterior_relevance,
               article_hemorrhage_relevance,
               article_noise_penalty
        ORDER BY frequency DESC
        LIMIT $candidate_limit
        """

        candidate_limit = max(limit * 30, 300)

        result = self.graph.run(
            query,
            symptoms=symptoms,
            candidate_limit=candidate_limit,
        )

        rows = [dict(record) for record in result]

        scored = [
            self._score_symptom_query_row(
                row,
                is_acute=is_acute,
                is_posterior=is_posterior,
                is_focal=is_focal,
                is_hemorrhage=is_hemorrhage,
            )
            for row in rows
        ]

        scored.sort(
            key=lambda x: float(x.get("clinical_score", 0.0) or 0.0),
            reverse=True
        )

        return scored[:limit]

    def get_related_articles(self, disease=None, symptom=None, limit=5):
        if disease:
            query = """
            MATCH (d:Disease {name: $disease})<-[:DISCUSSES|SUPPORTS_DISEASE]-(a:Article)
            OPTIONAL MATCH (d)-[:IS_SUBTYPE_OF]->(sf:StrokeFamily)
            OPTIONAL MATCH (d)-[:AFFECTS_TERRITORY]->(t:Territory)
            OPTIONAL MATCH (d)-[:HAS_IMAGING_FINDING]->(im:ImagingFinding)
            RETURN a.pmid as pmid,
                   a.title as title,
                   a.chunk_id as chunk_id,
                   a.article_type as article_type,
                   COLLECT(DISTINCT sf.name) as stroke_families,
                   COLLECT(DISTINCT t.name) as territories,
                   COLLECT(DISTINCT im.name) as imaging_findings,
                   coalesce(a.article_emergency_relevance, 0.0) as article_emergency_relevance,
                   coalesce(a.article_vascular_relevance, 0.0) as article_vascular_relevance,
                   coalesce(a.article_posterior_relevance, 0.0) as article_posterior_relevance,
                   coalesce(a.article_hemorrhage_relevance, 0.0) as article_hemorrhage_relevance,
                   coalesce(a.article_noise_penalty, 0.0) as article_noise_penalty,
                   coalesce(d.category, 'unknown') as category,
                   coalesce(d.acuity, 'unknown') as acuity,
                   coalesce(d.vascular_relevance, 0.0) as vascular_relevance,
                   coalesce(d.posterior_relevance, 0.0) as posterior_relevance,
                   coalesce(d.hemorrhage_relevance, 0.0) as hemorrhage_relevance,
                   coalesce(d.emergency_relevance, 0.0) as emergency_relevance,
                   coalesce(d.chronic_penalty, 0.0) as chronic_penalty,
                   coalesce(d.noise_penalty, 0.0) as noise_penalty
            ORDER BY article_emergency_relevance DESC,
                     article_vascular_relevance DESC,
                     article_posterior_relevance DESC,
                     article_noise_penalty ASC
            LIMIT $limit
            """
            result = self.graph.run(query, disease=disease, limit=limit)

        elif symptom:
            query = """
            MATCH (s:Symptom {name: $symptom})<-[:MENTIONS|MENTIONS_SYMPTOM]-(a:Article)
            OPTIONAL MATCH (a)-[:DISCUSSES|SUPPORTS_DISEASE]->(d:Disease)
            OPTIONAL MATCH (d)-[:IS_SUBTYPE_OF]->(sf:StrokeFamily)
            OPTIONAL MATCH (d)-[:AFFECTS_TERRITORY]->(t:Territory)
            OPTIONAL MATCH (d)-[:HAS_IMAGING_FINDING]->(im:ImagingFinding)
            RETURN a.pmid as pmid,
                   a.title as title,
                   a.chunk_id as chunk_id,
                   a.article_type as article_type,
                   COLLECT(DISTINCT sf.name) as stroke_families,
                   COLLECT(DISTINCT t.name) as territories,
                   COLLECT(DISTINCT im.name) as imaging_findings,
                   coalesce(a.article_emergency_relevance, 0.0) as article_emergency_relevance,
                   coalesce(a.article_vascular_relevance, 0.0) as article_vascular_relevance,
                   coalesce(a.article_posterior_relevance, 0.0) as article_posterior_relevance,
                   coalesce(a.article_hemorrhage_relevance, 0.0) as article_hemorrhage_relevance,
                   coalesce(a.article_noise_penalty, 0.0) as article_noise_penalty,
                   coalesce(d.category, 'unknown') as category,
                   coalesce(d.acuity, 'unknown') as acuity,
                   coalesce(d.vascular_relevance, 0.0) as vascular_relevance,
                   coalesce(d.posterior_relevance, 0.0) as posterior_relevance,
                   coalesce(d.hemorrhage_relevance, 0.0) as hemorrhage_relevance,
                   coalesce(d.emergency_relevance, 0.0) as emergency_relevance,
                   coalesce(d.chronic_penalty, 0.0) as chronic_penalty,
                   coalesce(d.noise_penalty, 0.0) as noise_penalty
            ORDER BY article_emergency_relevance DESC,
                     article_vascular_relevance DESC,
                     article_posterior_relevance DESC,
                     article_noise_penalty ASC
            LIMIT $limit
            """
            result = self.graph.run(query, symptom=symptom, limit=limit)

        else:
            return []

        return [dict(record) for record in result]


    def backfill_ontology_links(self, batch_size=500, sleep_seconds=0.05):
        """
        Add v7.12.4 ontology-style relationships to an existing Neo4j graph
        without deleting or rebuilding the Article/Disease/Symptom nodes.

        This is a migration/backfill helper. A full rebuild remains the cleanest
        path, but this lets you upgrade the current graph quickly.
        """
        total = 0
        skip = 0

        print("بدء backfill لعلاقات ontology فوق الغراف الحالي...")

        while True:
            rows = self.graph.run(
                """
                MATCH (a:Article)-[:DISCUSSES]->(d:Disease)
                OPTIONAL MATCH (a)-[:MENTIONS]->(s:Symptom)
                OPTIONAL MATCH (a)-[:REFERENCES]->(drug:Drug)
                OPTIONAL MATCH (a)-[:ASSOCIATED_WITH]->(risk:RiskFactor)
                RETURN a.chunk_id AS chunk_id,
                       a.pmid AS pmid,
                       a.title AS title,
                       d.name AS disease,
                       COLLECT(DISTINCT s.name) AS symptoms,
                       COLLECT(DISTINCT drug.name) AS drugs,
                       COLLECT(DISTINCT risk.name) AS risk_factors
                SKIP $skip
                LIMIT $batch_size
                """,
                skip=skip,
                batch_size=batch_size,
            ).data()

            if not rows:
                break

            tx = self.graph.begin()
            try:
                for row in rows:
                    disease = row.get("disease")
                    chunk_id = row.get("chunk_id")
                    if not disease or not chunk_id:
                        continue

                    title = row.get("title") or ""
                    entities = {
                        "symptoms": [x for x in (row.get("symptoms") or []) if x],
                        "drugs": [x for x in (row.get("drugs") or []) if x],
                        "risk_factors": [x for x in (row.get("risk_factors") or []) if x],
                    }
                    disease_props = self.classify_disease(disease)
                    article_props = self.classify_article(title, "")

                    article_node = Node("Article", chunk_id=chunk_id, pmid=row.get("pmid"), title=title)
                    disease_node = Node("Disease", name=disease)
                    tx.merge(article_node, "Article", "chunk_id")
                    tx.merge(disease_node, "Disease", "name")
                    self._merge_ontology_links(
                        tx,
                        article_node,
                        disease_node,
                        disease,
                        entities,
                        title,
                        "",
                        disease_props,
                        article_props,
                    )

                tx.commit()
            except Exception as e:
                tx.rollback()
                print(f"تحذير: فشل backfill عند skip={skip}: {e}")

            total += len(rows)
            skip += batch_size
            print(f"تمت معالجة {total} Article-Disease pairs")
            time.sleep(sleep_seconds)

        self.recreate_indexes()
        print("انتهى backfill لعلاقات ontology")


if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    kg.clear_graph()
    kg.build_from_chunks("app/kb/chunks", limit=50000, batch_size=64, disable_indexes=True)