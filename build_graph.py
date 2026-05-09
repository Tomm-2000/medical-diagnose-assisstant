# -*- coding: utf-8 -*-
import json
from pathlib import Path
from py2neo import Graph, Node, Relationship
import re
from tqdm import tqdm
import scispacy
import spacy
from spacy import displacy
import time

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
            "atrial fibrillation": "atrial fibrillation",
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

    def _count_term_hits(self, text, terms):
        low = self._normalize_text(text)
        return sum(1 for t in terms if t in low)

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
        target_labels = {"Disease", "Symptom", "Drug", "RiskFactor", "Article"}

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
        ]

        try:
            for stmt in statements:
                self.graph.run(stmt)
            print("تم إعادة إنشاء الفهارس")
        except Exception as e:
            print(f"تحذير: فشل إنشاء الفهارس - {e}")

    def extract_entities(self, text):
        """استخراج الكيانات الطبية من النص (نموذج مبسط + canonicalization + negation handling)"""
        entities = {
            "diseases": [],
            "symptoms": [],
            "drugs": [],
            "risk_factors": []
        }

        text_lower = self._normalize_text(text)

        disease_terms = list(self.disease_synonyms.keys())
        symptom_terms = list(self.symptom_synonyms.keys())
        drug_terms = list(self.drug_synonyms.keys())
        risk_factor_terms = list(self.risk_factor_synonyms.keys())

        for d in disease_terms:
            if d in text_lower and not self._is_negated(text_lower, d):
                entities["diseases"].append(self._canonicalize_term(d, self.disease_synonyms))

        for s in symptom_terms:
            if s in text_lower and not self._is_negated(text_lower, s):
                entities["symptoms"].append(self._canonicalize_term(s, self.symptom_synonyms))

        for dr in drug_terms:
            if dr in text_lower and not self._is_negated(text_lower, dr):
                entities["drugs"].append(self._canonicalize_term(dr, self.drug_synonyms))

        for r in risk_factor_terms:
            if r in text_lower and not self._is_negated(text_lower, r):
                entities["risk_factors"].append(self._canonicalize_term(r, self.risk_factor_synonyms))

        entities["diseases"] = list(dict.fromkeys(entities["diseases"]))
        entities["symptoms"] = list(dict.fromkeys(entities["symptoms"]))
        entities["drugs"] = list(dict.fromkeys(entities["drugs"]))
        entities["risk_factors"] = list(dict.fromkeys(entities["risk_factors"]))

        return entities

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

        return combined

    def build_from_chunks(self, chunks_dir, limit=None, batch_size=64, disable_indexes=True):
        """
        بناء الرسم البياني من الـ chunks مع تحسينات الأداء.
        """
        chunk_files = list(Path(chunks_dir).glob("*.json"))
        if limit:
            chunk_files = chunk_files[:limit]

        total_chunks = len(chunk_files)
        print(f"بدء بناء الرسم البياني من {total_chunks} chunk (قد يستغرق وقتاً طويلاً)")

        if disable_indexes:
            self.drop_indexes()

        articles_data = []

        for chunk_file in tqdm(chunk_files, desc="قراءة ملفات الـ chunks"):
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "")
            title = data.get("title", "")
            pmid = data.get("pmid")
            chunk_id = data.get("chunk_id") or chunk_file.stem
            full_text = title + " " + text

            articles_data.append((pmid, title, text, full_text, chunk_id))

        print("بدء استخراج الكيانات باستخدام scispacy (بالمعالجة المجمعة)...")
        all_entities = []

        for i in range(0, len(articles_data), batch_size):
            batch = articles_data[i:i + batch_size]
            batch_texts = [item[3] for item in batch]
            docs = list(nlp.pipe(batch_texts, batch_size=batch_size))

            for j, doc in enumerate(docs):
                entities_scispacy = {"diseases": [], "drugs": []}

                for ent in doc.ents:
                    ent_text = self._normalize_text(ent.text)

                    if ent.label_ == "DISEASE":
                        if not self._is_negated(batch_texts[j], ent_text):
                            entities_scispacy["diseases"].append(
                                self._canonicalize_term(ent_text, self.disease_synonyms)
                            )
                    elif ent.label_ == "CHEMICAL":
                        if not self._is_negated(batch_texts[j], ent_text):
                            entities_scispacy["drugs"].append(
                                self._canonicalize_term(ent_text, self.drug_synonyms)
                            )

                entities_old = self.extract_entities(batch_texts[j])

                combined = {
                    "diseases": list(set(entities_scispacy.get("diseases", []) + entities_old.get("diseases", []))),
                    "symptoms": entities_old.get("symptoms", []),
                    "drugs": list(set(entities_scispacy.get("drugs", []) + entities_old.get("drugs", []))),
                    "risk_factors": entities_old.get("risk_factors", [])
                }

                all_entities.append(combined)

        print("إنشاء العقد والعلاقات في Neo4j...")
        batch_insert_size = 500

        for start_idx in tqdm(range(0, len(articles_data), batch_insert_size), desc="إدراج دفعات"):
            end_idx = min(start_idx + batch_insert_size, len(articles_data))
            tx = self.graph.begin()

            try:
                for idx in range(start_idx, end_idx):
                    (pmid, title, text, _, chunk_id), entities = articles_data[idx], all_entities[idx]

                    article_props = self.classify_article(title, text)

                    article_node = Node(
                        "Article",
                        pmid=pmid,
                        title=title,
                        chunk_id=chunk_id,
                        source="PubMed",
                        article_type=article_props["article_type"],
                        article_emergency_relevance=float(article_props["article_emergency_relevance"]),
                        article_vascular_relevance=float(article_props["article_vascular_relevance"]),
                        article_posterior_relevance=float(article_props["article_posterior_relevance"]),
                        article_hemorrhage_relevance=float(article_props["article_hemorrhage_relevance"]),
                        article_noise_penalty=float(article_props["article_noise_penalty"]),
                    )

                    tx.merge(article_node, "Article", "chunk_id")

                    for disease in set(entities["diseases"]):
                        disease_props = self.classify_disease(disease)

                        disease_node = Node(
                            "Disease",
                            name=disease,
                            category=disease_props["category"],
                            acuity=disease_props["acuity"],
                            vascular_relevance=float(disease_props["vascular_relevance"]),
                            posterior_relevance=float(disease_props["posterior_relevance"]),
                            hemorrhage_relevance=float(disease_props["hemorrhage_relevance"]),
                            emergency_relevance=float(disease_props["emergency_relevance"]),
                            chronic_penalty=float(disease_props["chronic_penalty"]),
                            noise_penalty=float(disease_props["noise_penalty"]),
                        )

                        tx.merge(disease_node, "Disease", "name")
                        rel = Relationship(article_node, "DISCUSSES", disease_node)
                        tx.merge(rel)

                    for symptom in set(entities["symptoms"]):
                        symptom_node = Node("Symptom", name=symptom)
                        tx.merge(symptom_node, "Symptom", "name")
                        rel = Relationship(article_node, "MENTIONS", symptom_node)
                        tx.merge(rel)

                    for drug in set(entities["drugs"]):
                        drug_node = Node("Drug", name=drug)
                        tx.merge(drug_node, "Drug", "name")
                        rel = Relationship(article_node, "REFERENCES", drug_node)
                        tx.merge(rel)

                    for risk in set(entities["risk_factors"]):
                        risk_node = Node("RiskFactor", name=risk)
                        tx.merge(risk_node, "RiskFactor", "name")
                        rel = Relationship(article_node, "ASSOCIATED_WITH", risk_node)
                        tx.merge(rel)

                tx.commit()

            except Exception as e:
                tx.rollback()
                print(f"خطأ في الدفعة {start_idx}-{end_idx}: {e}. إعادة المحاولة بعد 5 ثوان...")
                time.sleep(5)

        if disable_indexes:
            self.recreate_indexes()

        print(f"تم بناء الرسم البياني من {len(articles_data)} chunk")

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
        MATCH (s:Symptom)<-[:MENTIONS]-(a:Article)-[:DISCUSSES]->(d:Disease)
        WHERE s.name IN $symptoms
        WITH d,
             COUNT(DISTINCT a) as frequency,
             COUNT(DISTINCT s) as matched_symptoms_count,
             COLLECT(DISTINCT a.pmid) as articles,
             avg(coalesce(a.article_emergency_relevance, 0.0)) as article_emergency_relevance,
             avg(coalesce(a.article_vascular_relevance, 0.0)) as article_vascular_relevance,
             avg(coalesce(a.article_posterior_relevance, 0.0)) as article_posterior_relevance,
             avg(coalesce(a.article_hemorrhage_relevance, 0.0)) as article_hemorrhage_relevance,
             avg(coalesce(a.article_noise_penalty, 0.0)) as article_noise_penalty
        RETURN d.name as disease,
               frequency,
               matched_symptoms_count,
               articles,
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
            MATCH (d:Disease {name: $disease})<-[:DISCUSSES]-(a:Article)
            RETURN a.pmid as pmid,
                   a.title as title,
                   a.chunk_id as chunk_id,
                   a.article_type as article_type,
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
            MATCH (s:Symptom {name: $symptom})<-[:MENTIONS]-(a:Article)
            OPTIONAL MATCH (a)-[:DISCUSSES]->(d:Disease)
            RETURN a.pmid as pmid,
                   a.title as title,
                   a.chunk_id as chunk_id,
                   a.article_type as article_type,
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


if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    kg.clear_graph()
    kg.build_from_chunks("app/kb/chunks", limit=50000, batch_size=64, disable_indexes=True)