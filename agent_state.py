# -*- coding: utf-8 -*-
"""
MedRAG v7.13 Agent State.

Structured state carried across the agentic workflow.  This schema contains no
clinical diagnosis rules; it only stores retrieval, evidence evaluation,
grounding, safety, and post-decision explainability metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class MedRAGAgentState:
    case_text: str
    normalized_case: str = ""
    intent: Dict[str, Any] = field(default_factory=dict)
    clinical_signals: Dict[str, Any] = field(default_factory=dict)
    initial_docs: List[Dict[str, Any]] = field(default_factory=list)
    feedback_query: str = ""
    second_stage_docs: List[Dict[str, Any]] = field(default_factory=list)
    reranked_docs: List[Dict[str, Any]] = field(default_factory=list)
    final_docs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_kg_verifications: Dict[str, Any] = field(default_factory=dict)
    candidate_kg_docs: List[Dict[str, Any]] = field(default_factory=list)
    grounding: Dict[str, Any] = field(default_factory=dict)
    candidate_diagnoses: List[str] = field(default_factory=list)
    evidence_judgments: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Dict[str, Any] = field(default_factory=dict)
    supporting_sources: List[Dict[str, Any]] = field(default_factory=list)
    conflicting_sources: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = "Evidence is insufficient"
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    agent_steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, step: str, **payload: Any) -> None:
        self.agent_steps.append({"step": step, **payload})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
