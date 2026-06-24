"""
schemas.py — tipos compartidos entre mia_common/, agents/ y webapp/. No reemplaza los
dataclasses propios de SAGE/DUALTEST/processRawText (Chunk, Completion, etc.), que
siguen viviendo en sus modulos -- esto es solo lo que cruza el limite entre etapas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CandidateText:
    """Un documento candidato encontrado por bibliography_agent, antes de curacion."""

    title: str
    source_url: str
    raw_html_or_text: str
    author: str
    date: str | None = None
    added_by_user: bool = False


@dataclass
class AuthorshipVerdict:
    is_by_author: bool
    confidence: float
    text_type: Literal[
        "original_prose", "review", "summary", "biography", "interview", "other"
    ]
    reasoning: str


@dataclass
class VoiceScore:
    distinctiveness: float
    is_boilerplate: bool
    reasoning: str


@dataclass
class MethodScore:
    """Score normalizado de un metodo MIA para un chunk, listo para el ensemble."""

    method: Literal["dualtest", "simia", "decop"]
    raw: Any
    normalized_score: float | None  # None == abstencion
    abstained: bool
    detail: dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    final_probability: float | None
    per_method: dict[str, dict]
    weights_used: dict[str, float]
    reason: str | None = None
