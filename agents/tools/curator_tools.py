"""
curator_tools.py — tools del curator_agent (Fase 2). Las dos tareas de curacion sin
precedente en el repo (¿es texto del autor o una resena/resumen?, ¿es un pasaje
caracteristico de su voz o boilerplate generico?) se implementan como LLM-as-judge,
pero el juicio en si lo hace el PROPIO curator_agent (via su system prompt con la
rubrica), no una llamada a LLM escondida dentro de estos tools -- los tools solo
registran el veredicto que el agent ya razono y aplican el threshold (que vive en
mia_common/settings.py, no hardcodeado en el prompt, para que sea facil de ajustar).

No hay benchmark etiquetado para validar ninguna de las dos rubricas (limitacion
metodologica reconocida) -- por eso los casos borderline van a revision humana en vez
de descartarse en silencio.
"""

from __future__ import annotations

from typing import Literal

from agents.tools.fs_tools import write_run_artifact
from mia_common.settings import settings


def record_authorship_verdict(
    run_id: str,
    document_id: str,
    is_by_author: bool,
    confidence: float,
    text_type: Literal["original_prose", "review", "summary", "biography", "interview", "other"],
    reasoning: str,
) -> dict:
    """Registra el veredicto de autoria que EL AGENTE ya evaluo para `document_id`
    (un documento scrapeado, antes de chunkear) y devuelve la decision de gating:
      - "keep": pasa a chunking.
      - "needs_human_review": confidence en zona borderline -- mostrar en la pantalla
        de revision en vez de descartar en silencio.
      - "drop": no es texto del autor (resena/resumen/biografia/etc), se descarta.
    """
    low, high = settings.authorship_review_band
    if not is_by_author:
        decision = "drop"
    elif confidence < low:
        decision = "drop"
    elif confidence < settings.authorship_min_confidence or confidence < high:
        decision = "needs_human_review"
    else:
        decision = "keep"

    record = {
        "document_id": document_id,
        "is_by_author": is_by_author,
        "confidence": confidence,
        "text_type": text_type,
        "reasoning": reasoning,
        "decision": decision,
    }
    write_run_artifact(run_id, "curation", f"authorship_{document_id}", record)
    return record


def record_voice_score(
    run_id: str,
    chunk_id: str,
    distinctiveness: float,
    is_boilerplate: bool,
    reasoning: str,
) -> dict:
    """Registra el puntaje de distintividad de voz que EL AGENTE ya evaluo para un
    chunk (despues de chunkear -- "voz caracteristica" es un juicio a nivel
    oracion/parrafo). decision="keep" si no es boilerplate y supera
    settings.voice_min_distinctiveness; "drop" en caso contrario."""
    decision = "drop" if is_boilerplate or distinctiveness < settings.voice_min_distinctiveness else "keep"

    record = {
        "chunk_id": chunk_id,
        "distinctiveness": distinctiveness,
        "is_boilerplate": is_boilerplate,
        "reasoning": reasoning,
        "decision": decision,
    }
    write_run_artifact(run_id, "curation", f"voice_{chunk_id}", record)
    return record
