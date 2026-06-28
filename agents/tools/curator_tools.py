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

from typing import Callable, Literal

from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact
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

    Freno duro en CODIGO (no solo en el prompt): si `document_id` YA tiene un
    veredicto registrado (de esta corrida o de una ronda de reemplazo anterior), esta
    llamada NO lo pisa -- devuelve el veredicto EXISTENTE con "already_recorded": true.
    Bug real que motivo esto: curator_agent reproceso "emma" en una ronda de reemplazo
    aunque ya tenia veredicto de la ronda anterior (el prompt ya le decia que lo
    saltee, pero un LLM no sigue esa instruccion el 100% de las veces) -- gasto ~22
    llamadas de voz de mas Y termino pisando el veredicto con uno inconsistente. Un
    veredicto de autoria, una vez registrado, es el registro historico de esa ronda --
    nunca se pisa solo."""
    try:
        existing = read_run_artifact(run_id, "curation", f"authorship_{document_id}")
        return {**existing, "already_recorded": True}
    except FileNotFoundError:
        pass

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


def _kept_chunks_so_far(run_id: str, document_id: str) -> int:
    """Cuenta cuantos chunks de `document_id` ya tienen decision="keep" en
    runs/<run_id>/curation/voice_*.json -- usado por record_voice_score para frenar
    en codigo, no solo confiar en que el agente cuente bien y pare solo (ver el
    procedimiento de lotes chicos en agents/subagents/curator_agent.py)."""
    count = 0
    for name in list_run_artifacts(run_id).get("curation", []):
        if not name.startswith("voice_"):
            continue
        prior = read_run_artifact(run_id, "curation", name.removesuffix(".json"))
        if prior.get("document_id") == document_id and prior.get("decision") == "keep":
            count += 1
    return count


def record_voice_score(
    run_id: str,
    document_id: str,
    chunk_id: str,
    distinctiveness: float,
    is_boilerplate: bool,
    reasoning: str,
) -> dict:
    """Registra el puntaje de distintividad de voz que EL AGENTE ya evaluo para un
    chunk (despues de chunkear -- "voz caracteristica" es un juicio a nivel
    oracion/parrafo). decision="keep" si no es boilerplate y supera
    settings.voice_min_distinctiveness; "drop" en caso contrario.

    Freno duro de costo (independiente de que el agente siga bien las instrucciones
    de pedir de a poco, ver curator_agent.py): si `document_id` ya alcanzo
    settings.curator_target_chunks_per_text chunks "keep", esta llamada NO se
    registra (decision="target_reached") -- llamar a esto mas chunks de un documento
    que ya llego al objetivo no tiene efecto, asi que no vale la pena seguir
    insistiendo con ese documento."""
    kept_so_far = _kept_chunks_so_far(run_id, document_id)
    if kept_so_far >= settings.curator_target_chunks_per_text:
        return {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "decision": "target_reached",
            "reasoning": (
                f"{document_id} ya tiene {kept_so_far} chunks 'keep' "
                f"(objetivo: {settings.curator_target_chunks_per_text}) -- no juzgues "
                f"mas chunks de este documento, segui con el siguiente paso."
            ),
        }

    decision = "drop" if is_boilerplate or distinctiveness < settings.voice_min_distinctiveness else "keep"

    record = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "distinctiveness": distinctiveness,
        "is_boilerplate": is_boilerplate,
        "reasoning": reasoning,
        "decision": decision,
    }
    write_run_artifact(run_id, "curation", f"voice_{chunk_id}", record)
    return record


def make_run_scoped_curator_tools(run_id: str) -> dict[str, Callable]:
    """Devuelve record_authorship_verdict/record_voice_score con `run_id` ya fijo via
    closure -- mismo motivo que make_run_scoped_fs_tools (agents/tools/fs_tools.py):
    curator_agent ya opera sobre UN run fijo durante toda su tarea."""

    def record_authorship_verdict_bound(
        document_id: str,
        is_by_author: bool,
        confidence: float,
        text_type: Literal["original_prose", "review", "summary", "biography", "interview", "other"],
        reasoning: str,
    ) -> dict:
        """Registra el veredicto de autoria que EL AGENTE ya evaluo para
        `document_id` y devuelve la decision de gating ("keep" / "needs_human_review"
        / "drop")."""
        return record_authorship_verdict(run_id, document_id, is_by_author, confidence, text_type, reasoning)

    def record_voice_score_bound(
        document_id: str,
        chunk_id: str,
        distinctiveness: float,
        is_boilerplate: bool,
        reasoning: str,
    ) -> dict:
        """Registra el puntaje de distintividad de voz que EL AGENTE ya evaluo para
        un chunk. decision="keep"/"drop"/"target_reached" segun el threshold y el
        objetivo de chunks por documento."""
        return record_voice_score(run_id, document_id, chunk_id, distinctiveness, is_boilerplate, reasoning)

    record_authorship_verdict_bound.__name__ = "record_authorship_verdict"
    record_voice_score_bound.__name__ = "record_voice_score"

    return {
        "record_authorship_verdict": record_authorship_verdict_bound,
        "record_voice_score": record_voice_score_bound,
    }
