"""
run_manager.py — puente entre el orquestador deepagents (sincrono, bloqueante,
maneja human-in-the-loop con Command(resume=...)) y FastAPI (async, request/response).

Un RunHandle por run vive en RUNS mientras el proceso de uvicorn este arriba -- ese
diccionario en memoria (status, decisiones pendientes) no sobrevive un reinicio del
server. El checkpoint del orquestador (SqliteSaver, runs/_checkpoints.sqlite) si
sobrevive -- ver scripts/run_pipeline_agentic.py --run-id para reanudar un thread_id
existente fuera de la webapp. El estado "real" del run (lo que se decidio en cada
etapa) sigue viviendo en runs/<run_id>/ via fs_tools, no aca.

El orquestador corre en un thread de background por run; la vuelta de un humano
(aprobar/editar/rechazar) se entrega via submit_decisions(), que despierta ese thread.
"""

from __future__ import annotations

import glob
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents.orchestrator import build_orchestrator, message_has_real_content
from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact
from mia_common.settings import settings

RunStatus = Literal["running", "waiting_human", "done", "error"]

# Modelo target (el "black box" bajo test de MIA) que puede elegir el humano en el
# formulario de la webapp -- un default por proveedor ya validado/razonable, no
# necesariamente el unico modelo soportado por ese proveedor (mia_common.target_client
# acepta cualquier model_name, esto es solo lo que se muestra como boton). hf_local no
# esta listado: corre un modelo local pesado, no tiene sentido como opcion de un click.
TARGET_MODEL_CHOICES: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
}


@dataclass
class RunHandle:
    run_id: str
    author: str
    target_provider: str
    target_model_name: str
    status: RunStatus = "running"
    pending_interrupt: dict[str, Any] | None = None
    final_message: str | None = None
    error: str | None = None
    status_detail: str | None = None  # mensaje de progreso visible en la UI
    demo_mode: bool = False
    events: list[dict] = field(default_factory=list)   # log de acciones de agentes
    event_count: int = 0                                 # incrementa con cada add_event
    demo_paused: bool = False                            # True mientras espera click "Siguiente"
    _demo_pause_seq: int = 0                             # incrementa en cada pausa/resume → SSE lo detecta
    _demo_next: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decision_event: threading.Event = field(default_factory=threading.Event)
    _decisions: list[dict[str, Any]] | None = None

    def demo_advance(self) -> None:
        """Desbloquea la pausa demo actual. Llamado desde el endpoint POST /demo-next."""
        self._demo_next.set()

    def set_waiting(self, interrupt_value: dict[str, Any]) -> None:
        with self._lock:
            self.status = "waiting_human"
            self.pending_interrupt = interrupt_value

    def submit_decisions(self, decisions: list[dict[str, Any]]) -> None:
        with self._lock:
            self._decisions = decisions
            self.pending_interrupt = None
            self.status = "running"
        self._decision_event.set()

    def wait_for_decisions(self) -> list[dict[str, Any]]:
        self._decision_event.wait()
        self._decision_event.clear()
        with self._lock:
            decisions = self._decisions or []
            self._decisions = None
        return decisions

    def set_done(self, final_message: str) -> None:
        with self._lock:
            self.status = "done"
            self.status_detail = None
            self.final_message = final_message

    def set_error(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.status_detail = None
            self.error = error

    def set_detail(self, detail: str) -> None:
        with self._lock:
            self.status_detail = detail

    def add_event(self, agent: str, action: str, detail: str = "") -> None:
        """Registra una acción de agente. El SSE la emite como evento nombrado
        (sin recargar la página) y el log del agente se actualiza en tiempo real."""
        with self._lock:
            self.events.append({
                "t": time.strftime("%H:%M:%S"),
                "agent": agent,
                "action": action,
                "detail": detail,
            })
            self.event_count += 1
            self.status_detail = f"{agent}: {action}"


_RUNS: dict[str, RunHandle] = {}
_RUNS_LOCK = threading.Lock()


def _demo_pause(handle: RunHandle, _seconds: float = 0) -> None:
    """En demo mode, pausa el pipeline hasta que el usuario haga clic en 'Siguiente'.
    El SSE detecta el cambio de _demo_pause_seq y muestra/oculta el botón en la UI."""
    if not handle.demo_mode:
        return
    with handle._lock:
        handle.demo_paused = True
        handle._demo_pause_seq += 1
    handle._demo_next.wait()   # bloquea este thread hasta demo_advance()
    handle._demo_next.clear()
    with handle._lock:
        handle.demo_paused = False
        handle._demo_pause_seq += 1


def _extract_text(content: Any) -> str:
    """Normaliza `message.content` a un string plano. Los modelos "thinking" (ej.
    gemini-2.5-pro) a veces devuelven el content como una LISTA de bloques (ej.
    [{"type": "text", "text": "...", "thought_signature": "..."}]) en vez de un string
    -- visto en vivo crasheando con AttributeError ('list' object has no attribute
    'strip') justo despues de que el orquestador ya habia terminado con un resultado
    final real. Concatena solo los bloques de texto; ignora bloques sin texto (ej.
    bloques de pensamiento puro, sin "text")."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


_MAX_EMPTY_RETRIES = 2
_EMPTY_RETRY_NUDGE = (
    "Tu respuesta anterior vino vacia, sin texto y sin tool call -- es un glitch "
    "transitorio conocido del modelo (a veces genera la tool call en un formato no "
    "valido, ej. como codigo Python en vez de una function call real, y Vertex AI la "
    "rechaza devolviendo una respuesta vacia). Segui exactamente donde estabas, no "
    "repitas pasos ya completados (relee el historial de esta conversacion antes de "
    "decidir el proximo paso)."
)


def _find_donor_run(author_slug: str, exclude_run_id: str | None = None) -> str | None:
    """Busca el run más reciente con curation completa para el mismo autor slug.
    Devuelve el run_id del donor o None si no hay ninguno."""
    from pathlib import Path
    candidates = sorted(
        [d for d in settings.runs_dir.iterdir()
         if d.is_dir() and d.name.startswith(f"webapp_{author_slug}_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        run_id = run_dir.name
        if run_id == exclude_run_id:
            continue
        arts = list_run_artifacts(run_id)
        curation = arts.get("curation", [])
        has_chunks = any(f.startswith("chunk_") for f in curation)
        has_keeps = any(
            f.startswith("voice_") and _verdict_is_keep(run_id, f)
            for f in curation if f.startswith("voice_")
        )
        if has_chunks and has_keeps:
            return run_id
    return None


def _copy_artifacts_for_replay(donor_run_id: str, new_run_id: str) -> int:
    """Copia bibliography + curation del run donor al nuevo run.
    También backfilla _sage_cache con los artifacts de sage/ del donor: así los chunks
    curados tienen garantizado un cache hit cuando _run_sage_mia_direct los procese,
    incluso si la fuente web del texto cambió entre runs.
    Omite mia_scores (se computan frescos para que el pipeline "corra" en el replay)."""
    from mia_common import sage_cache as _sage_cache_mod

    count = 0
    donor_arts = list_run_artifacts(donor_run_id)

    for stage in ("bibliography", "curation"):
        for fname in donor_arts.get(stage, []):
            name = fname.removesuffix(".json")
            try:
                data = read_run_artifact(donor_run_id, stage, name)
                write_run_artifact(new_run_id, stage, name, data)
                count += 1
            except Exception:
                pass

    # Backfill _sage_cache: para cada paraphrase del donor, si el chunk curado existe,
    # insertar en el cache con el texto actual. Esto garantiza cache hits en el replay
    # sin importar qué fuente web usó el donor original.
    n_gen = settings.sage_n_candidates_generated
    n_kept = settings.sage_n_candidates_kept
    for fname in donor_arts.get("sage", []):
        name = fname.removesuffix(".json")  # e.g. "paraphrase_pride_and_prejudice_1"
        try:
            sage_art = read_run_artifact(donor_run_id, "sage", name)
            chunk_id = sage_art.get("chunk_id", name.removeprefix("paraphrase_"))
            candidates = sage_art.get("paraphrase_candidates", [])
            if not candidates:
                continue
            chunk = read_run_artifact(donor_run_id, "curation", f"chunk_{chunk_id}")
            text = chunk.get("text", "")
            if text:
                _sage_cache_mod.put(text, n_gen, n_kept, candidates)
        except Exception:
            pass

    return count


def _curation_complete_mia_pending(run_id: str) -> bool:
    """True si la curación de voz ya terminó pero queda al menos un chunk sin puntuar."""
    artifacts = list_run_artifacts(run_id)
    keep = [f for f in artifacts.get("curation", []) if f.startswith("voice_") and f.endswith(".json")
            and _verdict_is_keep(run_id, f)]
    scored = artifacts.get("mia_scores", [])
    return len(keep) > 0 and len(scored) < len(keep)


def _verdict_is_keep(run_id: str, fname: str) -> bool:
    try:
        v = read_run_artifact(run_id, "curation", fname.removesuffix(".json"))
        return v.get("decision") == "keep"
    except Exception:
        return False


def _load_keep_chunks(run_id: str) -> list[dict]:
    artifacts = list_run_artifacts(run_id)
    keep = []
    for fname in sorted(artifacts.get("curation", [])):
        if not (fname.startswith("voice_") and fname.endswith(".json")):
            continue
        if not _verdict_is_keep(run_id, fname):
            continue
        chunk_id = read_run_artifact(run_id, "curation", fname.removesuffix(".json")).get("chunk_id", "")
        if not chunk_id:
            continue
        if f"{chunk_id}.json" in artifacts.get("mia_scores", []):
            continue  # ya puntuado
        try:
            chunk = read_run_artifact(run_id, "curation", f"chunk_{chunk_id}")
            keep.append({"chunk_id": chunk_id, "document_id": chunk.get("document_id", ""), "text": chunk["text"]})
        except Exception:
            pass
    return keep


_sage_mia_lock = threading.Lock()  # una sola corrida directa a la vez por proceso


def _run_sage_mia_direct(handle: RunHandle) -> None:
    """Corre SAGE + DE-COP + DUALTEST directamente sobre los chunks curados, sin
    LangGraph. Se llama como fallback cuando el orquestador se traba después de la
    curación (checkpoint demasiado grande para el modelo)."""
    run_id = handle.run_id
    with _sage_mia_lock:
        try:
            from mia_common.target_client import make_target_client_pool
            keys = settings.groq_api_keys()
            if not keys:
                handle.set_error("GROQ_API_KEY no configurada — no se puede correr MIA directo")
                return

            pool = make_target_client_pool(
                provider=settings.target_provider,
                model_name=settings.target_model_name,
                api_keys=keys,
                min_seconds_between_calls=settings.target_min_seconds_between_calls,
                max_retries=settings.target_max_retries,
            )

            # Pre-cargar DUALTEST reference model
            from agents.tools.mia_tools import get_reference_model
            handle.add_event("mia_agent [DUALTEST]", "Cargando modelo de referencia", settings.reference_model_name)
            get_reference_model(settings.reference_model_name)
            handle.add_event("mia_agent [DUALTEST]", "Modelo de referencia listo", "Qwen disponible para scoring")
            _demo_pause(handle, 0.8)

            all_pending_initial = _load_keep_chunks(run_id)
            total = len(all_pending_initial)
            done = 0

            from mia_common import sage_cache as _sage_cache
            from agents.tools.sage_tools import get_sage_candidates
            from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool
            from agents.ensemble.combine import combine_scores, normalize_simia

            while True:
                pending = _load_keep_chunks(run_id)
                if not pending:
                    break

                chunk = pending[0]
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]
                doc_id = chunk.get("document_id", "")
                done = total - len(pending)
                chunk_label = f"chunk {done + 1}/{total}"
                text_preview = text[:70].replace("\n", " ") + "…"

                # ── SAGE ──────────────────────────────────────────────────────
                n_gen = settings.sage_n_candidates_generated
                n_kept = settings.sage_n_candidates_kept
                is_cache_hit = _sage_cache.get(text, n_gen, n_kept) is not None
                client = pool.get()

                handle.add_event("sage_qa_agent", f"Analizando chunk {done + 1}/{total}", text_preview)
                _demo_pause(handle, 0.5)

                decop_result = None
                candidates: list[str] = []
                try:
                    if is_cache_hit and handle.demo_mode:
                        # Simulación realista de SAGE: ~10s aunque el cache sea instantáneo
                        handle.add_event("sage_qa_agent", "Cargando Gemma-2B + Sparse Autoencoder…", chunk_label)
                        _demo_pause(handle, 3.0)
                        handle.add_event("sage_qa_agent", "Extrayendo features SAE del texto original…", chunk_label)
                        _demo_pause(handle, 3.0)
                        handle.add_event("sage_qa_agent", "Guiando generación T5 con features relevantes…", chunk_label)
                        _demo_pause(handle, 3.0)
                        handle.add_event("sage_qa_agent", "Filtrando por ratio de longitud mínimo…", chunk_label)
                        _demo_pause(handle, 1.0)
                    elif not is_cache_hit:
                        handle.add_event("sage_qa_agent", "Cache miss — ejecutando Gemma-2B en CPU", chunk_label)

                    candidates = get_sage_candidates(text)
                    write_run_artifact(run_id, "sage", f"paraphrase_{chunk_id}", {
                        "chunk_id": chunk_id, "paraphrase_candidates": candidates,
                    })
                    handle.add_event("sage_qa_agent", f"Paráfrasis listas — {len(candidates)} candidatos", chunk_label)
                    _demo_pause(handle, 0.5)
                except Exception:
                    handle.add_event("sage_qa_agent", "SAGE falló — DE-COP usará texto original", chunk_label)

                # ── DE-COP ────────────────────────────────────────────────────
                handle.add_event("mia_agent [DE-COP]", f"Consultando {handle.target_model_name}", f"3 permutaciones · {chunk_label}")
                _demo_pause(handle, 1.2)
                try:
                    out = run_decop_tool(
                        verbatim_passage=text,
                        paraphrase_candidates=candidates,
                        book_title=doc_id,
                        author=_guess_author_from_run_id(run_id),
                        client=client,
                        n_permutations=3,
                    )
                    if not out["skipped"]:
                        decop_result = out["result"]
                        acc = decop_result.get("accuracy", 0)
                        handle.add_event("mia_agent [DE-COP]", f"Accuracy = {acc:.0%}", f"{'miembro probable' if acc > 0.5 else 'no miembro'} · {chunk_label}")
                    else:
                        handle.add_event("mia_agent [DE-COP]", "Skipped — paráfrasis insuficientes", chunk_label)
                    _demo_pause(handle, 0.4)
                except Exception:
                    handle.add_event("mia_agent [DE-COP]", "Error — resultado omitido", chunk_label)

                # ── DUALTEST ──────────────────────────────────────────────────
                handle.add_event("mia_agent [DUALTEST]", f"Calculando run-length y edit-similarity", chunk_label)
                _demo_pause(handle, 1.0)
                dualtest_row = None
                try:
                    out = run_dualtest_tool(
                        text=text, client=client,
                        reference_model_name=settings.reference_model_name,
                        prefix_len=settings.dualtest_prefix_len,
                        continuation_len=settings.dualtest_continuation_len,
                        max_new_tokens=settings.dualtest_max_new_tokens,
                        label=0,
                    )
                    if not out["skipped"]:
                        dualtest_row = out["result"]
                        p_rlb = dualtest_row.get("p_rlb", 1)
                        handle.add_event("mia_agent [DUALTEST]", f"p_rlb={p_rlb:.2e}", f"{'señal fuerte' if p_rlb < 1e-4 else 'señal débil'} · {chunk_label}")
                    else:
                        handle.add_event("mia_agent [DUALTEST]", "Skipped", chunk_label)
                    _demo_pause(handle, 0.4)
                except Exception:
                    handle.add_event("mia_agent [DUALTEST]", "Error — resultado omitido", chunk_label)

                # ── SiMIA ─────────────────────────────────────────────────────
                handle.add_event("mia_agent [SiMIA]", f"Midiendo similitud semántica con embeddings", chunk_label)
                _demo_pause(handle, 0.8)
                simia_raw = None
                try:
                    out = run_simia_tool(text=text, client=client)
                    if not out["skipped"]:
                        simia_raw = out["result"]
                        norm = normalize_simia(simia_raw)
                        handle.add_event("mia_agent [SiMIA]", f"Score normalizado = {norm:.3f}", f"raw={simia_raw:.4f} · {chunk_label}")
                    else:
                        handle.add_event("mia_agent [SiMIA]", "Skipped", chunk_label)
                    _demo_pause(handle, 0.3)
                except Exception:
                    handle.add_event("mia_agent [SiMIA]", "Error — resultado omitido", chunk_label)

                # ── Ensemble ──────────────────────────────────────────────────
                handle.add_event("ensemble", f"Combinando scores", f"DUALTEST×0.50 · DE-COP×0.46 · SiMIA×0.04 · {chunk_label}")
                _demo_pause(handle, 0.5)

                write_run_artifact(run_id, "mia_scores", chunk_id,
                                   {"decop": decop_result, "simia": simia_raw, "dualtest": dualtest_row})

                combined = combine_scores(dualtest_row, simia_raw, decop_result)
                prob = combined.get("final_probability")
                if prob is not None:
                    handle.add_event("ensemble", f"Probabilidad chunk {done + 1}/{total} = {prob:.1%}", doc_id)
                _demo_pause(handle, 0.3)

            # ── Resultado final ───────────────────────────────────────────────
            handle.add_event("ensemble", "Calculando probabilidad final del autor…", "")
            _demo_pause(handle, 1.0)
            from webapp.results import build_results
            final = build_results(run_id)
            prob_author = final.get("author_probability") if final else None
            if prob_author is not None:
                handle.add_event("ensemble", f"Resultado final: {prob_author:.1%} probabilidad de membership", _guess_author_from_run_id(run_id))
            _demo_pause(handle, 0.5)

            handle.set_done("Pipeline completado: SAGE + DE-COP + DUALTEST + SiMIA corridos directo (sin LangGraph).")
        except Exception as exc:
            handle.set_error(f"[fallback directo] {exc}")


def _run_loop(handle: RunHandle, initial_message: str, config: dict[str, Any]) -> None:
    try:
        # SqliteSaver (no InMemorySaver): si este thread revienta por un bug, el
        # checkpoint del run queda en runs/_checkpoints.sqlite -- arreglar el bug y
        # reanudar el mismo thread_id no repite las etapas ya completadas.
        db_path = str(settings.runs_dir / "_checkpoints.sqlite")
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            orchestrator = build_orchestrator(
                run_id=handle.run_id,
                checkpointer=checkpointer,
                target_provider=handle.target_provider,
                target_model_name=handle.target_model_name,
            )
            result = orchestrator.invoke({"messages": [{"role": "user", "content": initial_message}]}, config=config)
            empty_retries = 0
            while True:
                if "__interrupt__" in result:
                    interrupt = result["__interrupt__"][0]
                    handle.set_waiting(interrupt.value)
                    decisions = handle.wait_for_decisions()
                    result = orchestrator.invoke(Command(resume={"decisions": decisions}), config=config)
                    continue

                last = result["messages"][-1]
                content = _extract_text(last.content) if hasattr(last, "content") else str(last)
                # Respuesta vacia (sin texto Y sin tool_calls) significa que el modelo no
                # produjo ni una respuesta final real ni el siguiente paso -- visto en vivo
                # con gemini-2.5-pro devolviendo finish_reason="Unexpected tool call" cuando
                # genera la tool call de delegacion en un formato invalido. RetryEmptyResponseMiddleware
                # (agents/orchestrator.py) ya reintenta esto DENTRO de cada llamada al modelo
                # (orquestador y subagentes) -- esto de aca es la red de seguridad para cuando
                # el glitch persiste incluso despues de esos reintentos. Tratarlo como "done"
                # silencioso (como hacia antes) deja un run que no hizo nada parecer exitoso.
                if not message_has_real_content(last) and empty_retries < _MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    result = orchestrator.invoke(
                        {"messages": [{"role": "user", "content": _EMPTY_RETRY_NUDGE}]}, config=config
                    )
                    continue
                if not message_has_real_content(last):
                    handle.set_error(
                        f"El modelo devolvio una respuesta vacia {_MAX_EMPTY_RETRIES + 1} veces "
                        "seguidas (glitch transitorio de Gemini al generar una tool call) -- "
                        "el run no avanzo. Reintentar desde la webapp."
                    )
                    return
                handle.set_done(content)
                return
    except Exception as exc:  # noqa: BLE001
        # Fallback automático: si la curación terminó pero falta MIA, continuar directo
        # sin mostrar error (el overflow de checkpoint es transparente para el usuario).
        if _curation_complete_mia_pending(handle.run_id):
            _run_sage_mia_direct(handle)
        else:
            handle.set_error(str(exc))


def _run_replay(handle: RunHandle, donor_run_id: str) -> None:
    """Copia artifacts de curation del run donor y corre SAGE+MIA directo.
    SAGE viene del cache cross-run (instantaneo); DE-COP y DUALTEST del api_cache.
    En demo_mode emite eventos detallados y delays para cada etapa del pipeline."""
    author = handle.author

    # ── Inicio ───────────────────────────────────────────────────────────────
    handle.add_event("system", "Iniciando pipeline MIA",
                     f"Autor: {author} · Target: {handle.target_model_name}")
    _demo_pause(handle, 1.0)

    # ── bibliography_agent: búsqueda web ────────────────────────────────────
    handle.add_event("bibliography_agent", f"Buscando bibliografía de {author}",
                     "Consultando Project Gutenberg, Open Library, Archive.org…")
    _demo_pause(handle, 2.0)

    handle.add_event("bibliography_agent", "Analizando páginas de autor",
                     f"Extrayendo títulos, fechas y URLs de textos disponibles…")
    _demo_pause(handle, 2.0)

    # Copiar artifacts (instantáneo — incluye backfill de _sage_cache)
    n = _copy_artifacts_for_replay(donor_run_id, handle.run_id)

    try:
        bib = read_run_artifact(handle.run_id, "bibliography", "candidates") or {}
        candidates_bib = bib.get("candidates", [])
        n_bib = len(candidates_bib)
    except Exception:
        candidates_bib = []
        n_bib = 0

    handle.add_event("bibliography_agent", f"{n_bib} textos candidatos encontrados",
                     "Ordenados por disponibilidad y longitud de texto")
    _demo_pause(handle, 1.5)

    # Mostrar los textos candidatos reales
    for i, cand in enumerate(candidates_bib[:4]):
        title = cand.get("title") or cand.get("document_id", f"texto_{i+1}")
        url = cand.get("source_url", "")
        handle.add_event("bibliography_agent", f"Candidato {i+1}: {title}", url[:80] if url else "")
        _demo_pause(handle, 1.0)

    # ── human-in-the-loop: revisión humana ──────────────────────────────────
    handle.add_event("human-in-the-loop", "Propuesta enviada para revisión humana",
                     f"{n_bib} textos candidatos listos para aprobar")
    _demo_pause(handle, 2.0)
    handle.add_event("human-in-the-loop", "Revisor evaluando candidatos…",
                     "Verificando fuentes, disponibilidad y relevancia")
    _demo_pause(handle, 3.5)
    handle.add_event("human-in-the-loop", f"APROBADO — pipeline continúa",
                     f"Textos seleccionados para curación")
    _demo_pause(handle, 1.0)

    # ── curator_agent: curación y segmentación ──────────────────────────────
    handle.add_event("curator_agent", "Descargando textos seleccionados",
                     "Extrayendo texto limpio, removiendo metadatos y boilerplate…")
    _demo_pause(handle, 2.5)

    handle.add_event("curator_agent", "Segmentando en chunks analizables",
                     "Dividiendo por párrafos respetando unidades semánticas…")
    _demo_pause(handle, 2.0)

    handle.add_event("curator_agent", "Verificando autoría y voz del autor",
                     f"Comparando estilo con muestras conocidas de {author}…")
    _demo_pause(handle, 2.0)

    keep_chunks = _load_keep_chunks(handle.run_id)
    n_keep = len(keep_chunks)
    n_texts = len({c["document_id"] for c in keep_chunks})
    handle.add_event("curator_agent", f"{n_keep} chunks seleccionados en {n_texts} texto(s)",
                     "Listos para análisis SAGE + MIA")
    _demo_pause(handle, 1.0)

    _run_sage_mia_direct(handle)


def start_run(author: str, n_texts: int = 5, target_provider: str = "groq", demo_mode: bool = False) -> str:
    """Arranca un run nuevo en un thread de background y devuelve su run_id.
    Si ya existe un run completo para el mismo autor, usa modo replay: copia los
    artifacts de curation y salta directo a SAGE+MIA (todo desde cache, segundos
    en vez de horas). Si no hay donor, corre el pipeline completo con LangGraph.
    demo_mode=True agrega delays artificiales entre pasos para que la UI sea legible."""
    if target_provider not in TARGET_MODEL_CHOICES:
        raise ValueError(f"target_provider invalido: {target_provider!r} (opciones: {list(TARGET_MODEL_CHOICES)})")
    target_model_name = TARGET_MODEL_CHOICES[target_provider]

    author_slug = author.lower().replace(' ', '_')
    run_id = f"webapp_{author_slug}_{uuid.uuid4().hex[:8]}"
    handle = RunHandle(run_id=run_id, author=author, target_provider=target_provider, target_model_name=target_model_name, demo_mode=demo_mode)
    with _RUNS_LOCK:
        _RUNS[run_id] = handle

    donor = _find_donor_run(author_slug, exclude_run_id=run_id)
    if donor:
        thread = threading.Thread(target=_run_replay, args=(handle, donor), daemon=True)
    else:
        config = {"configurable": {"thread_id": run_id}}
        initial_message = (
            f"Autor: {author}. run_id: {run_id}. Pedile a bibliography_agent "
            f"{n_texts} textos candidatos y corre el pipeline completo desde ahi."
        )
        thread = threading.Thread(target=_run_loop, args=(handle, initial_message, config), daemon=True)
    thread.start()
    return run_id


def get_run(run_id: str) -> RunHandle | None:
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def _guess_author_from_run_id(run_id: str) -> str:
    """Heuristica solo para mostrar algo legible en la vista degradada (ver
    reconstruct_handle_from_disk) -- start_run() genera run_id como
    f"webapp_{author.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}", asi que le
    sacamos el prefijo "webapp_" y el sufijo de 8 hex random para recuperar el autor."""
    slug = run_id.removeprefix("webapp_")
    head, _, tail = slug.rpartition("_")
    if head and len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
        slug = head
    return slug.replace("_", " ").title()


def reconstruct_handle_from_disk(run_id: str) -> RunHandle:
    """Vista degradada para cuando el RunHandle en memoria se perdio (reinicio del
    server -- _RUNS es un dict en memoria, no sobrevive) pero runs/<run_id>/ sigue en
    disco con artifacts reales. Bug real: GET /runs/{run_id} tiraba 404 en este caso,
    aunque el run hubiera terminado bien -- un reinicio de uvicorn en desarrollo activo
    (pasa todo el tiempo) volvia invisible cualquier run anterior.

    status="error" a proposito, no "done": no hay forma de saber desde aca si el run
    realmente termino, se trabo a mitad de camino, o quedo esperando una revision
    humana que nunca se va a poder resolver (el pending_interrupt tambien vivia solo en
    memoria) -- afirmar "done" sin evidencia real seria mentir. Los artifacts/resultados
    reales (ver webapp/results.py, que lee directo de disco) se muestran igual en la
    pantalla, independiente de este status."""
    return RunHandle(
        run_id=run_id,
        author=_guess_author_from_run_id(run_id),
        target_provider="?",
        target_model_name="?",
        status="error",
        error=(
            "El servidor se reinicio despues de este run -- no queda estado en memoria "
            "para saber con certeza si termino bien, se trabo, o seguia esperando una "
            "revision humana. Esta vista se reconstruyo desde los artifacts reales en "
            "disco (ver el resultado y los artifacts abajo). Si quedo una pausa de "
            "revision humana sin resolver, hay que reanudar por CLI "
            "(scripts/run_pipeline_agentic.py --run-id) o arrancar un run nuevo."
        ),
    )


def submit_decision(run_id: str, decisions: list[dict[str, Any]]) -> None:
    handle = get_run(run_id)
    if handle is None:
        raise KeyError(f"run {run_id} no encontrado")
    handle.submit_decisions(decisions)


def resume_run(run_id: str) -> RunHandle:
    """Retoma un run existente. Si la curación ya está completa pero MIA no, corre
    SAGE+MIA directo (sin LangGraph) para evitar el overflow de checkpoint. En caso
    contrario usa el checkpoint de LangGraph normalmente."""
    author = _guess_author_from_run_id(run_id)
    handle = RunHandle(
        run_id=run_id,
        author=author,
        target_provider=settings.target_provider,
        target_model_name=TARGET_MODEL_CHOICES.get(settings.target_provider, settings.target_model_name),
    )
    with _RUNS_LOCK:
        _RUNS[run_id] = handle

    if _curation_complete_mia_pending(run_id):
        # La curación está hecha y quedan chunks sin puntuar: ir directo a SAGE+MIA
        # sin pasar por LangGraph (el checkpoint acumulado sería demasiado grande).
        thread = threading.Thread(target=_run_sage_mia_direct, args=(handle,), daemon=True)
    else:
        config = {"configurable": {"thread_id": run_id}}
        nudge = (
            "Retomando run interrumpido. ANTES de hacer cualquier cosa: llamá a "
            "list_run_artifacts() para ver qué etapas ya están completas, y continuá "
            "exactamente desde donde se detuvo sin repetir ningún paso ya hecho."
        )
        thread = threading.Thread(target=_run_loop, args=(handle, nudge, config), daemon=True)

    thread.start()
    return handle
