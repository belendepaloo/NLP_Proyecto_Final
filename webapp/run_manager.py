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
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decision_event: threading.Event = field(default_factory=threading.Event)
    _decisions: list[dict[str, Any]] | None = None

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


_RUNS: dict[str, RunHandle] = {}
_RUNS_LOCK = threading.Lock()


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
    Omite mia_scores y sage (se computan frescos, aunque SAGE viene del cache)."""
    count = 0
    for stage in ("bibliography", "curation"):
        for fname in list_run_artifacts(donor_run_id).get(stage, []):
            name = fname.removesuffix(".json")
            try:
                data = read_run_artifact(donor_run_id, stage, name)
                write_run_artifact(new_run_id, stage, name, data)
                count += 1
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
            get_reference_model(settings.reference_model_name)

            all_pending_initial = _load_keep_chunks(run_id)
            total = len(all_pending_initial)
            done = 0

            while True:
                pending = _load_keep_chunks(run_id)
                if not pending:
                    break

                chunk = pending[0]
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]
                done = total - len(pending)
                handle.set_detail(f"SAGE chunk {done + 1}/{total} — {chunk_id}")
                client = pool.get()

                # SAGE (con cache cross-run: si ya fue procesado, no recarga Gemma-2B)
                decop_result = None
                candidates: list[str] = []
                try:
                    from agents.tools.sage_tools import get_sage_candidates
                    candidates = get_sage_candidates(text)
                    write_run_artifact(run_id, "sage", f"paraphrase_{chunk_id}", {
                        "chunk_id": chunk_id, "paraphrase_candidates": candidates,
                    })
                except Exception:
                    pass  # SAGE fallo, DE-COP corre sin candidates (será skipped)

                # DE-COP (necesita >=3 candidates de SAGE)
                try:
                    from agents.tools.mia_tools import run_decop_tool
                    out = run_decop_tool(
                        verbatim_passage=text,
                        paraphrase_candidates=candidates,
                        book_title=chunk.get("document_id", "unknown"),
                        author=_guess_author_from_run_id(run_id),
                        client=client,
                        n_permutations=3,
                    )
                    if not out["skipped"]:
                        decop_result = out["result"]
                except Exception:
                    pass

                # DUALTEST
                dualtest_row = None
                try:
                    from agents.tools.mia_tools import run_dualtest_tool
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
                except Exception:
                    pass

                # SiMIA
                simia_raw = None
                try:
                    from agents.tools.mia_tools import run_simia_tool
                    out = run_simia_tool(text=text, client=client)
                    if not out["skipped"]:
                        simia_raw = out["result"]
                except Exception:
                    pass

                write_run_artifact(run_id, "mia_scores", chunk_id,
                                   {"decop": decop_result, "simia": simia_raw, "dualtest": dualtest_row})

            handle.set_done("Pipeline completado: SAGE + DE-COP + DUALTEST corridos directo (sin LangGraph).")
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
    SAGE viene del cache cross-run (instantaneo); DE-COP y DUALTEST del api_cache."""
    handle.set_detail(f"Replay: copiando artifacts de {donor_run_id}…")
    n = _copy_artifacts_for_replay(donor_run_id, handle.run_id)
    handle.set_detail(f"Replay: {n} artifacts copiados — arrancando SAGE+MIA")
    _run_sage_mia_direct(handle)


def start_run(author: str, n_texts: int = 5, target_provider: str = "groq") -> str:
    """Arranca un run nuevo en un thread de background y devuelve su run_id.
    Si ya existe un run completo para el mismo autor, usa modo replay: copia los
    artifacts de curation y salta directo a SAGE+MIA (todo desde cache, segundos
    en vez de horas). Si no hay donor, corre el pipeline completo con LangGraph."""
    if target_provider not in TARGET_MODEL_CHOICES:
        raise ValueError(f"target_provider invalido: {target_provider!r} (opciones: {list(TARGET_MODEL_CHOICES)})")
    target_model_name = TARGET_MODEL_CHOICES[target_provider]

    author_slug = author.lower().replace(' ', '_')
    run_id = f"webapp_{author_slug}_{uuid.uuid4().hex[:8]}"
    handle = RunHandle(run_id=run_id, author=author, target_provider=target_provider, target_model_name=target_model_name)
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
