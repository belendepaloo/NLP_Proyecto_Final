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

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents.orchestrator import build_orchestrator
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
            self.final_message = final_message

    def set_error(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = error


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
                has_tool_calls = bool(getattr(last, "tool_calls", None))
                # Respuesta vacia (sin texto Y sin tool_calls) significa que el modelo no
                # produjo ni una respuesta final real ni el siguiente paso -- visto en vivo
                # con gemini-2.5-pro devolviendo finish_reason="Unexpected tool call" cuando
                # genera la tool call de delegacion en un formato invalido. Tratarlo como
                # "done" silencioso (como hacia antes) deja un run que no hizo nada parecer
                # exitoso. Reintentar empujando al modelo, acotado, antes de rendirse.
                if not content.strip() and not has_tool_calls and empty_retries < _MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    result = orchestrator.invoke(
                        {"messages": [{"role": "user", "content": _EMPTY_RETRY_NUDGE}]}, config=config
                    )
                    continue
                if not content.strip() and not has_tool_calls:
                    handle.set_error(
                        f"El modelo devolvio una respuesta vacia {_MAX_EMPTY_RETRIES + 1} veces "
                        "seguidas (glitch transitorio de Gemini al generar una tool call) -- "
                        "el run no avanzo. Reintentar desde la webapp."
                    )
                    return
                handle.set_done(content)
                return
    except Exception as exc:  # noqa: BLE001 -- se muestra en la pantalla del run, no debe tumbar el thread en silencio
        handle.set_error(str(exc))


def start_run(author: str, n_texts: int = 5, target_provider: str = "groq") -> str:
    """Arranca un run nuevo en un thread de background y devuelve su run_id.
    `target_provider` decide que modelo "black box" se testea en este run (ver
    TARGET_MODEL_CHOICES) -- cada run construye su PROPIO orquestador (no hay un
    orquestador global compartido) precisamente porque mia_agent necesita un
    TargetClient bindeado a la eleccion de ESTE run."""
    if target_provider not in TARGET_MODEL_CHOICES:
        raise ValueError(f"target_provider invalido: {target_provider!r} (opciones: {list(TARGET_MODEL_CHOICES)})")
    target_model_name = TARGET_MODEL_CHOICES[target_provider]

    run_id = f"webapp_{author.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    handle = RunHandle(run_id=run_id, author=author, target_provider=target_provider, target_model_name=target_model_name)
    with _RUNS_LOCK:
        _RUNS[run_id] = handle

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


def submit_decision(run_id: str, decisions: list[dict[str, Any]]) -> None:
    handle = get_run(run_id)
    if handle is None:
        raise KeyError(f"run {run_id} no encontrado")
    handle.submit_decisions(decisions)
