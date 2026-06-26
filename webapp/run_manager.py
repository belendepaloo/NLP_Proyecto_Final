"""
run_manager.py — puente entre el orquestador deepagents (sincrono, bloqueante,
maneja human-in-the-loop con Command(resume=...)) y FastAPI (async, request/response).

Un RunHandle por run vive en RUNS mientras el proceso de uvicorn este arriba -- no hay
persistencia entre reinicios del server, igual que scripts/run_pipeline_agentic.py (que
tambien usa InMemorySaver). El estado "real" del run (lo que se decidio en cada etapa)
sigue viviendo en runs/<run_id>/ via fs_tools, no aca.

El orquestador corre en un thread de background por run; la vuelta de un humano
(aprobar/editar/rechazar) se entrega via submit_decisions(), que despierta ese thread.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.orchestrator import build_orchestrator

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


def _run_loop(handle: RunHandle, initial_message: str, config: dict[str, Any]) -> None:
    try:
        orchestrator = build_orchestrator(
            checkpointer=InMemorySaver(),
            target_provider=handle.target_provider,
            target_model_name=handle.target_model_name,
        )
        result = orchestrator.invoke({"messages": [{"role": "user", "content": initial_message}]}, config=config)
        while "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]
            handle.set_waiting(interrupt.value)
            decisions = handle.wait_for_decisions()
            result = orchestrator.invoke(Command(resume={"decisions": decisions}), config=config)
        last = result["messages"][-1]
        handle.set_done(last.content if hasattr(last, "content") else str(last))
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
