#!/usr/bin/env python3
"""
run_pipeline_agentic.py — milestone demostrable de la Fase 2: el orquestador
deepagents corriendo de punta a punta, con la pausa de revision humana (bibliography_agent)
manejada por stdin en vez de una pantalla web (eso es Fase 4).

Uso:
    python scripts/run_pipeline_agentic.py --author "Charles Dickens"
    python scripts/run_pipeline_agentic.py --run-id agentic_charles_dickens_abc123   # retomar

Necesita (ver .env.example): GROQ_API_KEY (target), y un modelo de agente que maneje
bien tool-calling anidado -- default google_vertexai:gemini-2.5-pro (necesita
GOOGLE_CLOUD_PROJECT + `gcloud auth application-default login`, ver
mia_common/settings.py) o ANTHROPIC_API_KEY (pasar --agent-model "anthropic:claude-...").

ADVERTENCIAS verificadas en vivo: Groq (`llama-3.3-70b-versatile`) como agent_model le
generaba argumentos mal formados a la tool `task` -- no usar Groq para este rol, solo
como target. `gemini-2.5-flash` via Vertex tambien mostro fallas intermitentes
(`MALFORMED_FUNCTION_CALL` en `write_todos`) -- `gemini-2.5-pro` no mostro este
problema en las mismas pruebas, por eso es el default.

El checkpointer es persistente (SQLite, runs/_checkpoints.sqlite) -- si el run crashea
por un bug, arreglar el bug y correr de nuevo con --run-id <el mismo run_id> retoma
desde el ultimo paso completado en vez de repetir todo desde el principio.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.orchestrator import build_orchestrator  # noqa: E402
from mia_common.settings import settings  # noqa: E402


def print_interrupt(interrupt) -> None:
    value = interrupt.value
    print("\n" + "=" * 70)
    print("PAUSA -- revision humana requerida")
    print("=" * 70)
    for req in value.get("action_requests", []):
        print(f"\nTool: {req['name']}")
        print(f"Args: {req['args']}")
        if req.get("description"):
            print(f"({req['description']})")


def ask_decision(action_request: dict) -> dict:
    print("\nDecision: [a]probar / [e]ditar / [r]echazar?")
    choice = input("> ").strip().lower()
    if choice.startswith("e"):
        print(f"Args actuales: {action_request['args']}")
        print("Pegar los args editados como JSON (o Enter para no cambiar nada):")
        raw = input("> ").strip()
        if raw:
            import json

            edited_args = json.loads(raw)
        else:
            edited_args = action_request["args"]
        return {
            "type": "edit",
            "edited_action": {"name": action_request["name"], "args": edited_args},
        }
    if choice.startswith("r"):
        message = input("Mensaje de rechazo para el agente: ").strip()
        return {"type": "reject", "message": message or "Rechazado por el usuario."}
    return {"type": "approve"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True, help='Ej: --author "Charles Dickens"')
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Override de mia_common.settings.agent_model (ver advertencia en el docstring del archivo).",
    )
    parser.add_argument("--n-texts", type=int, default=5, help="Cuantos textos pedirle a bibliography_agent.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Retomar un run existente (mismo thread_id) en vez de arrancar uno nuevo -- "
        "el checkpointer es persistente (SQLite, runs/_checkpoints.sqlite), asi que esto "
        "funciona incluso si el run anterior crasheo por un bug ya arreglado. No hace "
        "falta repetir bibliography_agent/curacion ya completados en ese run.",
    )
    parser.add_argument(
        "--target-provider",
        default=None,
        help="Override de mia_common.settings.target_provider (groq|openai|anthropic|google|hf_local).",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Override de mia_common.settings.target_model_name.",
    )
    args = parser.parse_args()

    # run_id se resuelve ANTES de build_orchestrator() -- se bindea via closure en las
    # tools del orquestador y de cada subagente (ver agents/orchestrator.py), asi que
    # tiene que existir antes de construir el grafo, no despues.
    run_id = args.run_id or f"agentic_{args.author.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": run_id}}

    db_path = str(settings.runs_dir / "_checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        orchestrator = build_orchestrator(
            run_id=run_id,
            checkpointer=checkpointer,
            agent_model=args.agent_model,
            target_provider=args.target_provider,
            target_model_name=args.target_model,
        )

        if args.run_id:
            print(f"=== Retomando run {run_id} (checkpoint persistente) ===")
            result = orchestrator.invoke(None, config=config)
        else:
            initial_message = (
                f"Autor: {args.author}. run_id: {run_id}. Pedile a bibliography_agent "
                f"{args.n_texts} textos candidatos y corre el pipeline completo desde ahi."
            )
            print(f"=== Arrancando run {run_id} ===")
            result = orchestrator.invoke({"messages": [{"role": "user", "content": initial_message}]}, config=config)

        while "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]
            print_interrupt(interrupt)
            decisions = [ask_decision(req) for req in interrupt.value.get("action_requests", [])]
            result = orchestrator.invoke(Command(resume={"decisions": decisions}), config=config)

        print("\n" + "=" * 70)
        print("Resultado final")
        print("=" * 70)
        last_message = result["messages"][-1]
        print(last_message.content if hasattr(last_message, "content") else last_message)
        print(f"\n(artifacts completos en runs/{run_id}/ -- run_id: {run_id} si hace falta retomar con --run-id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
