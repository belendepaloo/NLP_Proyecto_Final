#!/usr/bin/env python3
"""
run_pipeline_agentic.py — milestone demostrable de la Fase 2: el orquestador
deepagents corriendo de punta a punta, con la pausa de revision humana (bibliography_agent)
manejada por stdin en vez de una pantalla web (eso es Fase 4).

Uso:
    python scripts/run_pipeline_agentic.py --author "Charles Dickens"

Necesita (ver .env.example): GROQ_API_KEY (target), y un modelo de agente que maneje
bien tool-calling anidado -- GOOGLE_API_KEY (Gemini, default) o ANTHROPIC_API_KEY
(pasar --agent-model "anthropic:claude-...") o, si no hay otra, un modelo de Groq con
--agent-model "groq:..." (ver advertencia abajo).

ADVERTENCIA verificada en esta sesion: probamos con Groq llama-3.3-70b-versatile como
agent_model y la tool `task` (la que deepagents usa para delegar a subagentes) le
generaba argumentos mal formados -- ese modelo no maneja con suficiente fidelidad el
tool-calling anidado de deepagents. Funciona bien para el modelo TARGET (lo que se
testea con MIA), pero no se confirmo que funcione para el rol de agente razonador.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.orchestrator import build_orchestrator  # noqa: E402


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
    args = parser.parse_args()

    checkpointer = InMemorySaver()
    orchestrator = build_orchestrator(checkpointer=checkpointer, agent_model=args.agent_model)

    run_id = f"agentic_{args.author.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": run_id}}

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
    print(f"\n(artifacts completos en runs/{run_id}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
