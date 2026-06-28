from agents.subagents.bibliography_agent import build_bibliography_subagent
from agents.subagents.curator_agent import build_curator_subagent
from agents.subagents.flow_checker_agent import build_flow_checker_subagent
from agents.subagents.mia_agent import build_mia_subagent
from agents.subagents.sage_qa_agent import build_sage_qa_subagent


def build_static_subagents(run_id: str) -> list[dict]:
    """Los 4 subagentes que no necesitan nada mas que `run_id` bindeado (mia_agent
    queda afuera -- necesita un TargetClient real bindeado por run via
    build_mia_subagent(run_id, client, ...), ver agents/orchestrator.py y el docstring
    de agents/subagents/mia_agent.py para el porque)."""
    return [
        build_bibliography_subagent(run_id),
        build_curator_subagent(run_id),
        build_sage_qa_subagent(run_id),
        build_flow_checker_subagent(run_id),
    ]


__all__ = [
    "build_bibliography_subagent",
    "build_curator_subagent",
    "build_sage_qa_subagent",
    "build_flow_checker_subagent",
    "build_mia_subagent",
    "build_static_subagents",
]
