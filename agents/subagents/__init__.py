from agents.subagents.bibliography_agent import bibliography_subagent
from agents.subagents.curator_agent import curator_subagent
from agents.subagents.flow_checker_agent import flow_checker_subagent
from agents.subagents.mia_agent import build_mia_subagent
from agents.subagents.sage_qa_agent import sage_qa_subagent

# mia_agent no esta aca -- necesita un TargetClient real bindeado por run via
# build_mia_subagent(client, ...), ver agents/orchestrator.py y el docstring de
# agents/subagents/mia_agent.py para el porque.
STATIC_SUBAGENTS = [
    bibliography_subagent,
    curator_subagent,
    sage_qa_subagent,
    flow_checker_subagent,
]

__all__ = [
    "bibliography_subagent",
    "curator_subagent",
    "sage_qa_subagent",
    "flow_checker_subagent",
    "build_mia_subagent",
    "STATIC_SUBAGENTS",
]
