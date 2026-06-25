"""
orchestrator.py — el agente orquestador de la Fase 2: create_deep_agent con los 5
subagentes (agents/subagents/) + sus propios tools deterministicos (ensemble,
artifacts). Simula la presencia de un humano en el flujo: pausa para revision humana
(bibliography_agent -> propose_candidate_texts), e invoca flow_checker_agent entre
etapas para detectar anomalias antes de seguir.

No tiene todavia una skill persistente que aprenda entre runs (eso es Fase 3) -- el
parametro `skills` queda vacio a proposito, listo para sumarse despues sin tocar el
resto de esta funcion.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.ensemble.combine import aggregate_chunk_scores, aggregate_text_scores, combine_scores
from agents.subagents import ALL_SUBAGENTS
from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact
from mia_common.settings import settings

ORCHESTRATOR_SYSTEM_PROMPT = """Sos el orquestador del pipeline de Membership
Inference Attack. Dado el nombre de un autor, tu trabajo es coordinar todo el flujo
hasta producir una probabilidad final de que sus textos hayan estado en el training
set del modelo target, con desglose por metodo.

Etapas, en orden:
1. Delega a bibliography_agent (via la tool task) para encontrar y proponer textos
   candidatos del autor. Esto SIEMPRE pausa para revision humana -- no hay nada que
   hacer de tu parte ahi mas que esperar la confirmacion.
2. Una vez confirmados los candidatos, delega a curator_agent para limpiar, chunkear,
   verificar autoria, y seleccionar los chunks mas caracteristicos de la voz del autor.
3. Para cada chunk que sobrevivio la curacion: delega a sage_qa_agent (parafraseo +
   QA), y despues a mia_agent (DE-COP/SiMIA/DUALTEST) con los paraphrase candidates que
   produjo SAGE.
4. Llama a combine_scores con los 3 resultados crudos de mia_agent para ese chunk.
5. Despues de CADA etapa (1-4), delega a flow_checker_agent pasandole el run_id y el
   nombre de la etapa que acaba de terminar, para que valide que los artifacts tienen
   sentido. Si te devuelve recommended_action="escalate_to_human", parate y avisale al
   usuario en vez de seguir. Si es "retry_stage", reintenta esa etapa una vez antes de
   seguir. "skip_item" o "continue" significan seguir normalmente.
6. Cuando termines todos los chunks de un texto, llama a aggregate_chunk_scores. Cuando
   termines todos los textos del autor, llama a aggregate_text_scores para la
   probabilidad final.
7. Usa write_run_artifact/read_run_artifact/list_run_artifacts para tu propio
   debugging -- todo lo que pasa en el run queda en runs/<run_id>/, no dependas de tu
   propia memoria de la conversacion para saber que se hizo.

El mensaje inicial del usuario te va a dar un run_id ya generado -- usalo de forma
consistente en TODAS las llamadas a tools de este run (no generes uno propio)."""


def build_orchestrator(
    checkpointer: BaseCheckpointSaver | None = None,
    agent_model: str | None = None,
) -> Any:
    """Devuelve el grafo compilado del orquestador, listo para .invoke()/.stream().
    `checkpointer` es obligatorio si se va a usar el human-in-the-loop de
    bibliography_agent (sin el, los interrupts no pueden persistir estado)."""
    return create_deep_agent(
        model=agent_model or settings.agent_model,
        tools=[combine_scores, aggregate_chunk_scores, aggregate_text_scores,
               write_run_artifact, read_run_artifact, list_run_artifacts],
        subagents=ALL_SUBAGENTS,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
