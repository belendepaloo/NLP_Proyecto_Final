"""
orchestrator.py — el agente orquestador de la Fase 2: create_deep_agent con los 5
subagentes (agents/subagents/) + sus propios tools deterministicos (ensemble,
artifacts). Simula la presencia de un humano en el flujo: pausa para revision humana
(bibliography_agent -> propose_candidate_texts), e invoca flow_checker_agent entre
etapas para detectar anomalias antes de seguir.

Fase 3 (skill persistente): el backend de filesystem que se le pasa a
create_deep_agent esta scopeado a agents/skills/ (root_dir=settings.skill_dir.parent,
virtual_mode=True) -- el orquestador y los subagentes ganan read_file/ls ahi (y solo
ahi, no sobre el resto del repo) via la SkillsMiddleware de deepagents, mas
record_learning/record_calibration para escribir. runs/<run_id>/ sigue usando sus
propios tools dedicados (fs_tools.write_run_artifact, etc.), no este backend.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.ensemble.combine import aggregate_chunk_scores, aggregate_text_scores, combine_scores
from agents.subagents import STATIC_SUBAGENTS, build_mia_subagent
from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact
from agents.tools.skill_tools import record_calibration, record_learning
from mia_common.settings import settings
from mia_common.target_client import resolve_target_client

ORCHESTRATOR_SYSTEM_PROMPT = """Sos el orquestador del pipeline de Membership
Inference Attack. Dado el nombre de un autor, tu trabajo es coordinar todo el flujo
hasta producir una probabilidad final de que sus textos hayan estado en el training
set del modelo target, con desglose por metodo.

Etapas, en orden:
0. Antes de delegar nada: tenes una skill ("pipeline-learnings") con el historial de
   bugs/calibraciones de runs anteriores -- usa read_file para leer
   /pipeline-learnings/SKILL.md primero (instrucciones completas de como usarla), y
   despues /pipeline-learnings/learnings.jsonl y /pipeline-learnings/calibration_history.csv.
   Tenelos en cuenta durante todo el run (ej. que metodos son menos confiables, que
   modelos NO usar como agent_model si en algun momento delegaras eso).
1. Delega a bibliography_agent (via la tool task) para encontrar y proponer textos
   candidatos del autor. Esto SIEMPRE pausa para revision humana -- no hay nada que
   hacer de tu parte ahi mas que esperar la confirmacion. ANTES de pasar a la etapa 2,
   guarda la lista de candidatos ya aprobada/editada con
   write_run_artifact(run_id, "bibliography", "candidates", {...}) -- flow_checker_agent
   chequea esto despues de la etapa 1, y si no encuentra nada ahi va a recomendar
   escalate_to_human (correctamente: sin este artifact no hay registro de que se
   aprobo, aunque el humano haya dicho que si).
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
8. Al terminar (exito O falla parcial -- no solo cuando todo salio bien): llama
   record_learning por cada cosa nueva que un run futuro deberia saber, y
   record_calibration con los numeros agregados de este run. Ver
   /pipeline-learnings/SKILL.md para el detalle de cuando usar cada uno -- no es
   obligatorio inventar un aprendizaje si no paso nada nuevo, pero SI es obligatorio
   llamar record_calibration si llegaste a tener un resultado de aggregate_text_scores.

El mensaje inicial del usuario te va a dar un run_id ya generado -- usalo de forma
consistente en TODAS las llamadas a tools de este run (no generes uno propio)."""


def build_orchestrator(
    checkpointer: BaseCheckpointSaver | None = None,
    agent_model: str | None = None,
    target_provider: str | None = None,
    target_model_name: str | None = None,
) -> Any:
    """Devuelve el grafo compilado del orquestador, listo para .invoke()/.stream().
    `checkpointer` es obligatorio si se va a usar el human-in-the-loop de
    bibliography_agent (sin el, los interrupts no pueden persistir estado).
    `target_provider`/`target_model_name` elige el modelo "black box" bajo test de MIA
    para ESTE run (ver mia_common.settings -- "configurable por run desde la webapp",
    son los defaults si no se pasan). Por eso mia_agent no es un subagent estatico como
    los otros 4: necesita un TargetClient real, construido aca, bindeado via closures
    (ver agents/subagents/mia_agent.py)."""
    skills_backend = FilesystemBackend(root_dir=str(settings.skill_dir.parent), virtual_mode=True)
    target_client = resolve_target_client(
        target_provider or settings.target_provider,
        target_model_name or settings.target_model_name,
    )
    subagents = [*STATIC_SUBAGENTS, build_mia_subagent(target_client)]
    return create_deep_agent(
        model=agent_model or settings.agent_model,
        tools=[combine_scores, aggregate_chunk_scores, aggregate_text_scores,
               write_run_artifact, read_run_artifact, list_run_artifacts,
               record_learning, record_calibration],
        subagents=subagents,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        backend=skills_backend,
        skills=["/"],
    )
