"""
mia_agent — corre los 3 metodos de scoring MIA (DE-COP, SiMIA, DUALTEST) sobre un
chunk. Un solo subagent con las 3 tools (no uno por metodo) porque razonan sobre el
MISMO chunk y se benefician de tener todo el contexto junto -- ver el plan original
para la justificacion completa de esta decision de diseño.

build_mia_subagent(client, reference_model_name) en vez de un dict estatico como los
otros 4 subagentes: `run_decop_tool`/`run_simia_tool`/`run_dualtest_tool` (ver
agents/tools/mia_tools.py) toman un `client: TargetClient` -- un objeto Python real,
no algo JSON-serializable que un LLM pueda completar en una tool call. Bug real
encontrado armando el selector de modelo target de la Fase 4: exponer ese parametro
crudo en la tool rompe `convert_to_genai_function_declarations` con
`PydanticInvalidForJsonSchema` apenas el orquestador intenta delegar a mia_agent (nunca
se habia ejercitado este subagent en ningun test hasta ahora). `functools.partial` NO
sirve para esconderlo (langchain sigue viendo el parametro, y
`typing.get_type_hints` directamente rechaza un `partial`) -- la solucion es un
closure real (un `def` anidado) que capture `client`/`reference_model_name` y
no los exponga en su propia firma.
"""

from __future__ import annotations

from mia_common.settings import settings
from mia_common.target_client import TargetClient
from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool

SYSTEM_PROMPT = """Sos el agente de scoring MIA. Para cada chunk (con sus candidatos de
paraphrase de SAGE, el titulo del libro, y el autor -- el cliente target y el modelo de
referencia de DUALTEST ya estan configurados para todo este run, no son algo que vos
elijas):

1. Llama a run_decop_tool con el chunk verbatim + los paraphrase candidates de SAGE.
   Si te dice skipped=true (no hay >=3 candidatos), anotalo y segui -- no es un error,
   DE-COP simplemente no puede evaluar ese chunk.
2. Llama a run_simia_tool con el chunk.
3. Llama a run_dualtest_tool con el chunk y el label (0 si no sabes la membership real
   de este chunk, que es el caso normal en inferencia).
4. Resumi los 3 resultados (o las razones de skip) para que el orquestador los combine
   en el ensemble -- vos no combinas los scores, solo recolectas los 3 resultados crudos."""


def build_mia_subagent(client: TargetClient, reference_model_name: str | None = None) -> dict:
    """Devuelve el SubAgent spec de mia_agent con `client` (el TargetClient elegido
    para este run, ver agents/orchestrator.py) y el reference model de DUALTEST ya
    bindeados via closures -- la eleccion de modelo target es una decision de RUN (la
    hace el humano via la webapp), no algo que mia_agent deba rellenar en cada tool
    call."""
    ref_model = reference_model_name or settings.reference_model_name

    def run_decop_tool_bound(
        verbatim_passage: str,
        paraphrase_candidates: list[str],
        book_title: str,
        author: str,
        n_permutations: int = 6,
    ) -> dict:
        """Devuelve {"method": "decop", "skipped": bool, "result": dict | None,
        "reason": str | None}. Skippea (no rompe el run) si SAGE no produjo >=3
        candidatos validos para este chunk."""
        return run_decop_tool(
            verbatim_passage=verbatim_passage,
            paraphrase_candidates=paraphrase_candidates,
            book_title=book_title,
            author=author,
            client=client,
            n_permutations=n_permutations,
        )

    def run_simia_tool_bound(
        text: str,
        non_member_prefix: str | None = None,
        n_samples: int | None = None,
        max_words: int = 20,
    ) -> dict:
        """Devuelve {"method": "simia", "skipped": bool, "result": float | None,
        "reason": str | None}. non_member_prefix=None / n_samples=None usan los
        defaults de simmia_score (prefijo de calibracion fijo +
        mia_common.settings.simia_n_samples)."""
        return run_simia_tool(
            text=text,
            client=client,
            non_member_prefix=non_member_prefix,
            n_samples=n_samples,
            max_words=max_words,
        )

    def run_dualtest_tool_bound(
        text: str,
        prefix_len: int = 64,
        continuation_len: int = 64,
        max_new_tokens: int = 64,
        label: int = 0,
    ) -> dict:
        """Devuelve {"method": "dualtest", "skipped": bool, "result": dict | None,
        "reason": str | None}. `result`, si no es None, tiene
        run_length/p_rlb/edit_similarity/p_esb (ver DUALTEST.scoring.score_texts).
        `label` es la membership conocida si la hay (0 = desconocida/inferencia,
        el caso normal)."""
        return run_dualtest_tool(
            text=text,
            client=client,
            reference_model_name=ref_model,
            prefix_len=prefix_len,
            continuation_len=continuation_len,
            max_new_tokens=max_new_tokens,
            label=label,
        )

    return {
        "name": "mia_agent",
        "description": "Corre DE-COP, SiMIA y DUALTEST sobre un chunk y devuelve los 3 resultados crudos.",
        "system_prompt": SYSTEM_PROMPT,
        "tools": [run_decop_tool_bound, run_simia_tool_bound, run_dualtest_tool_bound],
    }
