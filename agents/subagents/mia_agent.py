"""
mia_agent — corre los 3 metodos de scoring MIA (DE-COP, SiMIA, DUALTEST) sobre un
chunk. Un solo subagent con las 3 tools (no uno por metodo) porque razonan sobre el
MISMO chunk y se benefician de tener todo el contexto junto -- ver el plan original
para la justificacion completa de esta decision de diseño.

build_mia_subagent(run_id, client, reference_model_name) en vez de un dict estatico
como los otros 4 subagentes: `run_decop_tool`/`run_simia_tool`/`run_dualtest_tool` (ver
agents/tools/mia_tools.py) toman un `client: TargetClient` -- un objeto Python real,
no algo JSON-serializable que un LLM pueda completar en una tool call. Bug real
encontrado armando el selector de modelo target de la Fase 4: exponer ese parametro
crudo en la tool rompe `convert_to_genai_function_declarations` con
`PydanticInvalidForJsonSchema` apenas el orquestador intenta delegar a mia_agent (nunca
se habia ejercitado este subagent en ningun test hasta ahora). `functools.partial` NO
sirve para esconderlo (langchain sigue viendo el parametro, y
`typing.get_type_hints` directamente rechaza un `partial`) -- la solucion es un
closure real (un `def` anidado) que capture `client`/`reference_model_name` y
no los exponga en su propia firma. `run_id` se bindea por el mismo mecanismo (via
make_run_scoped_fs_tools, ver agents/tools/fs_tools.py) por una razon distinta: no es
un objeto no-serializable, es para que el LLM no tenga que reproducirlo verbatim en
cada tool call (ver el docstring de esa funcion para el bug real que motivo esto).
"""

from __future__ import annotations

from mia_common.settings import settings
from mia_common.target_client import TargetClient
from agents.tools.fs_tools import make_run_scoped_fs_tools
from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool

_STEPS_WITH_SIMIA = """1. Llama a run_decop_tool con el chunk verbatim + los paraphrase candidates de SAGE.
   Si te dice skipped=true (no hay >=3 candidatos), anotalo y segui -- no es un error,
   DE-COP simplemente no puede evaluar ese chunk.
2. Llama a run_simia_tool con el chunk.
3. Llama a run_dualtest_tool con el chunk y el label (0 si no sabes la membership real
   de este chunk, que es el caso normal en inferencia).
4. Llama a write_run_artifact("mia_scores", chunk_id, {"decop": <el "result" de
   run_decop_tool, o null si skipped>, "simia": <el "result" de run_simia_tool, o null
   si skipped>, "dualtest": <el "result" de run_dualtest_tool, o null si skipped>})."""

_STEPS_WITHOUT_SIMIA = """1. Llama a run_decop_tool con el chunk verbatim + los paraphrase candidates de SAGE.
   Si te dice skipped=true (no hay >=3 candidatos), anotalo y segui -- no es un error,
   DE-COP simplemente no puede evaluar ese chunk.
2. Llama a run_dualtest_tool con el chunk y el label (0 si no sabes la membership real
   de este chunk, que es el caso normal en inferencia).
3. SiMIA esta DESACTIVADO por ahora (todavia no esta terminado, decision del usuario
   -- ver mia_common.settings.simia_enabled) -- NO intentes correrlo, no tenes la tool
   disponible. Llama a write_run_artifact("mia_scores", chunk_id, {"decop": <el
   "result" de run_decop_tool, o null si skipped>, "simia": null, "dualtest": <el
   "result" de run_dualtest_tool, o null si skipped>})."""

_FOOTER = """ -- el orquestador llama a combine_scores DESPUES, leyendo este artifact,
no tu resumen de texto (mismo motivo que el resto de la cadena: tu resumen no lleva los
dicts completos, solo lo esencial para que un humano lo lea). Resumi en texto los
resultados (o las razones de skip) para el humano que lea el run -- vos NO combinas
los scores, solo los recolectas y los persistis crudos."""

_HEADER = """Sos el agente de scoring MIA. Recibis el titulo del libro, el autor, y
una lista de chunk_id -- el cliente target y el modelo de referencia de DUALTEST ya
estan configurados para todo este run, no son algo que vos elijas. Para CADA chunk_id:

0. Llama a list_run_artifacts() UNA vez (la lista te sirve para todos los
   chunk_id, no hace falta repetirla). NUNCA llames a read_run_artifact para un
   archivo que no viste listado primero -- tira una excepcion que frena el run
   ENTERO (a diferencia de las tools de scoring, que devuelven skipped=true en vez de
   excepcionar). Fijate si "chunk_{chunk_id}.json" esta en "curation" -- si no esta,
   ese chunk_id no es real, anotalo como error y segui con el siguiente. Si esta,
   llama a read_run_artifact("curation", f"chunk_{chunk_id}") para el texto
   verbatim real (el campo "text" de ese artifact). Despues fijate si
   "paraphrase_{chunk_id}.json" aparece en la lista de "sage". Si SI esta, llama a
   read_run_artifact("sage", f"paraphrase_{chunk_id}") para el campo
   "paraphrase_candidates". Si NO esta listado, sage_qa_agent descarto ese chunk --
   segui sin DE-COP para el (paraphrase_candidates=[]), no inventes candidatos. NUNCA
   puntues un chunk con texto que no recuperaste de esta forma.
"""


def build_mia_subagent(run_id: str, client: TargetClient, reference_model_name: str | None = None) -> dict:
    """Devuelve el SubAgent spec de mia_agent con `client` (el TargetClient elegido
    para este run, ver agents/orchestrator.py), el reference model de DUALTEST, y las
    tools de filesystem ya bindeadas a `run_id` via closures -- la eleccion de modelo
    target es una decision de RUN (la hace el humano via la webapp), no algo que
    mia_agent deba rellenar en cada tool call.

    Si mia_common.settings.simia_enabled es False (default actual, a pedido del
    usuario: "SiMIA todavia no esta terminado"), run_simia_tool ni siquiera se incluye
    en las tools del subagent -- no alcanza con pedirle al prompt que no lo use, si la
    tool esta disponible un LLM puede llamarla igual."""
    fs = make_run_scoped_fs_tools(run_id)
    ref_model = reference_model_name or settings.reference_model_name
    steps = _STEPS_WITH_SIMIA if settings.simia_enabled else _STEPS_WITHOUT_SIMIA
    system_prompt = _HEADER + steps + _FOOTER

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

    tools = [
        fs["list_run_artifacts"],
        fs["read_run_artifact"],
        fs["write_run_artifact"],
        run_decop_tool_bound,
        run_dualtest_tool_bound,
    ]
    if settings.simia_enabled:
        tools.append(run_simia_tool_bound)

    methods_desc = "DE-COP, SiMIA y DUALTEST" if settings.simia_enabled else "DE-COP y DUALTEST (SiMIA desactivado)"
    return {
        "name": "mia_agent",
        "description": f"Corre {methods_desc} sobre un chunk y devuelve los resultados crudos.",
        "system_prompt": system_prompt,
        "tools": tools,
    }
