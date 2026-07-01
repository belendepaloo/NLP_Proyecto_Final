"""
sage_qa_agent — parafrasea cada chunk con SAGE y decide que hacer cuando la calidad no
alcanza (reintentar, descartar, o escalar). El chequeo de calidad en si es deterministico
(sage_quality_check, ver agents/tools/sage_tools.py) -- lo que vale una LLM call es la
DECISION de que hacer con un chunk que fallo, no el chequeo.

Lee el texto del chunk del disco (chunk_id) en vez de recibirlo en el mensaje de la
tarea, y persiste el resultado del paraphraseo ahi tambien -- mismo patron que
bibliography_agent (texto descargado) y curator_agent (candidatos aprobados, texto de
chunks): el resumen final de un subagent a su padre no lleva el contenido completo (para
no gastar tokens de mas), asi que el siguiente subagent en la cadena (mia_agent) no
podria recibir el paraphrase via la conversacion aunque el orquestador quisiera
relayarlo -- tiene que leerlo del mismo lugar.

build_sage_qa_subagent(run_id) en vez de un dict estatico: mismo motivo que
bibliography_agent/curator_agent -- run_id queda bindeado por closure (este es,
literalmente, el subagente donde se vio en vivo el bug que motivo el cambio: un
run_id mal reproducido en una tool call de read_run_artifact crasheo un run entero a
mitad de SAGE).
"""

from agents.tools.fs_tools import make_run_scoped_fs_tools
from agents.tools.sage_tools import run_sage_tool, sage_quality_check

SYSTEM_PROMPT = """Sos el agente de QA de SAGE. Recibis una lista de chunk_id. Antes
de arrancar, llama a list_run_artifacts() UNA vez y guarda esa lista -- la vas a
necesitar para cada chunk_id sin volver a pedirla. Para CADA chunk_id:

1. Fijate en la lista de "curation" que ya tenes si "chunk_{chunk_id}.json" esta ahi
   -- NUNCA llames a read_run_artifact para un archivo que no viste listado: devuelve
   {"error": "not_found"} si no existe (a diferencia de las tools de scoring, que
   devuelven skipped=true de forma limpia). Si no esta listado, ese chunk_id no
   es real -- anotalo como error en tu resumen y segui con el siguiente, no inventes
   texto. Si SI esta, llama a read_run_artifact("curation", f"chunk_{chunk_id}")
   para traer el artifact {"document_id", "chunk_id", "text"} que persistio
   curator_agent -- NUNCA parafrasees un texto que no recuperaste de esta forma.
2. Llama a run_sage_tool(text=<el campo "text" de ese artifact, NO el dict entero>)
   para parafrasearlo.
3. Por cada segmento narrativo en el resultado, llama a
   sage_quality_check(sage_segment, min_sps=0.7, min_length_ratio=0.75).
4. Si "passed" es true para todos los segmentos narrativos: el chunk esta listo. Llama
   a write_run_artifact("sage", f"paraphrase_{chunk_id}", {"chunk_id":...,
   "paraphrase_candidates": <lista de strings -- los "all_candidates" de los segmentos
   narrativos que pasaron el QA, juntos>}) -- mia_agent va a leer esto de ahi, no de tu
   resumen.
5. Si algun segmento fallo: decidi entre
   - REINTENTAR una vez (puede ser ruido de un candidato puntual de SAGE).
   - DESCARTAR el chunk si sigue fallando despues de reintentar (anotalo en tu resumen
     final, no rompas el run completo por un chunk, y NO llames a write_run_artifact
     para ese chunk -- su ausencia en "sage" es la señal de que se descarto).
   - ESCALAR (avisar al usuario) solo si fallan TODOS los chunks de un texto entero --
     eso indicaria un problema sistematico, no un chunk puntual.

Resumi al final cuantos chunks pasaron (con sus chunk_id, para que el orquestador pueda
pasarlos a mia_agent), cuantos se descartaron y por que."""


def build_sage_qa_subagent(run_id: str) -> dict:
    """Devuelve el SubAgent spec de sage_qa_agent con sus tools de filesystem
    bindeadas a `run_id` (ver el docstring del modulo). run_sage_tool/sage_quality_check
    no tocan runs/<run_id>/ (operan sobre el texto que ya se les paso), no necesitan
    bindeo."""
    fs = make_run_scoped_fs_tools(run_id)
    return {
        "name": "sage_qa_agent",
        "description": "Parafrasea chunks con SAGE y decide que hacer si la calidad no alcanza.",
        "system_prompt": SYSTEM_PROMPT,
        "tools": [
            fs["list_run_artifacts"],
            fs["read_run_artifact"],
            fs["write_run_artifact"],
            run_sage_tool,
            sage_quality_check,
        ],
    }
