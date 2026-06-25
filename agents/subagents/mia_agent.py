"""
mia_agent — corre los 3 metodos de scoring MIA (DE-COP, SiMIA, DUALTEST) sobre un
chunk. Un solo subagent con las 3 tools (no uno por metodo) porque razonan sobre el
MISMO chunk y se benefician de tener todo el contexto junto -- ver el plan original
para la justificacion completa de esta decision de diseño.
"""

from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool

SYSTEM_PROMPT = """Sos el agente de scoring MIA. Para cada chunk (con sus candidatos de
paraphrase de SAGE, el titulo del libro, el autor, y el cliente target ya configurado):

1. Llama a run_decop_tool con el chunk verbatim + los paraphrase candidates de SAGE.
   Si te dice skipped=true (no hay >=3 candidatos), anotalo y segui -- no es un error,
   DE-COP simplemente no puede evaluar ese chunk.
2. Llama a run_simia_tool con el chunk.
3. Llama a run_dualtest_tool con el chunk, el modelo de referencia, y el label.
4. Resumi los 3 resultados (o las razones de skip) para que el orquestador los combine
   en el ensemble -- vos no combinas los scores, solo recolectas los 3 resultados crudos."""

mia_subagent = {
    "name": "mia_agent",
    "description": "Corre DE-COP, SiMIA y DUALTEST sobre un chunk y devuelve los 3 resultados crudos.",
    "system_prompt": SYSTEM_PROMPT,
    "tools": [run_decop_tool, run_simia_tool, run_dualtest_tool],
}
