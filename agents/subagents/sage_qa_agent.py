"""
sage_qa_agent — parafrasea cada chunk con SAGE y decide que hacer cuando la calidad no
alcanza (reintentar, descartar, o escalar). El chequeo de calidad en si es deterministico
(sage_quality_check, ver agents/tools/sage_tools.py) -- lo que vale una LLM call es la
DECISION de que hacer con un chunk que fallo, no el chequeo.
"""

from agents.tools.sage_tools import run_sage_tool, sage_quality_check

SYSTEM_PROMPT = """Sos el agente de QA de SAGE. Para cada chunk de texto que te pasen:

1. Llama a run_sage_tool(text=chunk) para parafrasearlo.
2. Por cada segmento narrativo en el resultado, llama a
   sage_quality_check(sage_segment, min_sps=0.7, min_length_ratio=0.75).
3. Si "passed" es true para todos los segmentos narrativos, el chunk esta listo --
   pasalo a mia_agent tal cual.
4. Si algun segmento fallo: decidi entre
   - REINTENTAR una vez (puede ser ruido de un candidato puntual de SAGE).
   - DESCARTAR el chunk si sigue fallando despues de reintentar (anotalo en tu resumen
     final, no rompas el run completo por un chunk).
   - ESCALAR (avisar al usuario) solo si fallan TODOS los chunks de un texto entero --
     eso indicaria un problema sistematico, no un chunk puntual.

Resumi al final cuantos chunks pasaron, cuantos se descartaron y por que."""

sage_qa_subagent = {
    "name": "sage_qa_agent",
    "description": "Parafrasea chunks con SAGE y decide que hacer si la calidad no alcanza.",
    "system_prompt": SYSTEM_PROMPT,
    "tools": [run_sage_tool, sage_quality_check],
}
