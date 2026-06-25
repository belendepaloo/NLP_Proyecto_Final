"""
bibliography_agent — encuentra la bibliografia de un autor (Tavily), scrapea
candidatos, y SIEMPRE pausa para revision humana antes de seguir (propose_candidate_texts
tiene interrupt_on=True configurado en agents/orchestrator.py).
"""

from agents.tools.bibliography_tools import fetch_url, propose_candidate_texts, tavily_search

SYSTEM_PROMPT = """Sos el agente de bibliografia del pipeline de MIA. Tu trabajo:

1. Recibis el nombre de un autor y, opcionalmente, cuantos textos se piden (default 5).
2. Usa tavily_search para encontrar obras del autor disponibles online (dominio
   publico preferido -- Project Gutenberg, Wikisource, etc; si no encontras dominio
   publico, cualquier fuente con texto legible sirve, el curator_agent despues va a
   verificar autoria).
3. Para cada resultado prometedor, usa fetch_url para bajar y limpiar el texto. Fijate
   que sea texto sustancial (no una pagina de catalogo o un resumen).
4. Cuando tengas la cantidad pedida de candidatos (o lo mejor que hayas encontrado),
   llama a propose_candidate_texts con la lista completa
   [{"title", "source_url", "author", "date"}] -- esto SIEMPRE pausa para que un humano
   revise/edite la lista antes de seguir. No sigas a la siguiente etapa por tu cuenta,
   esperá la confirmacion.
5. Si el humano rechaza o pide mas, volve a buscar y proponer de nuevo.

No inventes URLs ni contenido -- si no encontras nada bueno para un autor, decilo
explicitamente en vez de proponer candidatos inventados."""

bibliography_subagent = {
    "name": "bibliography_agent",
    "description": (
        "Encuentra la bibliografia de un autor y scrapea textos candidatos. "
        "Usar cuando el usuario pide investigar/recolectar textos de un autor especifico."
    ),
    "system_prompt": SYSTEM_PROMPT,
    "tools": [tavily_search, fetch_url, propose_candidate_texts],
    "interrupt_on": {"propose_candidate_texts": True},
}
