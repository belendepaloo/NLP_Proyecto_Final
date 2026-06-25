"""
bibliography_tools.py — tools del bibliography_agent (Fase 2): encontrar la
bibliografia de un autor (Tavily), scrapear candidatos, y pausar para revision humana
antes de seguir (propose_candidate_texts, con interrupt_on configurado en el
orquestador -- ver agents/orchestrator.py).
"""

from __future__ import annotations

import requests

from mia_common.settings import settings

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_tavily_client = None


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        if not settings.tavily_api_key:
            raise RuntimeError(
                "TAVILY_API_KEY no esta configurada (ver .env.example) -- "
                "necesaria para que bibliography_agent busque la bibliografia del autor."
            )
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Busca en la web (Tavily) y devuelve una lista de {title, url, snippet} -- usar
    para encontrar la bibliografia/obras de un autor, ediciones de dominio publico, etc."""
    client = _get_tavily_client()
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content", "")[:500]}
        for r in resp.get("results", [])
    ]


def fetch_url(url: str) -> dict:
    """Descarga `url` y devuelve el texto limpio (trafilatura, ver
    processRawText.text_pipeline.clean_text) -- el mismo limpiador que usa el resto del
    pipeline, para que el chunking despues sea consistente."""
    from processRawText.text_pipeline import clean_text

    resp = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=30)
    resp.raise_for_status()
    cleaned = clean_text(resp.text, is_html=True)
    return {"url": url, "cleaned_text": cleaned, "n_chars": len(cleaned)}


def propose_candidate_texts(candidates: list[dict]) -> dict:
    """Terminal tool de la etapa de bibliografia. Llamar a este tool SIEMPRE dispara
    una pausa de revision humana (interrupt_on en el orquestador) -- el humano puede
    aprobar la lista tal cual, editarla (sacar/agregar candidatos), o rechazarla para
    que el agente busque de nuevo. `candidates`: lista de
    {"title", "source_url", "author", "date"} por documento propuesto."""
    return {"status": "awaiting_human_review", "candidates": candidates}
