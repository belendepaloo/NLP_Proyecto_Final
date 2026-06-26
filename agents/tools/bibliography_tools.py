"""
bibliography_tools.py — tools del bibliography_agent (Fase 2): encontrar la
bibliografia de un autor (Tavily), scrapear candidatos, y pausar para revision humana
antes de seguir (propose_candidate_texts, con interrupt_on configurado en el
orquestador -- ver agents/orchestrator.py).
"""

from __future__ import annotations

import subprocess
import time

from mia_common.settings import settings

FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 3

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
    pipeline, para que el chunking despues sea consistente.

    Usa `curl` via subprocess en vez de `requests`: confirmado que gutenberg.org acepta
    el handshake TLS de curl pero cuelga (timeout) el de urllib3/requests con el MISMO
    User-Agent de navegador -- bloqueo por fingerprint TLS (JA3), no por el header.

    `--http1.1` es necesario: con HTTP/2 (el default de curl si el server lo ofrece),
    la combinacion `--fail` + respuesta de error HTTP hace que curl aborte el stream de
    forma que reporta exit 56 (recv error / fallo de red) en vez de exit 22 (error HTTP
    claro) -- confirmado reproduciendo el mismo 404 con y sin --http1.1. No es
    gutenberg.org cortando la conexion, es esta combinacion especifica de flags de curl.

    El reintento con backoff queda como defensa ante fallos de red genuinamente
    transitorios (no el caso mas comun ahora que --http1.1 esta arreglado).

    NO lanza excepcion si los reintentos se agotan -- devuelve {"url", "error"} en vez
    de {"url", "cleaned_text", "n_chars"}. Motivo: una sola fuente caida (sitio lento,
    URL rota, 404) no tiene que tirar abajo el run entero -- el agente que llama a esto
    (bibliography_agent) tiene que poder ver el error y probar el siguiente candidato."""
    from processRawText.text_pipeline import clean_text

    last_error = ""
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        result = subprocess.run(
            ["curl", "-sL", "--http1.1", "--fail", "--max-time", "30", "-A", BROWSER_UA, url],
            capture_output=True,
        )
        if result.returncode == 0:
            cleaned = clean_text(result.stdout.decode("utf-8", errors="replace"), is_html=True)
            return {"url": url, "cleaned_text": cleaned, "n_chars": len(cleaned)}
        last_error = result.stderr.decode("utf-8", errors="replace").strip()
        if attempt < FETCH_MAX_RETRIES:
            time.sleep(FETCH_RETRY_BACKOFF_SECONDS * attempt)
    return {
        "url": url,
        "error": (
            f"curl fallo descargando {url} tras {FETCH_MAX_RETRIES} intentos "
            f"(ultimo exit {result.returncode}): {last_error or 'sin detalle (timeout u otro error de red)'}"
        ),
    }


def propose_candidate_texts(candidates: list[dict]) -> dict:
    """Terminal tool de la etapa de bibliografia. Llamar a este tool SIEMPRE dispara
    una pausa de revision humana (interrupt_on en el orquestador) -- el humano puede
    aprobar la lista tal cual, editarla (sacar/agregar candidatos), o rechazarla para
    que el agente busque de nuevo. `candidates`: lista de
    {"title", "source_url", "author", "date"} por documento propuesto."""
    return {"status": "awaiting_human_review", "candidates": candidates}
