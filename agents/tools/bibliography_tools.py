"""
bibliography_tools.py — tools del bibliography_agent (Fase 2): encontrar la
bibliografia de un autor (Tavily), scrapear candidatos, y pausar para revision humana
antes de seguir (propose_candidate_texts, con interrupt_on configurado en el
orquestador -- ver agents/orchestrator.py).
"""

from __future__ import annotations

import subprocess
import time
from typing import Callable

from agents.tools.chunk_tools import get_token_counter
from agents.tools.fs_tools import read_run_artifact, write_run_artifact
from mia_common.settings import settings
from processRawText.text_pipeline import chunk_text

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
    para encontrar la bibliografia/obras de un autor, ediciones de dominio publico, etc.

    `query` tiene que tener termino(s) de busqueda reales, no solo operadores tipo
    `site:`/`inurl:`/`filetype:` -- Tavily rechaza una query que sea SOLO eso (bug real
    visto en vivo: "site:gutenberg.org/ebooks/158" sin ningun otro termino crasheaba el
    run entero). Si necesitas restringir a un dominio, combinalo con palabras reales
    (ej. "site:gutenberg.org Emma Jane Austen"), no lo mandes solo.

    NO lanza excepcion si Tavily rechaza la query o falla la llamada -- devuelve
    [{"error": ...}] en vez de la lista de resultados. Motivo: una sola busqueda mal
    armada (mismo principio que fetch_url con una URL caida) no tiene que tirar abajo
    el run entero -- el agente que llama a esto tiene que poder ver el error y probar
    una query mejor."""
    client = _get_tavily_client()
    try:
        resp = client.search(query=query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 -- cualquier falla de Tavily (query invalida, rate limit, red) es recuperable
        return [{"error": f"{type(e).__name__}: {e}"}]
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


def fetch_and_chunk_document(run_id: str, document_id: str, url: str, target: int = 128, min_len: int = 24) -> dict:
    """Descarga `url` (via fetch_url), RECORTA el texto limpio a
    settings.bibliography_max_chars_per_document caracteres (~15 paginas -- nunca el
    libro entero, sin importar cuanto mida la fuente real) y lo chunkea -- todo
    server-side, en una sola llamada. Persiste CADA chunk en
    runs/<run_id>/curation/chunk_<document_id>_<i>; el texto completo (ni siquiera el
    recorte de 15 paginas) se devuelve nunca como string ni se guarda como un solo
    artifact -- vive en una variable de Python el tiempo que tarda esta funcion y se
    descarta. NUNCA se persiste runs/<run_id>/bibliography/text_<document_id> (ese
    artifact ya no existe en este pipeline): el libro completo no puede quedar guardado
    en ningun lugar, ni local ni en el contexto de un LLM.

    bibliography_agent llama a esto en vez de fetch_url + write_run_artifact por
    separado -- esa version anterior le pedia al LLM que copiara el texto completo como
    argumento de una tool call para guardarlo, y eso fallaba en silencio: ningun modelo
    reproduce fielmente un texto de cientos de miles de caracteres, terminaba
    "guardando" un resumen de ~900 caracteres sin avisar que habia recortado nada (bug
    real, ver pipeline-learnings).

    Devuelve {"document_id", "url", "n_chunks", "chunk_ids", "n_chars_used",
    "n_chars_available", "preview"} si la descarga funciono, o {"document_id", "url",
    "error"} si fetch_url fallo -- preview son los primeros ~300 caracteres del
    recorte, para que el agente pueda confirmar de un vistazo que esto es prosa real y
    no una pagina de catalogo/resumen, sin tener que leer todo el texto."""
    fetched = fetch_url(url)
    if "error" in fetched:
        return {"document_id": document_id, "url": url, "error": fetched["error"]}

    cleaned = fetched["cleaned_text"]
    capped = cleaned[: settings.bibliography_max_chars_per_document]

    chunks = chunk_text(capped, target=target, count_fn=get_token_counter(), min_len=min_len)
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_{i}"
        write_run_artifact(
            run_id, "curation", f"chunk_{chunk_id}",
            {"document_id": document_id, "chunk_id": chunk_id, "text": chunk.text},
        )
        chunk_ids.append(chunk_id)

    return {
        "document_id": document_id,
        "url": url,
        "n_chunks": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "n_chars_used": len(capped),
        "n_chars_available": len(cleaned),
        "preview": capped[:300],
    }


def propose_candidate_texts(run_id: str, candidates: list[dict]) -> dict:
    """Terminal tool de la etapa de bibliografia. Llamar a este tool SIEMPRE dispara
    una pausa de revision humana (interrupt_on en el orquestador) -- el humano puede
    aprobar la lista tal cual, editarla (sacar/agregar candidatos), o rechazarla para
    que el agente busque de nuevo. `candidates`: lista de
    {"document_id", "title", "source_url", "author", "date"} por documento propuesto.

    Persiste `candidates` en runs/<run_id>/bibliography/candidates.json DESPUES de que
    el humano resuelve la pausa (con los args ya aprobados/editados, nunca los
    originales que propuso el agente -- deepagents reemplaza `candidates` por la
    version editada antes de ejecutar esto si el humano elige "editar"). Que esto se
    persista ACA, adentro del tool que la pausa humana protege, en vez de que el
    orquestador lo guarde el solo despues, es deliberado: bug real encontrado en una
    sesion anterior -- el orquestador, cuando bibliography_agent no encontraba nada,
    fabricaba el una lista de candidatos y la guardaba como si el humano la hubiera
    aprobado, sin que ninguna pausa real hubiera ocurrido. Si el artifact en disco solo
    puede existir como consecuencia de este tool ejecutandose post-resume, un candidato
    fabricado por el orquestador ya no puede pasar por "aprobado" (ver
    pipeline-learnings antes de asumir que esto sigue roto).

    SUMA en vez de PISAR si ya existe runs/<run_id>/bibliography/candidates.json: una
    ronda de reemplazo (curator_agent descarto un candidato y bibliography_agent busco
    uno nuevo para esa misma corrida) llama a esto una segunda vez con SOLO los
    candidatos nuevos -- si pisara el archivo entero, se perderian los candidatos
    aprobados en la primera ronda. Mergea por document_id (un id repetido pisa solo esa
    entrada, no el resto)."""
    existing = []
    try:
        existing = read_run_artifact(run_id, "bibliography", "candidates")["candidates"]
    except FileNotFoundError:
        pass

    by_id = {c["document_id"]: c for c in existing}
    for c in candidates:
        by_id[c["document_id"]] = c
    merged = list(by_id.values())

    write_run_artifact(run_id, "bibliography", "candidates", {"candidates": merged})
    return {"status": "approved", "candidates": merged}


def make_run_scoped_bibliography_tools(run_id: str) -> dict[str, Callable]:
    """Devuelve fetch_and_chunk_document/propose_candidate_texts con `run_id` ya fijo
    via closure -- mismo motivo que make_run_scoped_fs_tools (agents/tools/fs_tools.py):
    bibliography_agent ya opera sobre UN run fijo durante toda su tarea, asi que no
    hay razon para dejar que el LLM tipee el run_id en cada llamada (y arriesgue un
    typo que mande chunks a un directorio fantasma)."""

    def fetch_and_chunk_document_bound(document_id: str, url: str, target: int = 128, min_len: int = 24) -> dict:
        """Descarga `url`, RECORTA a settings.bibliography_max_chars_per_document
        caracteres (~15 paginas -- nunca el libro entero) y chunkea -- todo
        server-side, en una sola llamada. Persiste CADA chunk en
        runs/<este run>/curation/chunk_<document_id>_<i>; el texto completo (ni
        siquiera el recorte) se devuelve nunca como string ni se guarda como un solo
        artifact. Devuelve {"document_id", "url", "n_chunks", "chunk_ids",
        "n_chars_used", "n_chars_available", "preview"} si funciono, o {"document_id",
        "url", "error"} si la descarga fallo."""
        return fetch_and_chunk_document(run_id, document_id, url, target=target, min_len=min_len)

    def propose_candidate_texts_bound(candidates: list[dict]) -> dict:
        """Terminal tool de la etapa de bibliografia -- SIEMPRE pausa para revision
        humana. `candidates`: lista de {"document_id", "title", "source_url",
        "author", "date"} por documento propuesto (el document_id de cada uno tiene
        que ser EXACTAMENTE el mismo que se uso al llamar a fetch_and_chunk_document).
        Suma a la lista existente de este run si ya hay candidatos aprobados de una
        ronda anterior (no la pisa)."""
        return propose_candidate_texts(run_id, candidates)

    fetch_and_chunk_document_bound.__name__ = "fetch_and_chunk_document"
    propose_candidate_texts_bound.__name__ = "propose_candidate_texts"

    return {
        "fetch_and_chunk_document": fetch_and_chunk_document_bound,
        "propose_candidate_texts": propose_candidate_texts_bound,
    }
