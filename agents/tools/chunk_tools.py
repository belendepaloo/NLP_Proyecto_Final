"""
chunk_tools.py — adapter delgado sobre processRawText.text_pipeline. Etapa
determinista (limpieza HTML + chunking por tokens), por eso son funciones planas, no
un subagente LLM (ver Fase 2 del plan: "chunker" es una tool, no un subagent).
"""

from __future__ import annotations

from dataclasses import asdict

from processRawText.text_pipeline import (
    chunk_text,
    clean_text,
    make_tiktoken_counter,
)


_token_counter = None


def get_token_counter():
    """Contador unico de tokens (tiktoken cl100k_base) para que el largo de chunk sea
    comparable entre todos los textos/metodos del pipeline -- mismo principio que ya
    usa scrape_clean_chunk.ipynb."""
    global _token_counter
    if _token_counter is None:
        try:
            _token_counter = make_tiktoken_counter()
        except Exception:
            from processRawText.text_pipeline import est_token_count

            _token_counter = est_token_count
    return _token_counter


def clean_html_tool(raw: str, is_html: bool | None = None) -> str:
    """Limpia HTML/texto crudo: trafilatura + normalizacion (ver text_pipeline.clean_text)."""
    return clean_text(raw, is_html=is_html)


def chunk_text_tool(clean: str, target: int = 128, min_len: int = 24, source: str | None = None,
                     date: str | None = None) -> list[dict]:
    """Chunkea texto ya limpio en ventanas verbatim de ~`target` tokens. Devuelve dicts
    JSON-serializables (no dataclasses), listos para escribir como artifact o pasar a
    un tool de deepagents.

    OJO: este tool recibe el texto COMPLETO como argumento -- bien para textos cortos,
    pero un libro entero (~700K caracteres, medido en vivo con novelas reales de Jane
    Austen via bibliography_agent) significa mandarle esa cadena entera a Gemini como
    argumento de function-calling. Eso es justo el riesgo que ya advertia este repo
    ("va a necesitar chunkear por capitulo/pagina, no el libro entero junto") y que
    causo un run real carisimo (~$2.81) que ademas nunca termino de chunkear. Para
    bibliography_agent usar bibliography_tools.fetch_and_chunk_document() en vez de
    este -- descarga, recorta a ~15 paginas, y chunkea server-side en una sola llamada,
    sin que el texto completo (ni siquiera el recorte) pase nunca por el contexto del
    LLM ni se guarde como un solo artifact."""
    chunks = chunk_text(
        clean,
        target=target,
        count_fn=get_token_counter(),
        min_len=min_len,
        source=source,
        date=date,
    )
    return [asdict(c) for c in chunks]
