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

from agents.tools.fs_tools import read_run_artifact, write_run_artifact

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
    curator_agent usar clean_and_chunk_document() en vez de este -- chunkea server-side
    sin que el texto completo pase nunca por el contexto del LLM."""
    chunks = chunk_text(
        clean,
        target=target,
        count_fn=get_token_counter(),
        min_len=min_len,
        source=source,
        date=date,
    )
    return [asdict(c) for c in chunks]


def clean_and_chunk_document(run_id: str, document_id: str, target: int = 128, min_len: int = 24) -> dict:
    """Lee el texto de runs/<run_id>/bibliography/text_<document_id> (ya viene limpio
    de bibliography_tools.fetch_url, no hace falta re-limpiarlo), lo chunkea, y persiste
    CADA chunk en runs/<run_id>/curation/chunk_<document_id>_<i> -- todo server-side, el
    texto completo del documento NUNCA se pasa como argumento de una tool call ni se
    devuelve entero en la respuesta (ver el docstring de chunk_text_tool sobre por que
    eso es un problema real con libros completos, no solo teorico).

    Devuelve {"document_id", "n_chunks", "chunk_ids"} -- liviano a proposito. El agente
    tiene que leer el texto de cada chunk individualmente con
    read_run_artifact(run_id, "curation", f"chunk_{chunk_id}") cuando lo necesite (ver
    el procedimiento de lotes chicos en agents/subagents/curator_agent.py), nunca todos
    de una."""
    doc = read_run_artifact(run_id, "bibliography", f"text_{document_id}")
    chunks = chunk_text(doc["cleaned_text"], target=target, count_fn=get_token_counter(), min_len=min_len)

    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_{i}"
        write_run_artifact(
            run_id, "curation", f"chunk_{chunk_id}", {"document_id": document_id, "chunk_id": chunk_id, "text": chunk.text}
        )
        chunk_ids.append(chunk_id)

    return {"document_id": document_id, "n_chunks": len(chunk_ids), "chunk_ids": chunk_ids}
