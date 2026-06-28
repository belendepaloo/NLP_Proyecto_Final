"""
results.py — construye la tabla de resultados (por chunk, por texto, y la probabilidad
final del autor) leyendo SOLO los artifacts persistidos en runs/<run_id>/ -- nunca el
resumen de texto que el orquestador escribe al final de su conversacion. Mismo
principio que el resto del pipeline (ver agents/orchestrator.py, agents/subagents/*):
el resumen de un LLM puede tener errores/redondeos/typos, el artifact en disco no.

Reusa agents.ensemble.combine (las mismas funciones que el orquestador ya llama) en
vez de reimplementar la logica de agregacion -- si algun dia cambia el ensemble,
cambia para los dos lugares a la vez.
"""

from __future__ import annotations

from agents.ensemble.combine import aggregate_chunk_scores, aggregate_text_scores, combine_scores
from agents.tools.fs_tools import list_run_artifacts, read_run_artifact


def _document_id_for_chunk(run_id: str, chunk_id: str) -> str:
    """El chunk en si (runs/<run_id>/curation/chunk_<chunk_id>.json) ya tiene
    "document_id" persistido por bibliography_tools.fetch_and_chunk_document -- mas
    confiable que parsear el chunk_id como string (un document_id puede tener "_" en
    el medio, ej. "sense_and_sensibility_fulltext_4")."""
    try:
        return read_run_artifact(run_id, "curation", f"chunk_{chunk_id}").get("document_id", chunk_id)
    except FileNotFoundError:
        return chunk_id


def build_results(run_id: str) -> dict | None:
    """Devuelve {"author_probability", "n_texts_scored", "n_texts_total", "texts": [
    {"document_id", "text_probability", "n_chunks_scored", "n_chunks_total", "chunks":
    [{"chunk_id", "final_probability", "per_method", "raw"}]}]} a partir de
    runs/<run_id>/mia_scores/*.json. Devuelve None si todavia no hay ningun chunk
    puntuado (nada que mostrar)."""
    artifacts = list_run_artifacts(run_id)
    score_files = artifacts.get("mia_scores", [])
    if not score_files:
        return None

    chunks_by_document: dict[str, list[dict]] = {}
    for fname in score_files:
        chunk_id = fname.removesuffix(".json")
        mia = read_run_artifact(run_id, "mia_scores", chunk_id)
        decop_result = mia.get("decop")
        simia_raw = mia.get("simia")
        dualtest_row = mia.get("dualtest")

        combined = combine_scores(dualtest_row=dualtest_row, simia_raw=simia_raw, decop_result=decop_result)
        document_id = _document_id_for_chunk(run_id, chunk_id)
        chunks_by_document.setdefault(document_id, []).append({
            "chunk_id": chunk_id,
            "final_probability": combined["final_probability"],
            "per_method": combined["per_method"],
            "reason": combined["reason"],
            "raw": {
                "decop_accuracy": decop_result.get("accuracy") if decop_result else None,
                "decop_n_queries": decop_result.get("n_queries") if decop_result else None,
                "dualtest_p_rlb": dualtest_row.get("p_rlb") if dualtest_row else None,
                "dualtest_p_esb": dualtest_row.get("p_esb") if dualtest_row else None,
                "simia_raw": simia_raw,
            },
        })

    texts = []
    for document_id, chunk_entries in chunks_by_document.items():
        agg = aggregate_chunk_scores(chunk_entries)
        texts.append({
            "document_id": document_id,
            "text_probability": agg["text_probability"],
            "n_chunks_scored": agg["n_chunks_scored"],
            "n_chunks_total": agg["n_chunks_total"],
            "chunks": sorted(chunk_entries, key=lambda c: c["chunk_id"]),
        })
    texts.sort(key=lambda t: t["document_id"])

    author_agg = aggregate_text_scores(texts)
    return {
        "author_probability": author_agg["author_probability"],
        "n_texts_scored": author_agg["n_texts_scored"],
        "n_texts_total": author_agg["n_texts_total"],
        "texts": texts,
    }
