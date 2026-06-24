#!/usr/bin/env python3
"""
run_pipeline_manual.py — milestone demostrable de la Fase 1: pipeline determinista
end-to-end (chunks ya limpios -> SAGE -> DE-COP/SiMIA/DUALTEST -> ensemble) SIN
agentes y SIN scraping/curacion automatica.

Usa processRawText/Datasets/dataset_len128.csv (200 chunks de ~128 tokens ya
limpiados+chunkeados por scrape_clean_chunk.ipynb sobre 3 novelas de Dickens via
Gutenberg, label=1/member, y un articulo de Wikipedia de 2025, label=0/non-member) en
vez de re-correr clean_html_tool/chunk_text_tool sobre el HTML crudo de la novela
completa: chunk_text (pysbd) escala mal sobre un libro entero de una sola vez (~5 min
para "A Tale of Two Cities" en pruebas) -- clean_html_tool/chunk_text_tool ya se
probaron por separado sobre slices mas chicos y funcionan; este script los reutiliza
para texto NUEVO que llegue del scraping (Fase 2), no para re-procesar lo que ya esta
chunkeado.

Pensado para correr de punta a punta una vez que haya GROQ_API_KEY (.env) y, si se
quiere probar SAGE de verdad, transformer_lens+sae_lens instalados. Sin esas dos cosas
el script sigue corriendo y muestra claramente que etapas se saltearon y por que --
no es un requisito tener todo instalado para ver la forma del pipeline.

Uso:
    python scripts/run_pipeline_manual.py [--max-chunks-per-text N] [--no-sage]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ensemble.combine import aggregate_chunk_scores, aggregate_text_scores, combine_scores  # noqa: E402
from agents.tools.fs_tools import write_run_artifact  # noqa: E402
from mia_common.settings import settings  # noqa: E402
from mia_common.target_client import make_target_client  # noqa: E402

DATASET_CSV = Path(__file__).resolve().parents[1] / "processRawText" / "Datasets" / "dataset_len128.csv"
RUN_ID = "manual_phase1_smoke_test"
AUTHOR = "Charles Dickens"


def load_chunks_by_title() -> dict[str, list[dict]]:
    df = pd.read_csv(DATASET_CSV)
    return {title: g.to_dict("records") for title, g in df.groupby("title")}


def try_build_target_client():
    if not settings.groq_api_key:
        print("  [SKIP target client] GROQ_API_KEY no configurada -- ver .env.example.")
        return None
    return make_target_client(
        provider=settings.target_provider,
        model_name=settings.target_model_name,
        api_key=settings.groq_api_key,
        min_seconds_between_calls=settings.target_min_seconds_between_calls,
        max_retries=settings.target_max_retries,
    )


def try_run_sage(text: str, use_sage: bool) -> dict | None:
    if not use_sage:
        return None
    try:
        from agents.tools.sage_tools import run_sage_tool

        return run_sage_tool(text)
    except ImportError as e:
        print(f"  [SKIP SAGE] {e}")
        return None


def score_chunk(chunk_text_str: str, book_title: str, label: int, client, sage_result: dict | None) -> dict:
    from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool

    decop_result = None
    simia_raw = None
    dualtest_row = None

    if client is not None:
        # candidatos de paraphrase de SAGE para ESTE chunk (si SAGE corrio y el chunk
        # mapea a un solo segmento narrativo -- simplificacion para el smoke test)
        candidates: list[str] = []
        if sage_result is not None:
            for seg in sage_result.get("segments", []):
                candidates.extend(c["text"] for c in seg.get("all_candidates", []))

        decop_out = run_decop_tool(
            verbatim_passage=chunk_text_str,
            paraphrase_candidates=candidates,
            book_title=book_title,
            author=AUTHOR,
            client=client,
            n_permutations=3,
        )
        if not decop_out["skipped"]:
            decop_result = decop_out["result"]
        else:
            print(f"    [SKIP DE-COP] {decop_out['reason']}")

        simia_out = run_simia_tool(chunk_text_str, client, n_samples=1, max_words=10)
        if not simia_out["skipped"]:
            simia_raw = simia_out["result"]
        else:
            print(f"    [SKIP SiMIA] {simia_out['reason']}")

        dualtest_out = run_dualtest_tool(
            chunk_text_str,
            client,
            reference_model_name=settings.reference_model_name,
            prefix_len=50,
            continuation_len=32,
            max_new_tokens=32,
            label=label,
        )
        if not dualtest_out["skipped"]:
            dualtest_row = dualtest_out["result"]
        else:
            print(f"    [SKIP DUALTEST] {dualtest_out['reason']}")

    return combine_scores(dualtest_row=dualtest_row, simia_raw=simia_raw, decop_result=decop_result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks-per-text", type=int, default=2)
    parser.add_argument("--no-sage", action="store_true")
    args = parser.parse_args()

    client = try_build_target_client()
    text_results = []

    for title, chunks in load_chunks_by_title().items():
        label = int(chunks[0]["label"])
        print(f"\n=== {title} (label={label}, {'member' if label else 'non-member'}) ===")
        print(f"  {len(chunks)} chunks de ~128 tokens disponibles (ya limpios/chunkeados)")

        chunk_results = []
        for chunk in chunks[: args.max_chunks_per_text]:
            chunk_str = chunk["text"]
            sage_result = try_run_sage(chunk_str, use_sage=not args.no_sage)
            chunk_score = score_chunk(chunk_str, title, label, client, sage_result)
            chunk_results.append(chunk_score)
            print(f"    chunk -> final_probability={chunk_score['final_probability']}")

        write_run_artifact(RUN_ID, "results", title.replace(" ", "_"), {
            "title": title, "label": label, "chunk_results": chunk_results,
        })
        text_rollup = aggregate_chunk_scores(chunk_results)
        text_rollup.update({"title": title, "label": label})
        text_results.append(text_rollup)
        print(f"  -> probabilidad a nivel texto: {text_rollup['text_probability']} "
              f"({text_rollup['n_chunks_scored']}/{text_rollup['n_chunks_total']} chunks con score)")

    author_rollup = aggregate_text_scores(text_results)
    write_run_artifact(RUN_ID, "results", "author_final", {"author": AUTHOR, **author_rollup, "texts": text_results})

    print(f"\n=== Resultado final para {AUTHOR} ===")
    print(f"Probabilidad de membership: {author_rollup['author_probability']} "
          f"({author_rollup['n_texts_scored']}/{author_rollup['n_texts_total']} textos con score)")
    if author_rollup["author_probability"] is None:
        print(
            "\n(Ningun metodo MIA pudo correr -- falta GROQ_API_KEY en .env. El pipeline "
            "de limpieza/chunking/ensemble corrio igual; ver runs/manual_phase1_smoke_test/ "
            "para los artifacts.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
