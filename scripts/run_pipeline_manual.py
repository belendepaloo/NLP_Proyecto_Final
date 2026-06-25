#!/usr/bin/env python3
"""
run_pipeline_manual.py — milestone demostrable de la Fase 1: pipeline determinista
end-to-end (chunks ya limpios -> SAGE -> DE-COP/SiMIA/DUALTEST -> ensemble) SIN
agentes y SIN scraping/curacion automatica.

Usa processRawText/Datasets/dataset_len128.csv (ver scripts/expand_dataset.py: 5
novelas de Gutenberg como member + 5 capitulos de Royal Road de 2025/2026 como
non-member) en vez de re-correr clean_html_tool/chunk_text_tool sobre el HTML crudo
completo: chunk_text (pysbd) escala mal sobre un libro entero de una sola vez (~5 min
medido) -- clean_html_tool/chunk_text_tool ya se probaron por separado sobre slices
mas chicos y funcionan; este script los reutiliza para texto NUEVO que llegue del
scraping (Fase 2), no para re-procesar lo que ya esta chunkeado.

Paralelizado: los chunks se procesan concurrentemente repartidos sobre un pool de
clientes target (uno por GROQ_API_KEY disponible, ver mia_common.settings.groq_api_keys
y TargetClientPool) -- con una sola key sigue siendo correcto, solo que sin paralelismo
real contra la API (el cache + los locks de SAGE/DUALTEST hacen que esto sea seguro en
cualquier caso). Todo resultado crudo por metodo (no solo el score final del ensemble)
se persiste en runs/<run_id>/results/ -- toda llamada a la API ya se cachea aparte en
runs/_api_cache/ (ver mia_common/cache.py).

Pensado para correr de punta a punta una vez que haya GROQ_API_KEY/GROQ_API_KEYS (.env)
y, si se quiere probar SAGE de verdad, transformer_lens+sae_lens instalados. Sin esas
dos cosas el script sigue corriendo y muestra claramente que etapas se saltearon y por
que -- no es un requisito tener todo instalado para ver la forma del pipeline.

Uso:
    python scripts/run_pipeline_manual.py [--chunks-per-text N] [--seed N] [--no-sage] [--workers N]
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ensemble.combine import aggregate_chunk_scores, aggregate_text_scores, combine_scores  # noqa: E402
from agents.tools.fs_tools import write_run_artifact  # noqa: E402
from mia_common.settings import settings  # noqa: E402
from mia_common.target_client import DailyCapError, TargetClientPool, make_target_client_pool  # noqa: E402

DATASET_CSV = Path(__file__).resolve().parents[1] / "processRawText" / "Datasets" / "dataset_len128.csv"
RUN_ID = "manual_phase1_smoke_test"
AUTHOR = "Charles Dickens"

_print_lock = threading.Lock()  # log entrelazado de threads -> imprimir de a una linea


def load_chunks_by_title() -> dict[str, list[dict]]:
    df = pd.read_csv(DATASET_CSV)
    return {title: g.to_dict("records") for title, g in df.groupby("title")}


def try_build_target_pool() -> TargetClientPool | None:
    keys = settings.groq_api_keys()
    if not keys:
        print("  [SKIP target client] GROQ_API_KEY/GROQ_API_KEYS no configurada -- ver .env.example.")
        return None
    pool = make_target_client_pool(
        provider=settings.target_provider,
        model_name=settings.target_model_name,
        api_keys=keys,
        min_seconds_between_calls=settings.target_min_seconds_between_calls,
        max_retries=settings.target_max_retries,
    )
    print(f"  Pool de {len(pool)} key(s) listo ({settings.target_provider}/{settings.target_model_name}).")
    return pool


def try_run_sage(text: str, use_sage: bool) -> dict | None:
    if not use_sage:
        return None
    try:
        from agents.tools.sage_tools import run_sage_tool

        return run_sage_tool(text)
    except ImportError as e:
        with _print_lock:
            print(f"  [SKIP SAGE] {e}")
        return None


def score_chunk(chunk_text_str: str, book_title: str, label: int, client, sage_result: dict | None) -> dict:
    """Corre los 3 metodos sobre un chunk y devuelve TANTO el resultado crudo de cada
    uno como el score combinado del ensemble -- todo se persiste, no solo el final."""
    from agents.tools.mia_tools import run_decop_tool, run_dualtest_tool, run_simia_tool

    decop_out = simia_out = dualtest_out = None
    decop_result = simia_raw = dualtest_row = None

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

        simia_out = run_simia_tool(chunk_text_str, client, max_words=10)
        if not simia_out["skipped"]:
            simia_raw = simia_out["result"]

        dualtest_out = run_dualtest_tool(
            chunk_text_str,
            client,
            reference_model_name=settings.reference_model_name,
            prefix_len=settings.dualtest_prefix_len,
            continuation_len=settings.dualtest_continuation_len,
            max_new_tokens=settings.dualtest_max_new_tokens,
            label=label,
        )
        if not dualtest_out["skipped"]:
            dualtest_row = dualtest_out["result"]

    ensemble = combine_scores(dualtest_row=dualtest_row, simia_raw=simia_raw, decop_result=decop_result)
    return {
        "chunk_text": chunk_text_str,
        "decop": decop_out,
        "simia": simia_out,
        "dualtest": dualtest_out,
        "ensemble": ensemble,
    }


def process_one_chunk(
    title: str, label: int, chunk_idx: int, chunk_text_str: str, pool: TargetClientPool | None, use_sage: bool
) -> dict:
    client = pool.get() if pool is not None else None
    sage_result = try_run_sage(chunk_text_str, use_sage=use_sage)
    record = score_chunk(chunk_text_str, title, label, client, sage_result)
    record.update({"title": title, "label": label})
    # se escribe ACA, apenas termina este chunk -- no al final de todo el run, para no
    # perder el trabajo ya hecho si otro chunk en paralelo tira DailyCapError despues.
    write_run_artifact(RUN_ID, "chunks", f"{title.replace(' ', '_')}_{chunk_idx}", record)
    with _print_lock:
        decop_skip = record["decop"]["reason"] if record["decop"] and record["decop"]["skipped"] else None
        simia_skip = record["simia"]["reason"] if record["simia"] and record["simia"]["skipped"] else None
        dualtest_skip = record["dualtest"]["reason"] if record["dualtest"] and record["dualtest"]["skipped"] else None
        if decop_skip:
            print(f"    [{title}] [SKIP DE-COP] {decop_skip}")
        if simia_skip:
            print(f"    [{title}] [SKIP SiMIA] {simia_skip}")
        if dualtest_skip:
            print(f"    [{title}] [SKIP DUALTEST] {dualtest_skip}")
        print(f"    [{title}] chunk -> final_probability={record['ensemble']['final_probability']}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-per-text",
        type=int,
        default=settings.chunks_per_text,
        help="Cuantos chunks por libro entran al pipeline costoso (default: "
        "mia_common.settings.chunks_per_text, hoy %(default)s). Control de costo/computo: "
        "subilo/bajalo aca o cambia el default en mia_common/settings.py sin tocar mas nada.",
    )
    parser.add_argument("--seed", type=int, default=settings.chunk_sample_seed)
    parser.add_argument("--no-sage", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Chunks en paralelo. Default: cantidad de keys en el pool (1 si no hay "
        "GROQ_API_KEYS configurada, ver .env.example).",
    )
    args = parser.parse_args()

    pool = try_build_target_pool()
    rng = random.Random(args.seed)

    tasks: list[tuple[str, int, int, str]] = []
    for title, chunks in load_chunks_by_title().items():
        label = int(chunks[0]["label"])
        n_sample = min(args.chunks_per_text, len(chunks))
        # muestreo aleatorio (seed fijo, reproducible) en vez de los primeros N -- asi
        # no testeamos siempre solo el comienzo del libro.
        sampled = rng.sample(chunks, n_sample)
        print(f"=== {title} (label={label}, {'member' if label else 'non-member'}): "
              f"{len(chunks)} chunks disponibles -> {n_sample} seleccionados (seed={args.seed}) ===")
        tasks.extend((title, label, idx, c["text"]) for idx, c in enumerate(sampled))

    workers = args.workers or (len(pool) if pool is not None else 1)
    print(f"\nProcesando {len(tasks)} chunks con {workers} worker(s) en paralelo...\n")

    records_by_title: dict[str, list[dict]] = {}
    daily_cap_hit = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_chunk, title, label, idx, chunk_text_str, pool, not args.no_sage): title
            for title, label, idx, chunk_text_str in tasks
        }
        for future in as_completed(futures):
            try:
                record = future.result()
                records_by_title.setdefault(record["title"], []).append(record)
            except DailyCapError as e:
                if not daily_cap_hit:
                    daily_cap_hit = True
                    print(
                        f"\n[DAILY CAP] {e}\nSe llego al limite diario de tokens de Groq -- "
                        "cancelando tareas pendientes (best-effort) y guardando los resultados "
                        "parciales ya calculados en vez de perder todo el run."
                    )
                for pending in futures:
                    pending.cancel()  # solo afecta las que todavia no arrancaron

    text_results = []
    for title, records in records_by_title.items():
        label = records[0]["label"]
        chunk_results = [r["ensemble"] for r in records]
        write_run_artifact(RUN_ID, "results", title.replace(" ", "_"), {
            "title": title, "label": label, "chunk_records": records,
        })
        text_rollup = aggregate_chunk_scores(chunk_results)
        text_rollup.update({"title": title, "label": label})
        text_results.append(text_rollup)
        print(f"-> {title}: probabilidad a nivel texto = {text_rollup['text_probability']} "
              f"({text_rollup['n_chunks_scored']}/{text_rollup['n_chunks_total']} chunks con score)")

    author_rollup = aggregate_text_scores(text_results)
    write_run_artifact(RUN_ID, "results", "author_final", {"author": AUTHOR, **author_rollup, "texts": text_results})

    print(f"\n=== Resultado final para {AUTHOR} ===")
    if daily_cap_hit:
        print("(PARCIAL -- se corto por limite diario de Groq, ver mensaje [DAILY CAP] arriba)")
    print(f"Probabilidad de membership: {author_rollup['author_probability']} "
          f"({author_rollup['n_texts_scored']}/{author_rollup['n_texts_total']} textos con score)")
    if author_rollup["author_probability"] is None:
        print(
            "\n(Ningun metodo MIA pudo correr -- falta GROQ_API_KEY/GROQ_API_KEYS en .env. "
            "El pipeline de limpieza/chunking/ensemble corrio igual; ver "
            f"runs/{RUN_ID}/ para los artifacts.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
