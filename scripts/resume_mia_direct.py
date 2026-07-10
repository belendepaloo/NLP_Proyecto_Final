#!/usr/bin/env python3
"""
resume_mia_direct.py — corre MIA scoring directamente sobre los chunks ya curados
de un run existente, sin pasar por LangGraph.

Uso para retomar runs atascados que ya completaron la curacion pero nunca llegaron
a mia_agent:

    python scripts/resume_mia_direct.py --run-id webapp_jane_austen_5e58bc59

El script:
1. Lee todos los voice verdicts "keep" de runs/<run_id>/curation/voice_*.json
2. Saltea SAGE y DE-COP (evita el problema de RAM del modelo local)
3. Corre DUALTEST (Qwen/Qwen2.5-0.5B local + Groq) para cada chunk
4. Escribe mia_scores/<chunk_id>.json en el mismo run_id

Despues de correr esto, la webapp muestra el resultado final sin necesidad
de que el agente LangGraph lo haga.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact  # noqa: E402
from mia_common.settings import settings  # noqa: E402
from mia_common.target_client import DailyCapError, make_target_client_pool  # noqa: E402

_print_lock = threading.Lock()


def load_keep_chunks(run_id: str) -> list[dict]:
    """Lee todos los voice verdicts 'keep' y carga el texto del chunk correspondiente."""
    runs_dir = settings.runs_dir / run_id / "curation"
    keep = []
    for fpath in sorted(runs_dir.glob("voice_*.json")):
        verdict = json.loads(fpath.read_text())
        if verdict.get("decision") == "keep":
            chunk_id = verdict["chunk_id"]
            try:
                chunk = read_run_artifact(run_id, "curation", f"chunk_{chunk_id}")
                keep.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id", chunk_id.rsplit("_", 1)[0]),
                    "text": chunk["text"],
                })
            except (FileNotFoundError, KeyError) as e:
                print(f"  [WARN] chunk_{chunk_id}.json no encontrado o sin 'text': {e}")
    return keep


def already_scored(run_id: str, chunk_id: str) -> bool:
    artifacts = list_run_artifacts(run_id)
    return f"{chunk_id}.json" in artifacts.get("mia_scores", [])


def run_sage_for_chunk(chunk_id: str, text: str) -> list[str]:
    """Corre SAGE y persiste el artifact. Devuelve la lista plana de paraphrase candidates."""
    from agents.tools.sage_tools import run_sage_tool
    result = run_sage_tool(text)
    candidates = [c["text"] for seg in result.get("segments", []) for c in seg.get("all_candidates", [])]
    write_run_artifact(run_id_global, "sage", f"paraphrase_{chunk_id}", {
        "chunk_id": chunk_id,
        "paraphrase_candidates": candidates,
        **result,
    })
    return candidates


def score_chunk(run_id: str, chunk: dict, pool, reference_model_name: str, use_sage: bool) -> dict:
    client = pool.get()
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    # SAGE + DE-COP (opcional)
    decop_result = None
    simia_raw = None

    if use_sage:
        try:
            with _print_lock:
                print(f"  [{chunk_id}] SAGE...")
            candidates = run_sage_for_chunk(chunk_id, text)
            with _print_lock:
                print(f"  [{chunk_id}] SAGE OK ({len(candidates)} candidates) → DE-COP...")
            from agents.tools.mia_tools import run_decop_tool
            out = run_decop_tool(
                verbatim_passage=text,
                paraphrase_candidates=candidates,
                book_title=chunk.get("document_id", "unknown"),
                author="Jane Austen",
                client=client,
                n_permutations=3,
            )
            if not out["skipped"]:
                decop_result = out["result"]
            else:
                with _print_lock:
                    print(f"  [{chunk_id}] DE-COP skipped: {out.get('reason')}")
        except Exception as e:
            with _print_lock:
                print(f"  [{chunk_id}] SAGE/DE-COP error: {e}")

    # DUALTEST: Qwen/Qwen2.5-0.5B (local) + Groq (cacheado si ya corrió)
    dualtest_row = None
    try:
        from agents.tools.mia_tools import run_dualtest_tool
        out = run_dualtest_tool(
            text=text,
            client=client,
            reference_model_name=reference_model_name,
            prefix_len=settings.dualtest_prefix_len,
            continuation_len=settings.dualtest_continuation_len,
            max_new_tokens=settings.dualtest_max_new_tokens,
            label=0,
        )
        if not out["skipped"]:
            dualtest_row = out["result"]
        else:
            with _print_lock:
                print(f"  [{chunk_id}] DUALTEST skipped: {out.get('reason')}")
    except Exception as e:
        with _print_lock:
            print(f"  [{chunk_id}] DUALTEST error: {e}")

    write_run_artifact(run_id, "mia_scores", chunk_id, {"decop": decop_result, "simia": simia_raw, "dualtest": dualtest_row})

    from agents.ensemble.combine import combine_scores
    combined = combine_scores(dualtest_row=dualtest_row, simia_raw=simia_raw, decop_result=decop_result)
    with _print_lock:
        print(f"  [{chunk_id}] final_probability={combined['final_probability']}")
    return {"chunk_id": chunk_id, **combined}


# global para run_sage_for_chunk (SAGE usa un lock interno, no puede recibir run_id fácil)
run_id_global: str = ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-score chunks ya procesados")
    parser.add_argument("--n-chunks", type=int, default=None, help="Limitar a N chunks (por defecto todos)")
    parser.add_argument("--no-sage", action="store_true", help="Saltear SAGE y DE-COP")
    args = parser.parse_args()

    global run_id_global
    run_id = args.run_id
    run_id_global = run_id
    use_sage = not args.no_sage
    print(f"\n=== resume_mia_direct para run: {run_id} ===\n")

    # Verificar que el run existe
    run_dir = settings.runs_dir / run_id
    if not run_dir.exists():
        print(f"ERROR: runs/{run_id}/ no existe")
        return 1

    # Cargar chunks keep
    keep_chunks = load_keep_chunks(run_id)
    if not keep_chunks:
        print("ERROR: no hay chunks con decision='keep' en curation/")
        return 1
    print(f"Chunks a puntuar: {len(keep_chunks)}")

    # Limitar cantidad
    if args.n_chunks:
        keep_chunks = keep_chunks[: args.n_chunks]
        print(f"Limitando a {args.n_chunks} chunks (--n-chunks)")

    # Filtrar ya procesados (a menos que --force)
    if not args.force:
        pending = [c for c in keep_chunks if not already_scored(run_id, c["chunk_id"])]
        skipped_count = len(keep_chunks) - len(pending)
        if skipped_count:
            print(f"  ({skipped_count} ya tienen mia_scores, saltear -- usa --force para re-scorear)")
        keep_chunks = pending

    if not keep_chunks:
        print("Todos los chunks ya tienen score. Nada que hacer.")
        return 0

    # Construir pool de clientes Groq
    keys = settings.groq_api_keys()
    if not keys:
        print("ERROR: GROQ_API_KEY/GROQ_API_KEYS no configurada en .env")
        return 1
    pool = make_target_client_pool(
        provider=settings.target_provider,
        model_name=settings.target_model_name,
        api_keys=keys,
        min_seconds_between_calls=settings.target_min_seconds_between_calls,
        max_retries=settings.target_max_retries,
    )
    print(f"Pool de {len(pool)} key(s) ({settings.target_provider}/{settings.target_model_name})")

    # Pre-cargar modelos locales UNA vez
    print(f"Cargando reference model DUALTEST: {settings.reference_model_name} ...")
    try:
        from agents.tools.mia_tools import get_reference_model
        get_reference_model(settings.reference_model_name)
        print("  OK")
    except Exception as e:
        print(f"  ERROR cargando reference model: {e}")
        print("  DUALTEST no va a poder correr. Abortando.")
        return 1

    if use_sage:
        print("Cargando SAGE (Gemma-2B)...")
        try:
            from agents.tools.sage_tools import _get_sage
            _get_sage(n_candidates_generated=settings.sage_n_candidates_generated,
                      n_candidates_kept=settings.sage_n_candidates_kept)
            print("  OK")
        except Exception as e:
            print(f"  ERROR cargando SAGE: {e}")
            print("  Usando --no-sage automáticamente.")
            use_sage = False

    # SAGE es single-threaded (lock interno); DE-COP+DUALTEST pueden ser paralelos
    # pero como SAGE serializa, usar 1 worker evita contención y es más claro
    workers = args.workers or (1 if use_sage else len(pool))
    print(f"\nProcesando {len(keep_chunks)} chunks con {workers} worker(s) "
          f"({'SAGE+DE-COP+DUALTEST' if use_sage else 'solo DUALTEST'})...\n")

    results = []
    daily_cap_hit = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(score_chunk, run_id, chunk, pool, settings.reference_model_name, use_sage): chunk["chunk_id"]
            for chunk in keep_chunks
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except DailyCapError as e:
                if not daily_cap_hit:
                    daily_cap_hit = True
                    print(f"\n[DAILY CAP] {e}\nGuardando resultados parciales...")
                for f in futures:
                    f.cancel()

    print(f"\n=== Resultado final ===")
    if daily_cap_hit:
        print("(PARCIAL -- se cortó por límite diario de Groq)")

    n_scored = sum(1 for r in results if r.get("final_probability") is not None)
    probs = [r["final_probability"] for r in results if r.get("final_probability") is not None]
    author_prob = sum(probs) / len(probs) if probs else None
    print(f"Chunks con score: {n_scored}/{len(keep_chunks)}")
    print(f"Probabilidad estimada (promedio simple): {author_prob}")
    print(f"\nArtifacts escritos en runs/{run_id}/mia_scores/")
    print("La webapp ahora muestra el resultado al visitar el run.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
