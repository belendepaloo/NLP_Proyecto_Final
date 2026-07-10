#!/usr/bin/env python3
"""
Agrega SAGE + DE-COP a chunks que ya tienen DUALTEST pero no tienen DE-COP.
Uso: python scripts/add_decop_to_run.py --run-id webapp_jane_austen_6d14be70
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, write_run_artifact
from mia_common.settings import settings
from mia_common.target_client import make_target_client_pool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_id = args.run_id

    artifacts = list_run_artifacts(run_id)

    # Chunks con DUALTEST pero sin DE-COP
    pending = []
    for fname in artifacts.get("mia_scores", []):
        chunk_id = fname.removesuffix(".json")
        mia = read_run_artifact(run_id, "mia_scores", chunk_id)
        if mia.get("decop") is not None:
            print(f"  [SKIP] {chunk_id} ya tiene DE-COP")
            continue
        try:
            chunk = read_run_artifact(run_id, "curation", f"chunk_{chunk_id}")
            pending.append({"chunk_id": chunk_id, "text": chunk["text"],
                            "document_id": chunk.get("document_id", ""),
                            "existing_mia": mia})
        except Exception as e:
            print(f"  [WARN] no se pudo cargar chunk_{chunk_id}: {e}")

    print(f"\n{len(pending)} chunks para agregar DE-COP\n")
    if not pending:
        return 0

    from agents.tools.sage_tools import get_sage_candidates

    # Pool Groq para DE-COP
    pool = make_target_client_pool(
        provider=settings.target_provider,
        model_name=settings.target_model_name,
        api_keys=settings.groq_api_keys(),
        min_seconds_between_calls=settings.target_min_seconds_between_calls,
        max_retries=settings.target_max_retries,
    )
    from webapp.run_manager import _guess_author_from_run_id
    author = _guess_author_from_run_id(run_id)

    for i, chunk in enumerate(pending):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        print(f"[{i+1}/{len(pending)}] {chunk_id} — SAGE...", flush=True)

        candidates: list[str] = []
        try:
            candidates = get_sage_candidates(text)
            write_run_artifact(run_id, "sage", f"paraphrase_{chunk_id}",
                               {"chunk_id": chunk_id, "paraphrase_candidates": candidates})
            print(f"  SAGE OK ({len(candidates)} candidates)", flush=True)
        except Exception as e:
            print(f"  SAGE ERROR: {e}", flush=True)

        decop_result = None
        if len(candidates) >= 3:
            print(f"  DE-COP...", flush=True)
            try:
                from agents.tools.mia_tools import run_decop_tool
                out = run_decop_tool(
                    verbatim_passage=text,
                    paraphrase_candidates=candidates,
                    book_title=chunk["document_id"],
                    author=author,
                    client=pool.get(),
                    n_permutations=3,
                )
                if not out["skipped"]:
                    decop_result = out["result"]
                    print(f"  DE-COP OK: accuracy={decop_result.get('accuracy')}", flush=True)
                else:
                    print(f"  DE-COP skipped: {out.get('reason')}", flush=True)
            except Exception as e:
                print(f"  DE-COP ERROR: {e}", flush=True)
        else:
            print(f"  DE-COP skipped: solo {len(candidates)} candidates", flush=True)

        # Actualizar mia_scores conservando DUALTEST existente
        updated = {**chunk["existing_mia"], "decop": decop_result}
        write_run_artifact(run_id, "mia_scores", chunk_id, updated)
        print(f"  mia_scores actualizado", flush=True)

    print(f"\nListo. Recargá la página para ver DE-COP en la tabla.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
