"""
sage_cache.py — cache cross-run de parafraseos SAGE. Gemma-2B en CPU tarda ~30 min
por chunk; el mismo texto con los mismos params siempre produce los mismos candidatos
(T5 con beam search, determinista). Se guarda en runs/_sage_cache/ igual que _api_cache.

Clave: SHA256(text + n_generated + n_kept) — si cambian los hiperparámetros de SAGE,
el cache miss es automático y se recomputa.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mia_common.settings import settings

SAGE_CACHE_DIR: Path = settings.runs_dir / "_sage_cache"


def _key(text: str, n_generated: int, n_kept: int) -> str:
    blob = json.dumps({"text": text, "n_generated": n_generated, "n_kept": n_kept}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def get(text: str, n_generated: int, n_kept: int) -> list[str] | None:
    path = SAGE_CACHE_DIR / f"{_key(text, n_generated, n_kept)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["candidates"]


def put(text: str, n_generated: int, n_kept: int, candidates: list[str]) -> None:
    SAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAGE_CACHE_DIR / f"{_key(text, n_generated, n_kept)}.json"
    path.write_text(json.dumps({
        "candidates": candidates,
        "n_generated": n_generated,
        "n_kept": n_kept,
        "text_preview": text[:120],
    }, indent=2, ensure_ascii=False))
