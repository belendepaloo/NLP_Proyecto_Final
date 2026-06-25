"""
cache.py — cache de llamadas a APIs externas (request + response), para no re-gastar
presupuesto recalculando la misma llamada dos veces. Instruccion del usuario
(2026-06-25): TODA llamada a una API externa se persiste, sin excepcion.

Calls deterministicas (temperature=0, el caso de DE-COP y DUALTEST): se cachean por
contenido sin mas — un hit reusa siempre la respuesta anterior, que es correcto porque
se espera la MISMA salida para el mismo input.

Calls con sampling (temperature>0, el caso de SiMIA): cachear solo por contenido
colapsaria las N muestras independientes que el metodo necesita en una sola respuesta
repetida. El caller tiene que pasar `sample_index` (0..N-1) para que cada muestra
tenga su propio slot estable -- asi una rerun exacta del mismo experimento reproduce
las mismas N muestras sin gastar API de nuevo, pero dentro de un mismo calculo las N
muestras siguen siendo llamadas independientes la primera vez que se piden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mia_common.settings import settings

CACHE_DIR: Path = settings.runs_dir / "_api_cache"


def _cache_key(provider: str, model: str, payload: dict, sample_index: int | None) -> str:
    blob = json.dumps(
        {"provider": provider, "model": model, "payload": payload, "sample_index": sample_index},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def get(provider: str, model: str, payload: dict, sample_index: int | None = None) -> Any | None:
    path = CACHE_DIR / f"{_cache_key(provider, model, payload, sample_index)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["response"]


def put(provider: str, model: str, payload: dict, response: Any, sample_index: int | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(provider, model, payload, sample_index)}.json"
    path.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "request": payload,
                "sample_index": sample_index,
                "response": response,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
