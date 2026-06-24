"""
combine.py — ensemble simple, no aprendizaje online (alcance academico, ver el plan).
Promedio pesado de los 3 metodos, renormalizado sobre los que efectivamente respondieron
para un chunk dado (si uno abstiene, no cuenta como 0 -- los pesos se renormalizan).
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from mia_common.settings import settings

WEIGHTS_PATH = settings.ensemble_weights_path


def load_weights() -> dict[str, float]:
    return yaml.safe_load(Path(WEIGHTS_PATH).read_text())


def normalize_dualtest(row: dict) -> float:
    """row tiene p_rlb/p_esb en [0,1] (mas bajo = mas sospechoso de memorizacion, ver
    DUALTEST/metrics.py). Proxy graduado: 1 - min(p_rlb, p_esb), clippeado a [0,1].
    La decision calibrada binaria (DUALTEST/calibration.py) se muestra aparte en el
    detalle por transparencia, pero no es lo que entra a esta cuenta."""
    p = min(row.get("p_rlb", 1.0), row.get("p_esb", 1.0))
    return max(0.0, min(1.0, 1.0 - p))


def normalize_simia(raw: float | None, k: float = 1.0) -> float | None:
    """simmia_score devuelve -mean(ratios), sin acotar. sigmoid(k*raw) centrado en 0
    (ratio=1 -> raw=0 -> 0.5, "sin senal"). `k` es un placeholder razonado, no
    calibrado contra datos etiquetados -- documentar en SKILL.md (Fase 3) si se ajusta."""
    if raw is None:
        return None
    return 1.0 / (1.0 + math.exp(-k * raw))


def normalize_decop(accuracy: float) -> float:
    """Ya esta en [0,1] (fraccion de permutaciones correctas). OJO: 0.25 es chance
    level con 4 opciones, no 0 -- el PESO de decop, no un rescalado aca, es lo que debe
    absorber eso hasta que haya calibracion real (Fase 3)."""
    return accuracy


def combine_scores(
    dualtest_row: dict | None,
    simia_raw: float | None,
    decop_result: dict | None,
    weights: dict[str, float] | None = None,
) -> dict:
    """Combina los scores de un solo chunk. Cualquiera de los tres puede venir None
    (metodo abstuvo/fue skippeado para este chunk, ver agents/tools/mia_tools.py)."""
    weights = weights or load_weights()
    scores: dict[str, float] = {}
    used_weights: dict[str, float] = {}

    if dualtest_row is not None:
        scores["dualtest"] = normalize_dualtest(dualtest_row)
        used_weights["dualtest"] = weights["dualtest"]

    if simia_raw is not None:
        s = normalize_simia(simia_raw)
        if s is not None:
            scores["simia"] = s
            used_weights["simia"] = weights["simia"]

    if decop_result is not None:
        scores["decop"] = normalize_decop(decop_result["accuracy"])
        used_weights["decop"] = weights["decop"]

    if not scores:
        return {"final_probability": None, "per_method": {}, "weights_used": {}, "reason": "todos los metodos abstuvieron"}

    total_w = sum(used_weights.values())
    final = sum(scores[m] * used_weights[m] for m in scores) / total_w

    return {
        "final_probability": final,
        "per_method": {m: {"normalized": scores[m], "weight": used_weights[m]} for m in scores},
        "weights_used": dict(used_weights),
        "reason": None,
    }


def aggregate_chunk_scores(chunk_results: list[dict]) -> dict:
    """Rollup chunk -> texto. Media simple sobre los chunks con final_probability no
    nulo (excluye los que abstuvieron en los 3 metodos). Simplificacion deliberada --
    revisar si hay textos con muchos mas chunks sobrevivientes que otros (ver riesgos
    del plan)."""
    probs = [c["final_probability"] for c in chunk_results if c.get("final_probability") is not None]
    if not probs:
        return {"text_probability": None, "n_chunks_scored": 0, "n_chunks_total": len(chunk_results)}
    return {
        "text_probability": sum(probs) / len(probs),
        "n_chunks_scored": len(probs),
        "n_chunks_total": len(chunk_results),
    }


def aggregate_text_scores(text_results: list[dict]) -> dict:
    """Rollup texto -> autor. Misma logica (media simple) que aggregate_chunk_scores."""
    probs = [t["text_probability"] for t in text_results if t.get("text_probability") is not None]
    if not probs:
        return {"author_probability": None, "n_texts_scored": 0, "n_texts_total": len(text_results)}
    return {
        "author_probability": sum(probs) / len(probs),
        "n_texts_scored": len(probs),
        "n_texts_total": len(text_results),
    }
