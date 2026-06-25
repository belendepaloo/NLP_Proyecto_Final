"""
simia.py — implementacion de SimMIA, fiel al paper "Membership Inference on LLMs in
the Wild" (Yi & Li, CUHK, arXiv:2601.11314, 2026), que lo presenta como nuevo SOTA en
black-box MIA. Refactor de notebooks/simMIA.ipynb (que ya tenia la formula correcta)
con dos bugs reales arreglados (encontrados 2026-06-25 al revisar contra el paper):

  1. El pipeline llamaba a esto con non_member_prefix="" (default) -- la condicion
     "perturbada" terminaba siendo casi identica a la normal, dando ratio~1 siempre
     y por lo tanto una senal CONSTANTE sin importar membership. Ahora carga por
     default un prefijo non-member FIJO real (ver non_member_calibration.txt /
     scripts/build_simia_calibration_set.py), tal como pide el paper ("fixed
     non-member prefixes reduce variance" vs. eleccion aleatoria).
  2. n_samples=1 en el pipeline -- el paper usa N=10 (reducido desde 100 "to reduce
     API costs") para estimar s(xi|x<i)=E[sim(xi,x_i_hat)]; con N=1 esa esperanza es
     en realidad una sola muestra, much{isimo} ruido. Default ahora en
     mia_common.settings.simia_n_samples (10).

Formula (igual a la del paper, ya estaba bien):
    sim(a,b)    = (cos(Enc(a),Enc(b)) + 1) / 2                  -- embeddings, [0,1]
    s(xi|x<i)   = mean_j sim(xi, sample_j)                       -- E sobre N samples
    SimMIA(x)   = -1/L * sum_i [ s(xi|P (+) x<i) / s(xi|x<i) ]    -- P = prefijo non-member fijo

Mayor score (menos negativo / mas alto) = mas evidencia de membership, segun el paper.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from mia_common.settings import settings
from mia_common.target_client import TargetClient

_embedder: SentenceTransformer | None = None  # lazy singleton, carga no es gratis
_CALIBRATION_PATH = Path(__file__).resolve().parent / "non_member_calibration.txt"
_default_non_member_prefix: str | None = None  # lazy, cacheado tras la primera carga


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def get_default_non_member_prefix(max_chars: int | None = None) -> str:
    """Prefijo non-member FIJO (ver scripts/build_simia_calibration_set.py) -- 5
    capitulos de Royal Road posteriores al cutoff de Llama 3.1, DISTINTOS de los
    non-member usados en el dataset de evaluacion, para que la perturbacion nunca se
    solape con lo que se esta midiendo."""
    global _default_non_member_prefix
    if _default_non_member_prefix is None:
        if not _CALIBRATION_PATH.exists():
            raise FileNotFoundError(
                f"{_CALIBRATION_PATH} no existe -- correr "
                "scripts/build_simia_calibration_set.py primero."
            )
        _default_non_member_prefix = _CALIBRATION_PATH.read_text(encoding="utf-8")
    max_chars = max_chars if max_chars is not None else settings.simia_calibration_chars
    return _default_non_member_prefix[:max_chars]


def _similarity(w1: str, w2: str) -> float:
    if not w1 or not w2:
        return 0.0
    embedder = _get_embedder()
    e1 = embedder.encode([w1])
    e2 = embedder.encode([w2])
    cos = cosine_similarity(e1, e2)[0][0]
    return (cos + 1) / 2


def _word_score(target_word: str, sampled_words: list[str]) -> float:
    sampled = [w for w in sampled_words if w]
    if not sampled:
        return 0.0
    return float(np.mean([_similarity(target_word, w) for w in sampled]))


def model_generate_fn_factory(client: TargetClient, max_tokens: int = 5) -> Callable[..., str]:
    """Misma prompt que el paper/notebook: pide SOLO la siguiente palabra dado un
    prefijo. `sample_index` se pasa al cliente para que el cache (mia_common.cache)
    le de un slot propio a cada una de las N muestras independientes -- sin esto, N
    llamadas con el mismo prompt pegarian todas contra el mismo cache hit."""

    def generate(prefix: str, sample_index: int | None = None) -> str:
        prompt = (
            "Complete the following text with ONLY the next word.\n\n"
            f"Text:\n{prefix}\n\nNext word:"
        )
        out = client.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_tokens,
            temperature=0.7,
            cache_sample_index=sample_index,
        )
        out = out.strip()
        return out.split()[0] if out else ""

    return generate


def simmia_score(
    text: str,
    client: TargetClient,
    non_member_prefix: str | None = None,
    n_samples: int | None = None,
    max_words: int = 20,
    sleep_between_calls: float = 0.0,
) -> float | None:
    """SimMIA(x) = -1/L * sum_i [s(xi|P+x<i) / s(xi|x<i)], ver docstring del modulo.

    `non_member_prefix=None` (default) carga el prefijo de calibracion fijo via
    get_default_non_member_prefix() -- pasar "" explicito solo si de verdad se quiere
    desactivar la perturbacion (no deberia hacerse en uso normal, rompe la formula).
    `n_samples=None` (default) usa mia_common.settings.simia_n_samples.

    Por cada posicion i en las primeras `max_words` palabras del texto: compara la
    similitud promedio (embedding) entre la palabra real y N continuaciones generadas
    por el modelo bajo el prefijo normal vs. bajo el prefijo perturbado. Devuelve
    -mean(ratios); None si no se pudo calcular ningun ratio (abstencion -- el ensemble
    debe tratarlo como "sin senal", no como 0)."""
    if non_member_prefix is None:
        non_member_prefix = get_default_non_member_prefix()
    if n_samples is None:
        n_samples = settings.simia_n_samples

    generate = model_generate_fn_factory(client)
    words = text.split()[:max_words]
    ratios = []

    for i in range(1, len(words)):
        prefix = " ".join(words[:i])
        target_word = words[i]

        normal_samples = []
        perturbed_samples = []
        for s in range(n_samples):
            normal_samples.append(generate(prefix, sample_index=s))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)
            perturbed_samples.append(generate(f"{non_member_prefix} {prefix}", sample_index=s))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)

        normal_score = _word_score(target_word, normal_samples)
        if normal_score > 1e-8:
            perturbed_score = _word_score(target_word, perturbed_samples)
            ratios.append(perturbed_score / normal_score)

    return float(-np.mean(ratios)) if ratios else None
