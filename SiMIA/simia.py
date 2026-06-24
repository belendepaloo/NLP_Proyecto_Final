"""
simia.py — version importable de notebooks/simMIA.ipynb.

Mismo algoritmo que el notebook (next-word black-box ratio test: compara que tan bien
predice el modelo la siguiente palabra real bajo un prefijo "normal" vs. bajo un
prefijo perturbado por un texto non-member). El notebook media AUC~0.5-0.6 en
BookTection -- senal debil, asumida como tal por el resto del pipeline (ver pesos en
agents/ensemble/weights.yaml).

Diferencia con el notebook: en vez de un cliente Groq global hardcodeado, recibe un
mia_common.target_client.TargetClient inyectado (cualquier provider).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from mia_common.target_client import TargetClient

_embedder: SentenceTransformer | None = None  # lazy singleton, carga no es gratis


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


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


def model_generate_fn_factory(client: TargetClient, max_tokens: int = 5) -> Callable[[str], str]:
    """Misma prompt que el notebook: pide SOLO la siguiente palabra dado un prefijo."""

    def generate(prefix: str) -> str:
        prompt = (
            "Complete the following text with ONLY the next word.\n\n"
            f"Text:\n{prefix}\n\nNext word:"
        )
        out = client.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_tokens,
            temperature=0.7,
        )
        out = out.strip()
        return out.split()[0] if out else ""

    return generate


def simmia_score(
    text: str,
    client: TargetClient,
    non_member_prefix: str = "",
    n_samples: int = 3,
    max_words: int = 20,
    sleep_between_calls: float = 0.0,
) -> float | None:
    """Mismo algoritmo que simmia_score() en notebooks/simMIA.ipynb.

    Por cada posicion i en las primeras `max_words` palabras del texto: compara la
    similitud promedio (embedding) entre la palabra real y `n_samples` continuaciones
    generadas por el modelo bajo el prefijo "normal" vs. bajo el prefijo perturbado
    (non_member_prefix + prefijo). Devuelve -mean(ratios); None si no se pudo calcular
    ningun ratio (abstencion -- el ensemble debe tratarlo como "sin senal", no como 0).
    """
    generate = model_generate_fn_factory(client)
    words = text.split()[:max_words]
    ratios = []

    for i in range(1, len(words)):
        prefix = " ".join(words[:i])
        target_word = words[i]

        normal_samples = []
        perturbed_samples = []
        for _ in range(n_samples):
            normal_samples.append(generate(prefix))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)
            perturbed_samples.append(generate(f"{non_member_prefix} {prefix}"))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)

        normal_score = _word_score(target_word, normal_samples)
        if normal_score > 1e-8:
            perturbed_score = _word_score(target_word, perturbed_samples)
            ratios.append(perturbed_score / normal_score)

    return float(-np.mean(ratios)) if ratios else None
