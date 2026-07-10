"""
simia.py — implementacion de SimMIA portada 1:1 desde el notebook de referencia del
equipo (simmia_decop/notebooks/simMIA.ipynb, commit a63ec742 "simmia and decop done",
mergeado a main via PR #2 feature/decop_simmia). Formula, parametros y prompt son los
que decidio esa version; este modulo solo adapta la interfaz para que sea importable
desde agents/tools/mia_tools.py (misma firma publica de siempre: simmia_score(text,
client, non_member_prefix, n_samples, max_words)).

Formula (idem notebook, celda "SimMIA samples"):
    sim(a,b)            = (cos(Enc(a),Enc(b)) + 1) / 2                    -- embeddings, [0,1]
    normal_word_score    = mean_j sim(x_i, gen_j)             bajo prefijo x_<i
    perturbed_word_score = mean_j sim(x_i, gen_j)             bajo prefijo P + x_<i
    relative_word_score  = -perturbed_word_score / (normal_word_score + EPS)
    SimMIA(x)            = mean_i relative_word_score_i

Mayor score (menos negativo) = mas evidencia de membership.

Diferencias deliberadas respecto de la version anterior de este archivo (bugfix del
2026-06-25, ya no aplica porque se reemplazo la formula entera por la del notebook):
    - N_SAMPLES=3 (no 10): asi lo dejo el notebook, no el default anterior del paper.
    - temperature=1.0, top_p=1.0 (no 0.7).
    - prompt: system="Continue the text naturally. Output only the next word or very
      short continuation.", user=prefix (no "Complete the following text...").
    - EPS=1e-8 en el denominador en vez de descartar la posicion si normal_score<=1e-8.
    - non_member_prefix: el notebook lo arma muestreando 3 non-members al azar del
      dataset de evaluacion completo (build_non_member_prefix), algo que no existe como
      tal en el pipeline de agentes (que ve un chunk a la vez, no el dataset entero).
      Aca se preserva el mecanismo de prefijo FIJO pre-armado offline
      (SiMIA/non_member_calibration.txt, ver scripts/build_simia_calibration_set.py)
      como equivalente practico -- incluye texto non-member real, solo que elegido una
      vez de antemano en vez de al azar en cada llamada.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from mia_common.settings import settings
from mia_common.target_client import TargetClient

_EPS = 1e-8

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
    if not isinstance(w1, str) or not isinstance(w2, str):
        return 0.0
    if not w1.strip() or not w2.strip():
        return 0.0
    embedder = _get_embedder()
    e1 = embedder.encode([w1])
    e2 = embedder.encode([w2])
    cos = cosine_similarity(e1, e2)[0][0]
    return float((cos + 1) / 2)


def _first_word(text: str | None) -> str:
    """Idem first_word() del notebook: limpia puntuacion lider antes de tomar el
    primer token (ej. '"word' -> 'word'), a diferencia de un .split()[0] plano."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^[^\w]+", "", text)
    parts = text.split()
    return parts[0] if parts else ""


def _word_score(target_word: str, sampled_words: list[str]) -> float:
    sampled = [w for w in sampled_words if w]
    if not sampled:
        return 0.0
    return float(np.mean([_similarity(target_word, w) for w in sampled]))


def model_generate_fn_factory(client: TargetClient, max_tokens: int = 3) -> Callable[..., str]:
    """Mismo prompt que el notebook: pide continuar el texto naturalmente, no un
    formato de instruccion tipo 'Complete the following text'. `sample_index` se pasa
    al cliente para que el cache (mia_common.cache) le de un slot propio a cada una de
    las N muestras independientes -- sin esto, N llamadas con el mismo prompt pegarian
    todas contra el mismo cache hit."""

    def generate(prefix: str, sample_index: int | None = None) -> str:
        out = client.chat(
            [
                {
                    "role": "system",
                    "content": "Continue the text naturally. Output only the next word or very short continuation.",
                },
                {"role": "user", "content": prefix},
            ],
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            cache_sample_index=sample_index,
        )
        return _first_word(out)

    return generate


def simmia_score(
    text: str,
    client: TargetClient,
    non_member_prefix: str | None = None,
    n_samples: int | None = None,
    max_words: int = 20,
    sleep_between_calls: float = 0.0,
) -> float | None:
    """SimMIA(x) = mean_i [ -perturbed_word_score_i / (normal_word_score_i + EPS) ],
    ver docstring del modulo. Formula y parametros identicos a
    simmia_decop/notebooks/simMIA.ipynb (commit a63ec742).

    `non_member_prefix=None` (default) carga el prefijo de calibracion fijo via
    get_default_non_member_prefix() -- pasar "" explicito solo si de verdad se quiere
    desactivar la perturbacion (no deberia hacerse en uso normal, rompe la formula).
    `n_samples=None` (default) usa mia_common.settings.simia_n_samples (3, idem
    notebook).

    Por cada posicion i en las primeras `max_words` palabras del texto: compara la
    similitud promedio (embedding) entre la palabra real y N continuaciones generadas
    por el modelo bajo el prefijo normal vs. bajo el prefijo perturbado. Devuelve
    mean(relative_word_score); None si el texto no tiene ni 2 palabras (no hay ninguna
    posicion para evaluar)."""
    if non_member_prefix is None:
        non_member_prefix = get_default_non_member_prefix()
    if n_samples is None:
        n_samples = settings.simia_n_samples

    generate = model_generate_fn_factory(client)
    words = text.split()[:max_words]
    if len(words) < 2:
        return None

    relative_scores = []

    for i in range(1, len(words)):
        prefix = " ".join(words[:i])
        target_word = words[i]
        perturbed_prefix = (non_member_prefix + " " + prefix).strip()

        normal_samples = []
        perturbed_samples = []
        for s in range(n_samples):
            normal_samples.append(generate(prefix, sample_index=s))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)
            perturbed_samples.append(generate(perturbed_prefix, sample_index=s))
            if sleep_between_calls:
                time.sleep(sleep_between_calls)

        normal_word_score = _word_score(target_word, normal_samples)
        perturbed_word_score = _word_score(target_word, perturbed_samples)
        relative_scores.append(-perturbed_word_score / (normal_word_score + _EPS))

    return float(np.mean(relative_scores)) if relative_scores else None
