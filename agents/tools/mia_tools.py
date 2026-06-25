"""
mia_tools.py — adapters delgados sobre los tres metodos de scoring MIA (DE-COP, SiMIA,
DUALTEST), todos consumiendo el mismo mia_common.target_client.TargetClient. En la
Fase 2 estas tres funciones se agrupan en un solo subagent (`mia_agent`, ver el plan:
"un solo subagent con 3 tools sobre el mismo chunk" en vez de un subagent por metodo).

Ninguna de las tres lanza una excepcion que tumbe el pipeline completo por un chunk
problematico -- devuelven un resultado con `abstained=True`/`skipped` cuando no se
puede calcular, para que agents/ensemble/combine.py decida que hacer con eso.
"""

from __future__ import annotations

import threading

from mia_common.target_client import TargetClient

_reference_model_singleton = None
# El reference model de DUALTEST es un modelo de PyTorch local (singleton) -- no
# garantizado thread-safe para forward passes concurrentes. Al paralelizar chunks,
# este lock serializa el uso del reference model sin bloquear las llamadas a la API
# del target (que tienen su propio lock por cliente en mia_common.target_client).
_dualtest_lock = threading.Lock()


def get_reference_model(model_name: str):
    """Singleton lazy: el reference model de DUALTEST es siempre local/white-box
    (nunca el target), y no tiene sentido recargarlo en cada llamada."""
    global _reference_model_singleton
    if _reference_model_singleton is None:
        from DUALTEST.reference_model import ReferenceModel

        _reference_model_singleton = ReferenceModel(model_name, device=None)
    return _reference_model_singleton


def run_decop_tool(
    verbatim_passage: str,
    paraphrase_candidates: list[str],
    book_title: str,
    author: str,
    client: TargetClient,
    n_permutations: int = 6,
) -> dict:
    """Devuelve {"method": "decop", "skipped": bool, "result": dict | None, "reason": str | None}.
    Skippea (no rompe el run) si SAGE no produjo >=3 candidatos validos para este chunk."""
    from DE_COP.decop import decop_score

    if len(paraphrase_candidates) < 3:
        return {
            "method": "decop",
            "skipped": True,
            "result": None,
            "reason": f"solo {len(paraphrase_candidates)} paraphrase candidates (necesita >=3)",
        }
    result = decop_score(
        verbatim_passage=verbatim_passage,
        paraphrase_candidates=paraphrase_candidates,
        book_title=book_title,
        author=author,
        client=client,
        n_permutations=n_permutations,
    )
    return {"method": "decop", "skipped": False, "result": result, "reason": None}


def run_simia_tool(
    text: str,
    client: TargetClient,
    non_member_prefix: str | None = None,
    n_samples: int | None = None,
    max_words: int = 20,
) -> dict:
    """Devuelve {"method": "simia", "skipped": bool, "result": float | None, "reason": str | None}.
    non_member_prefix=None / n_samples=None usan los defaults de simmia_score (prefijo
    de calibracion fijo + mia_common.settings.simia_n_samples) -- no hardcodear "" ni 1
    aca, ver el docstring de SiMIA/simia.py sobre por que eso rompia la formula."""
    from SiMIA.simia import simmia_score

    score = simmia_score(
        text=text,
        client=client,
        non_member_prefix=non_member_prefix,
        n_samples=n_samples,
        max_words=max_words,
    )
    if score is None:
        return {
            "method": "simia",
            "skipped": True,
            "result": None,
            "reason": "no se pudo calcular ningun ratio (texto muy corto o sin senal)",
        }
    return {"method": "simia", "skipped": False, "result": score, "reason": None}


def run_dualtest_tool(
    text: str,
    client: TargetClient,
    reference_model_name: str,
    prefix_len: int = 64,
    continuation_len: int = 64,
    max_new_tokens: int = 64,
    label: int = 0,
) -> dict:
    """Devuelve {"method": "dualtest", "skipped": bool, "result": dict | None, "reason": str | None}.
    `result`, si no es None, tiene run_length/p_rlb/edit_similarity/p_esb (ver
    DUALTEST.scoring.score_texts)."""
    from mia_common.target_client import as_dualtest_target
    from DUALTEST.scoring import score_texts

    reference = get_reference_model(reference_model_name)
    dualtest_target = as_dualtest_target(client, max_new_tokens=max_new_tokens)
    with _dualtest_lock:
        df = score_texts(
            texts=[text],
            target=dualtest_target,
            reference=reference,
            prefix_len=prefix_len,
            continuation_len=continuation_len,
            max_new_tokens=max_new_tokens,
            label=label,
            dataset_name="pipeline",
        )
    if df.empty:
        return {"method": "dualtest", "skipped": True, "result": None, "reason": "texto demasiado corto"}
    row = df.iloc[0].to_dict()
    return {"method": "dualtest", "skipped": False, "result": row, "reason": None}
