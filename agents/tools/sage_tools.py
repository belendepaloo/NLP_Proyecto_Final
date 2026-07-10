"""
sage_tools.py — adapter sobre SAGE.sage.SAGE. El QA check (sage_quality_check) es
deterministico (SAGE ya emite sps/wordsim), no necesita otra LLM call -- la decision
de QUE HACER cuando falla (reintentar/descartar/escalar) es trabajo del sage_qa_agent
de la Fase 2, no de esta funcion.

NOTA: SAGE.sage importa transitivamente sae_lens + transformer_lens (via SAGE.sps),
dependencias pesadas que pueden no estar instaladas en todos los entornos de
desarrollo. _get_sage() las carga de forma lazy y devuelve un error claro si faltan,
en vez de romper el import de este modulo entero.
"""

from __future__ import annotations

import threading

from mia_common.settings import settings  # dispara el env-bridge de HF_TOKEN/GOOGLE_CLOUD_PROJECT
# antes de que SAGE.sps intente bajar el modelo gated google/gemma-2b. Sin este
# import, llamar run_sage_tool() desde un script que no haya importado
# mia_common.settings por su cuenta falla con 401 aunque el .env tenga el token bien
# (se detecto exactamente asi al probar esto en aislado).

_sage_singleton = None
# SAGE carga modelos de PyTorch (T5 + Gemma-2B via transformer_lens) que no estan
# garantizados thread-safe para forward passes concurrentes sobre la misma instancia.
# Al paralelizar chunks (ThreadPoolExecutor), este lock serializa el uso de SAGE sin
# bloquear las llamadas a la API del target (que corren en threads distintos, sobre
# clientes distintos del pool).
_sage_lock = threading.Lock()


def _get_sage(
    device: str | None = None,
    min_length_ratio: float = 0.75,
    n_candidates_generated: int = 4,
    n_candidates_kept: int = 3,
):
    """Singleton lazy: instanciar SAGE() carga Gemma-2B + SAE (pesado), no se quiere
    pagar ese costo mas de una vez por proceso (ver webapp/main.py startup en Fase 4).
    n_candidates_generated/n_candidates_kept quedan fijos para la vida del singleton
    (no son algo que sage_qa_agent deba variar por llamada, ver el docstring de
    run_sage_tool)."""
    global _sage_singleton
    if _sage_singleton is None:
        try:
            from SAGE.sage import SAGE
        except ImportError as e:
            raise ImportError(
                "SAGE.sage no se pudo importar -- faltan transformer_lens/sae_lens "
                "(dependencias pesadas de SPS, ver requirements.txt). Instalalas con "
                "`pip install transformer_lens sae_lens` para poder usar este tool."
            ) from e
        _sage_singleton = SAGE(
            device=device,
            min_length_ratio=min_length_ratio,
            n_candidates_generated=n_candidates_generated,
            n_candidates_kept=n_candidates_kept,
        )
    return _sage_singleton


def get_sage_candidates(text: str, device: str | None = None, min_length_ratio: float = 0.75) -> list[str]:
    """Devuelve los candidatos de paráfrasis para `text`, usando cache cross-run.

    Si el texto ya fue procesado con los mismos parámetros (n_generated/n_kept de
    settings), devuelve la lista cacheada en runs/_sage_cache/ sin cargar Gemma-2B.
    Si hay cache miss, corre SAGE, guarda el resultado y lo devuelve."""
    from mia_common import sage_cache
    n_gen = settings.sage_n_candidates_generated
    n_kept = settings.sage_n_candidates_kept
    cached = sage_cache.get(text, n_gen, n_kept)
    if cached is not None:
        return cached
    sage_out = run_sage_tool(text, device=device, min_length_ratio=min_length_ratio)
    candidates = [c["text"] for seg in sage_out.get("segments", [])
                  for c in seg.get("all_candidates", [])]
    sage_cache.put(text, n_gen, n_kept, candidates)
    return candidates


def run_sage_tool(text: str, device: str | None = None, min_length_ratio: float = 0.75) -> dict:
    """Paraphrasea `text` con SAGE. Devuelve el dict tal cual lo emite SAGE().paraphrase()
    (original/paraphrase/segments, cada segmento con sps/wordsim/final_score/all_candidates).

    n_candidates_generated/n_candidates_kept (ver mia_common.settings,
    sage_n_candidates_generated=4/sage_n_candidates_kept=3 por defecto) NO se exponen
    como parametro de esta tool -- es una decision de calidad/costo del humano, no algo
    que sage_qa_agent deba variar por chunk (mismo criterio que min_length_ratio/min_sps
    en sage_quality_check, que tampoco varian entre llamadas)."""
    with _sage_lock:
        sage = _get_sage(
            device=device,
            min_length_ratio=min_length_ratio,
            n_candidates_generated=settings.sage_n_candidates_generated,
            n_candidates_kept=settings.sage_n_candidates_kept,
        )
        return sage.paraphrase(text)


def sage_quality_check(sage_segment: dict, min_sps: float = 0.7, min_length_ratio: float = 0.75) -> dict:
    """Chequeo determinista de calidad de un segmento de SAGE.paraphrase(...)["segments"].
    Un segmento estructural (titulos/citas/etc, no parafraseado) siempre pasa. Un
    segmento narrativo falla si sps < min_sps, o si el largo del seleccionado cayo por
    debajo de min_length_ratio del original (trunca/resume en vez de parafrasear)."""
    if sage_segment["type"] == "structural":
        return {"passed": True, "reasons": []}

    reasons = []
    sps = sage_segment.get("sps")
    if sps is None or sps < min_sps:
        reasons.append(f"sps {sps} < {min_sps}")

    original = sage_segment.get("original", "")
    selected = sage_segment.get("selected", "")
    ratio = len(selected) / max(len(original), 1)
    if ratio < min_length_ratio:
        reasons.append(f"length_ratio {ratio:.2f} < {min_length_ratio}")

    return {"passed": not reasons, "reasons": reasons}
