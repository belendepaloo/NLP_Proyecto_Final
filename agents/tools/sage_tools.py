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

import mia_common.settings  # noqa: F401 -- dispara el env-bridge de HF_TOKEN/GOOGLE_CLOUD_PROJECT
# antes de que SAGE.sps intente bajar el modelo gated google/gemma-2b. Sin este
# import, llamar run_sage_tool() desde un script que no haya importado
# mia_common.settings por su cuenta falla con 401 aunque el .env tenga el token bien
# (se detecto exactamente asi al probar esto en aislado).

_sage_singleton = None


def _get_sage(device: str | None = None, min_length_ratio: float = 0.75):
    """Singleton lazy: instanciar SAGE() carga Gemma-2B + SAE (pesado), no se quiere
    pagar ese costo mas de una vez por proceso (ver webapp/main.py startup en Fase 4)."""
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
        _sage_singleton = SAGE(device=device, min_length_ratio=min_length_ratio)
    return _sage_singleton


def run_sage_tool(text: str, device: str | None = None, min_length_ratio: float = 0.75) -> dict:
    """Paraphrasea `text` con SAGE. Devuelve el dict tal cual lo emite SAGE().paraphrase()
    (original/paraphrase/segments, cada segmento con sps/wordsim/final_score/all_candidates)."""
    sage = _get_sage(device=device, min_length_ratio=min_length_ratio)
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
