"""
metrics.py

Funciones de scoring centrales de DUALTEST. Implementa:
    - Run-Length-Based (RLB)               -- Seccion 3.3
    - Edit-Similarity-Based (ESB)           -- Seccion 3.4
    - ESB + largo comprimido con Zlib       -- Seccion 4.2.3 / Tabla 7 (el paper NO da
      una formula exacta para esta; ver el docstring de `esb_zlib_score` para nuestra
      reconstruccion, claramente marcada como tal).

Las tres producen SCORES escalares que despues calibration.py convierte en decisiones
binarias via umbrales. Ninguna de estas funciones decide "memorizado o no" por si
misma -- los umbrales son especificos por dataset/dominio y deben calibrarse siguiendo
el protocolo de dos etapas de la Seccion 4.1 (100% precision en el setting normal, luego
0% FPR en los Generalization Sets A/B adversariales).
"""

import zlib
from typing import List, Tuple


def run_length(target_tokens: List[int], source_tokens: List[int]) -> int:
    """
    Seccion 3.3: "The run length is the number of consecutive completion tokens that
    exactly match the source continuation, stopping at the first mismatch."
    """
    r = 0
    for t, s in zip(target_tokens, source_tokens):
        if t != s:
            break
        r += 1
    return r


def _levenshtein(a: str, b: str) -> int:
    """DP clasico O(len(a)*len(b)); suficiente para strings de hasta ~64 tokens."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def edit_similarity(a: str, b: str) -> float:
    """
    Seccion 3.4, basado en Ippolito et al. (2022). NOTA SOBRE LA CONVENCION: Ippolito
    et al. definen literalmente

        EditSim(x,y) = EditDistance(x,y) / max(|x|,|y|)

    que en realidad es una DISTANCIA normalizada (0 = identicos, 1 = completamente
    distintos). La regla de decision de DUALTEST necesita la direccion opuesta ("alta
    similitud ... combinada con baja probabilidad ... evidencia de memorizacion"), asi
    que aca devolvemos el complemento:

        similitud = 1 - EditDistance(x,y) / max(|x|,|y|)

    es decir 1.0 = strings identicos, 0.0 = maximamente distintos. Esto es una
    correccion de notacion, no un cambio metodologico -- lo marcamos explicitamente
    porque si se copia literal de Ippolito sin invertirla, la regla de decision queda
    al reves.
    """
    if len(a) == 0 and len(b) == 0:
        return 1.0
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def rlb_score(target_tokens: List[int], source_tokens: List[int], reference_model,
              prefix_token_ids: List[int]) -> Tuple[int, float]:
    """
    Devuelve (run_length r, p_RLB) donde
        p_RLB = P_ref(el modelo de referencia reproduce la continuacion fuente
                       para al menos r tokens)
              = prod_{i=1}^{r} P_ref(source_tokens[i] | prefijo, source_tokens[:i])

    Senal de memorizacion: r alto Y p_RLB bajo.
    """
    r = run_length(target_tokens, source_tokens)
    if r == 0:
        return 0, 1.0  # sin match alguno -> nada que explicar, p es trivial/irrelevante
    p = reference_model.sequence_probability(prefix_token_ids, source_tokens, up_to=r)
    return r, p


def esb_score(target_text: str, target_tokens: List[int], source_text: str,
              reference_model, prefix_token_ids: List[int]) -> Tuple[float, float]:
    """
    Devuelve (s, p_ESB) donde
        s     = edit_similarity(target_text, source_text)   [continuacion real, max 64 tok]
        p_ESB = P_ref(el modelo de referencia genera independientemente esta misma
                       completion que produjo el TARGET | prefijo)
              = prod_{i=1}^{L} P_ref(target_tokens[i] | prefijo, target_tokens[:i])

    Senal de memorizacion: s alto Y p_ESB bajo.
    """
    s = edit_similarity(target_text, source_text)
    p = reference_model.sequence_probability(prefix_token_ids, target_tokens)
    return s, p


def esb_zlib_score(target_text: str, target_tokens: List[int], source_text: str,
                    reference_model, prefix_token_ids: List[int]) -> Tuple[float, float]:
    """
    Seccion 4.2.3 / Tabla 7: "an upgraded method that combines edit similarity with the
    length of Zlib-compressed text. This modification improves robustness against
    repetitive non-members that would yield a too conservative similarity threshold."

    *** El paper NO publica la formula exacta de combinacion. *** Esta es nuestra
    reconstruccion, basada en la misma logica del baseline de Kaneko et al. (2024) con
    Zlib (que divide un score derivado de probabilidad por el largo comprimido con Zlib
    para penalizar completions muy compresibles/repetitivas): reemplazamos el p_ESB
    crudo por

        p_ESB_corregido = p_ESB / len(zlib.compress(target_text))

    es decir, exigimos ADEMAS que la completion no sea trivialmente compresible (un
    proxy de "esto no es solo un patron repetitivo facil") antes de tratar un p_ESB
    bajo como evidencia de memorizacion. Tratar esta funcion como una reconstruccion
    razonada y claramente marcada, no como una transcripcion literal del paper.
    """
    s = edit_similarity(target_text, source_text)
    p = reference_model.sequence_probability(prefix_token_ids, target_tokens)
    compressed_len = len(zlib.compress(target_text.encode("utf-8")))
    corrected_p = p / max(compressed_len, 1)
    return s, corrected_p
