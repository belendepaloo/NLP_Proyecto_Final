"""
calibration.py

Implementa el protocolo de evaluacion de la Seccion 4.1:

  1. Setting NORMAL: calibrar umbral(es) para lograr 100% de precision (cero falsos
     positivos) en un set limpio de miembros/no-miembros, maximizando recall.
  2. Setting ADVERSARIAL: tomar ese mismo umbral y medir el False Positive Rate sobre
     los Generalization Sets (A: simple, B: dificil). Si FPR > 0, hay que endurecer el
     umbral (perdiendo recall) hasta llegar a FPR = 0% tambien ahi.

  El numero que finalmente se reporta (igual que en las Tablas 4, 5, 7, 12 del paper) es
  el recall que sobrevive exigiendo AMBAS condiciones simultaneamente.

Este modulo es generico: sirve para RLB (1 score a umbralar: p, "mas bajo = mas
sospechoso") y para ESB (2 scores: s "mas alto = mas sospechoso", p "mas bajo = mas
sospechoso").
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CalibrationResult:
    threshold: float
    recall_normal: float
    fpr_adversarial: float
    threshold_2: float = None  # usado para el segundo eje de ESB (similitud)


def _is_flagged(score, threshold, lower_is_suspicious):
    return score <= threshold if lower_is_suspicious else score >= threshold


def calibrate_1d(member_scores: List[float], nonmember_scores: List[float],
                  adversarial_scores: List[float],
                  lower_is_suspicious: bool = True) -> CalibrationResult:
    """
    Calibracion para RLB: un solo score = p_RLB (mas bajo = mas sospechoso).

    Paso 1: entre los umbrales que dan 0 falsos positivos en `nonmember_scores`
            (= 100% precision), elegir el que maximiza recall en `member_scores`.
    Paso 2: revisar FPR en `adversarial_scores`; si >0, seguir endureciendo el umbral
            hasta FPR=0, re-midiendo recall en cada paso.
    """
    candidate_thresholds = sorted(set(member_scores) | set(nonmember_scores))
    candidate_thresholds.sort(reverse=not lower_is_suspicious)

    best = CalibrationResult(threshold=float("-inf") if lower_is_suspicious else float("inf"),
                              recall_normal=0.0, fpr_adversarial=1.0)

    for t in candidate_thresholds:
        flagged_nonmember = [s for s in nonmember_scores if _is_flagged(s, t, lower_is_suspicious)]
        if len(flagged_nonmember) > 0:
            continue  # no cumple 100% precision en setting normal

        flagged_member = [s for s in member_scores if _is_flagged(s, t, lower_is_suspicious)]
        recall = len(flagged_member) / max(len(member_scores), 1)

        flagged_adv = [s for s in adversarial_scores if _is_flagged(s, t, lower_is_suspicious)]
        fpr = len(flagged_adv) / max(len(adversarial_scores), 1)

        if fpr == 0.0 and recall >= best.recall_normal:
            best = CalibrationResult(threshold=t, recall_normal=recall, fpr_adversarial=fpr)

    return best


def calibrate_2d(member: List[Tuple[float, float]], nonmember: List[Tuple[float, float]],
                  adversarial: List[Tuple[float, float]]) -> CalibrationResult:
    """
    Calibracion para ESB: cada item es (s, p) donde s alto + p bajo = sospechoso.
    Busqueda en grilla sobre ambos umbrales (s_grid x p_grid), misma logica de dos
    etapas que la version 1D.

    Nota de performance: la busqueda es O(|s_values| * |p_values| * n_muestras). Para
    datasets grandes conviene discretizar los valores candidatos (percentiles) en vez de
    usar todos los valores observados.
    """
    s_values = sorted(set(x[0] for x in member) | set(x[0] for x in nonmember))
    p_values = sorted(set(x[1] for x in member) | set(x[1] for x in nonmember))

    best = CalibrationResult(threshold=float("inf"), threshold_2=float("-inf"),
                              recall_normal=0.0, fpr_adversarial=1.0)

    def flagged(items, s_thr, p_thr):
        return [(s, p) for (s, p) in items if s >= s_thr and p <= p_thr]

    for s_thr in s_values:
        for p_thr in p_values:
            if len(flagged(nonmember, s_thr, p_thr)) > 0:
                continue  # no cumple 100% precision en setting normal
            recall = len(flagged(member, s_thr, p_thr)) / max(len(member), 1)
            fpr = len(flagged(adversarial, s_thr, p_thr)) / max(len(adversarial), 1)
            if fpr == 0.0 and recall >= best.recall_normal:
                best = CalibrationResult(threshold=p_thr, threshold_2=s_thr,
                                          recall_normal=recall, fpr_adversarial=fpr)
    return best
