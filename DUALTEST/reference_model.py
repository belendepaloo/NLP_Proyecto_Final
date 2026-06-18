"""
reference_model.py

Implementa el componente de "modelo de referencia pequeno" de DUALTEST
(Practical Memorization Tests for Detecting Copyrighted Data in LLMs, ICLR 2026, anonimo).

Rol en el paper (Seccion 3.2):
    "we use a smaller, open-source reference model as a proxy for generalization.
    If the target's output is very close to the source continuation and the reference
    model assigns low probability to producing something that close, we treat that as
    evidence of memorization."

Este modulo necesita acceso WHITE-BOX al modelo de referencia (logits token a token),
porque tanto RLB como ESB requieren probabilidades teacher-forced. El paper es explicito
en que SOLO el modelo target tiene que ser black-box (Seccion 3.1, "Non-Privileged
Access"); el modelo de referencia no tiene esa restriccion -- lo corremos nosotros
localmente y necesitamos sus logits.

Eleccion de modelo de referencia (Apendice D del paper, "Effect of the Reference
Model"): el paper muestra que el recall cae cuanto mas grande/capaz es el modelo de
referencia, porque empieza a memorizar el tambien y deja de ser un buen proxy de
"generalizacion pura". Para una RTX 4090 (24GB VRAM), recomendamos una familia chica y
fluida en espanol para poder repetir el ablation del Apendice D variando el tamano:
    - Qwen2.5: 0.5B / 1.5B / 3B / 7B (misma familia/tokenizer, buen soporte multilingue)
    - Llama-3.2: 1B / 3B (mas Llama-3.1-8B como punto de comparacion "grande")
Preferir las versiones BASE (no -Instruct) si el target tambien genera en modo
"continuar texto crudo" en vez de modo chat, para que la distribucion de probabilidad
del modelo de referencia sea comparable al estilo de generacion que estamos evaluando.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


class ReferenceModel:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

    @torch.no_grad()
    def teacher_forced_token_probs(self, prefix_ids: List[int], continuation_ids: List[int]) -> List[float]:
        """
        Devuelve P_ref(continuation_ids[i] | prefix_ids + continuation_ids[:i]) para cada i.

        Este es el bloque base usado por las dos variantes:
          - RLB (Seccion 3.3): p_RLB(r) = prod_{i=1}^{r} P_ref(src_token_i | prefijo, src_tokens_{<i})
            ("probability of generating the same source tokens with this run-length
            (or longer) under a small reference model")
          - ESB (Seccion 3.4): p_ESB = prod_{i=1}^{L} P_ref(target_token_i | prefijo, target_tokens_{<i})
            ("probability of generating the same completion as the target LLM given
            the source prefix")

        La unica diferencia entre los dos usos es QUE secuencia de tokens se pasa como
        `continuation_ids` (los tokens fuente para RLB, los tokens que efectivamente
        genero el target para ESB).
        """
        if len(continuation_ids) == 0:
            return []
        full_ids = list(prefix_ids) + list(continuation_ids)
        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids).logits[0]  # (seq_len, vocab)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        n_prefix = len(prefix_ids)
        token_probs = []
        for i, tok in enumerate(continuation_ids):
            pos = n_prefix + i - 1  # logits en pos predicen el token en pos+1
            lp = log_probs[pos, tok].item()
            token_probs.append(float(torch.exp(torch.tensor(lp))))
        return token_probs

    def sequence_probability(self, prefix_ids: List[int], continuation_ids: List[int],
                              up_to: int = None) -> float:
        """
        Producto de las probabilidades token a token sobre los primeros `up_to` tokens
        (o toda la continuacion si up_to=None). Implementa literalmente p_RLB(r) o
        p_ESB como un escalar. Calculado en log-espacio internamente para evitar
        underflow en secuencias largas; el valor devuelto es la probabilidad lineal
        (coincide con el eje "Completion probability of a smaller model" de la Fig. 1
        derecha del paper).
        """
        probs = self.teacher_forced_token_probs(prefix_ids, continuation_ids)
        if up_to is not None:
            probs = probs[:up_to]
        if not probs:
            return 1.0
        log_p = sum(torch.log(torch.tensor(max(p, 1e-300))).item() for p in probs)
        return float(torch.exp(torch.tensor(log_p)))

    def sequence_log_probability(self, prefix_ids: List[int], continuation_ids: List[int],
                                  up_to: int = None) -> float:
        """Igual que arriba pero devuelve log-probabilidad (mas estable numericamente;
        usar esto para los umbrales en la practica, convirtiendo threshold_p a
        log-espacio una sola vez)."""
        probs = self.teacher_forced_token_probs(prefix_ids, continuation_ids)
        if up_to is not None:
            probs = probs[:up_to]
        if not probs:
            return 0.0
        return sum(torch.log(torch.tensor(max(p, 1e-300))).item() for p in probs)
