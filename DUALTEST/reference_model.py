import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


class ReferenceModel:
    def __init__(self, model_name: str, device: None, dtype=torch.bfloat16):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

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
          - RLB: p_RLB(r) = prod_{i=1}^{r} P_ref(src_token_i | prefijo, src_tokens_{<i})
            ("probability of generating the same source tokens with this run-length
            (or longer) under a small reference model")
          - ESB: p_ESB = prod_{i=1}^{L} P_ref(target_token_i | prefijo, target_tokens_{<i})
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
            pos = n_prefix + i - 1
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
        (coincide con el eje "Completion probability of a smaller model")
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
