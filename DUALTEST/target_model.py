import torch

from dataclasses import dataclass
from typing import Optional, List, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Completion:
    text: str
    token_ids: Optional[List[int]] = None  


class HFLocalTarget:
    """Target corrido localmente via HuggingFace transformers."""

    def __init__(self, model_name: str, device: None, dtype=torch.bfloat16):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.has_tokenizer = True

    @torch.no_grad()
    def complete(self, prefix_text: str, max_new_tokens: int = 64,
                 do_sample: bool = False, temperature: float = 1.0) -> Completion:
        """
        Por defecto, decodificacion greedy (do_sample=False), que es el modo principal
        usado en el paper. Para replicar tambien la variante "Temperature" del Apendice
        B (sampling con temperatura=1, usada solo en los experimentos de GPT-4), pasar
        do_sample=True, temperature=1.0.
        """
        ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                           pad_token_id=self.tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        out = self.model.generate(ids, **gen_kwargs)
        new_ids = out[0, ids.shape[1]:].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Completion(text=text, token_ids=new_ids)


class APITarget:
    """
    Wrapper generico para un target via API cerrada. `call_fn` es SU funcion que toma
    un string de prompt y devuelve un string de completion -- conecten aca su cliente de
    OpenAI/Anthropic/lo que usen para el rol de "target". No se asume acceso a tokenizer,
    por lo que el prefijo deberia construirse con prefixing.split_by_words en vez de
    split_by_tokens (ver prefixing.py).
    """

    def __init__(self, call_fn: Callable[..., str], max_new_tokens: int = 64):
        self.call_fn = call_fn
        self.max_new_tokens = max_new_tokens
        self.has_tokenizer = False

    def complete(self, prefix_text: str, **kwargs) -> Completion:
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        text = self.call_fn(prefix_text, max_new_tokens=max_new_tokens, **kwargs)
        return Completion(text=text, token_ids=None)
