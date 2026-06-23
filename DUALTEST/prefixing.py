"""
prefixing.py

Parte un texto fuente en pares (prefijo, continuacion), siguiendo Secciones 3.3/3.4:
    "From each source document, we take a 64-token prefix and ask the target model for
    up to 64 completion tokens."

Dos modos:
    - por tokens (default, fiel a los experimentos principales del paper con
      Pythia/LLaMA-2): usa un tokenizer para cortar exactamente 64 tokens de prefijo y
      reservar hasta 64 tokens de continuacion real (ground truth).
    - por palabras (fallback que el propio paper usa en el Apendice de GPT-4,
      "Standard" = prefijo de 50 palabras, porque las APIs cerradas no exponen
      tokenizer).

OJO con un desajuste importante: si el tokenizer del modelo de referencia es distinto
al del target (ej. target = API cerrada, referencia = Qwen2.5), lo que tiene que
mantenerse constante es EL TEXTO en lenguaje natural del prefijo, no el conteo exacto
de tokens bajo cada tokenizer distinto -- re-tokenizar el mismo string con cada modelo
por separado. Un prefijo de exactamente 64 tokens solo tiene sentido pleno cuando un
unico tokenizer gobierna tanto la construccion del prefijo como el conteo de run-length
(es decir, target open-weight de la misma familia/tokenizer que la referencia). Si el
target es una API cerrada, acepten la pequena discrepancia de longitud entre tokenizers,
tal como efectivamente hace el paper en su propio apendice con GPT-4.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PrefixSplit:
    prefix_text: str
    continuation_text: str
    prefix_token_ids: Optional[List[int]] = None
    continuation_token_ids: Optional[List[int]] = None


def split_by_tokens(text: str, tokenizer, prefix_len: int = 64, continuation_len: int = 64) -> PrefixSplit:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    prefix_ids = ids[:prefix_len]
    continuation_ids = ids[prefix_len: prefix_len + continuation_len]
    return PrefixSplit(
        prefix_text=tokenizer.decode(prefix_ids),
        continuation_text=tokenizer.decode(continuation_ids),
        prefix_token_ids=prefix_ids,
        continuation_token_ids=continuation_ids,
    )


def split_by_words(text: str, prefix_len: int = 50, continuation_len: int = 64) -> PrefixSplit:
    """Fallback usado en el paper (Apendice B) cuando no hay tokenizer del target
    disponible. 50 palabras es el numero que ellos mismos usan ("Standard")."""
    words = text.split()
    prefix_words = words[:prefix_len]
    continuation_words = words[prefix_len: prefix_len + continuation_len]
    return PrefixSplit(
        prefix_text=" ".join(prefix_words),
        continuation_text=" ".join(continuation_words),
    )
