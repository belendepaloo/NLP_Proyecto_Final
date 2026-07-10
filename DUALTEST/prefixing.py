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
