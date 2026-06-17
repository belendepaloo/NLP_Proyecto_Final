"""
text_pipeline.py — Limpieza y chunking de texto crudo para MIA (DE-COP / Min-K% / etc.)

Flujo:  raw  ->  clean_text()  ->  chunk_text()  ->  build_chunk_dataset() -> DataFrame

Principios (importantes para que el MIA sea válido):
  * Los chunks son substrings VERBATIM contiguos del texto limpio (nunca se reescriben).
  * El largo se mide con UN solo contador (model-agnostic) para que sea comparable entre modelos.
  * Se reporta n_tokens por chunk para poder igualar la distribución de largo member vs non-member.
  * Buckets alineados a los benchmarks: 64 / 128 / 256 (WikiMIA 32/64/128/256, BookTection 64/128/256).
"""

import re
import html
import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    import pysbd
    _SEG = pysbd.Segmenter(language="en", clean=False)
    def _split_sentences(text: str):
        return [s.strip() for s in _SEG.segment(text) if s.strip()]
except ImportError:
    # fallback regex: corta en . ! ? seguido de espacio + mayuscula/comilla
    _SENT_RE = re.compile(r'(?<=[.!?])["”\')\]]?\s+(?=[A-Z"“\'(\[])')
    def _split_sentences(text: str):
        return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ----------------------------------------------------------------------------- contadores de largo
def word_count(text: str) -> int:
    """Largo en palabras (whitespace). Offline, model-agnostic. Es el default."""
    return len(text.split())

def est_token_count(text: str) -> int:
    """Estimacion de tokens sin tokenizer (≈ palabras * 1.3 para ingles). Solo aproximada."""
    return round(len(text.split()) * 1.3)

def make_tiktoken_counter(encoding: str = "cl100k_base") -> Callable[[str], int]:
    """Contador real con tiktoken (necesita red la 1ra vez). Usalo como count_fn."""
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return lambda t: len(enc.encode(t))

def make_hf_counter(model_name: str, token: Optional[str] = None) -> Callable[[str], int]:
    """Contador real con el tokenizer de un modelo de HuggingFace (ideal en Colab).
       Ej: make_hf_counter('meta-llama/Llama-2-7b-hf')."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, token=token)
    return lambda t: len(tok.encode(t, add_special_tokens=False))


# ----------------------------------------------------------------------------- limpieza
_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_GUTENBERG_END   = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*", re.I | re.S)

def _looks_like_html(s: str) -> bool:
    s_low = s[:2000].lower()
    return ("<html" in s_low or "<div" in s_low or "<p>" in s_low or "<body" in s_low
            or s_low.count("<") > 10)

def clean_text(raw: str, is_html: Optional[bool] = None,
               strip_gutenberg: bool = True) -> str:
    """raw (HTML o texto plano) -> texto limpio, listo para chunkear.
       - HTML: trafilatura saca menus/ads/footer/sidebar (heuristico, sin LLM).
       - Project Gutenberg: saca el header/footer de licencia.
       - normaliza unicode, une guiones de fin de linea, colapsa espacios."""
    if raw is None:
        return ""
    if is_html is None:
        is_html = _looks_like_html(raw)

    if is_html and _HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            raw, include_comments=False, include_tables=False,
            favor_precision=True, no_fallback=False)
        text = extracted or ""
    else:
        text = raw

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    if strip_gutenberg:
        text = _GUTENBERG_START.split(text)[-1]
        text = _GUTENBERG_END.split(text)[0]

    # une palabras cortadas por guion al final de linea:  exam-\nple -> example
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # saltos de linea simples dentro de un parrafo -> espacio; dobles = separacion de parrafo
    text = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ----------------------------------------------------------------------------- chunking
BUCKETS = (64, 128, 256)

def _nearest_bucket(n: int, buckets=BUCKETS) -> int:
    return min(buckets, key=lambda b: abs(b - n))

@dataclass
class Chunk:
    text: str
    n_words: int
    n_tokens: int
    char_len: int
    length_bucket: int
    source: Optional[str] = None
    date: Optional[str] = None

def chunk_text(clean: str, target: int = 128,
               count_fn: Callable[[str], int] = est_token_count,
               min_len: int = 24, hard_truncate: bool = False,
               source: Optional[str] = None, date: Optional[str] = None):
    """Empaqueta oraciones contiguas en ventanas de ~target unidades (las que mida count_fn).
       Corta en frontera de oracion (chunk <= target). Cada chunk es verbatim.
       - min_len: descarta chunks demasiado cortos (en la unidad de count_fn).
       - hard_truncate: si una sola oracion ya supera target, la corta a target palabras.
       Devuelve list[Chunk]."""
    sentences = _split_sentences(clean)
    chunks, cur, cur_len = [], [], 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            txt = " ".join(cur).strip()
            n_tok = count_fn(txt)
            if n_tok >= min_len:
                chunks.append(Chunk(
                    text=txt, n_words=word_count(txt), n_tokens=n_tok,
                    char_len=len(txt), length_bucket=_nearest_bucket(n_tok),
                    source=source, date=date))
        cur, cur_len = [], 0

    for sent in sentences:
        s_len = count_fn(sent)
        if s_len > target and not cur and hard_truncate:
            # oracion sola mas larga que target -> recorte verbatim a 'target' palabras
            words = sent.split()
            approx_words = max(1, int(target / (s_len / max(1, len(words)))))
            cur = [" ".join(words[:approx_words])]; cur_len = count_fn(cur[0]); flush(); continue
        if cur_len + s_len > target and cur:
            flush()
        cur.append(sent); cur_len += s_len
    flush()
    return chunks


# ----------------------------------------------------------------------------- dataset + dedup
def _norm_key(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()

def build_chunk_dataset(items, target: int = 128,
                        count_fn: Callable[[str], int] = est_token_count,
                        is_html: Optional[bool] = None,
                        min_len: int = 24, hard_truncate: bool = False,
                        dedup: bool = True) -> pd.DataFrame:
    """items: lista de dicts {'raw':..., 'source':..., 'date':...} (source/date opcionales).
       Limpia, chunkea y arma un DataFrame; deduplica chunks identicos (exacto, normalizado).
       Columnas: text, source, date, n_words, n_tokens, char_len, length_bucket."""
    rows, seen = [], set()
    for it in items:
        clean = clean_text(it["raw"], is_html=is_html)
        for ch in chunk_text(clean, target=target, count_fn=count_fn,
                             min_len=min_len, hard_truncate=hard_truncate,
                             source=it.get("source"), date=it.get("date")):
            if dedup:
                k = _norm_key(ch.text)
                if k in seen:
                    continue
                seen.add(k)
            rows.append(ch.__dict__)
    cols = ["text", "source", "date", "n_words", "n_tokens", "char_len", "length_bucket"]
    return pd.DataFrame(rows, columns=cols)