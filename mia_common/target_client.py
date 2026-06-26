"""
target_client.py — cliente unico, pluggable, para el modelo "black box" target bajo
test de MIA. Generaliza DUALTEST/target_model.py::APITarget (que ya definia la idea
correcta: "ustedes traen su propio call_fn") agregando lo que DE-COP y SiMIA necesitan
y hoy duplican cada uno a su manera:

  - .chat(messages) para prompts de instruccion de un turno (DE-COP MCQ, SiMIA
    next-word) ademas de .complete(prompt) para continuacion cruda (DUALTEST).
  - reintentos con backoff + deteccion de cap diario, generalizando la logica
    `safe_ask` que hoy vive solo en el notebook de DE-COP (regex "try again in Xm Ys").
  - soporte multi-provider: groq (default, OpenAI-compatible), openai, anthropic,
    google, o hf_local (modelo HF corrido localmente, via DUALTEST.target_model).

DUALTEST.target_model.APITarget / HFLocalTarget NO se tocan (fidelidad al paper) --
`as_dualtest_target()` aca abajo es el adapter que los conecta con este cliente.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

from mia_common import cache


class DailyCapError(Exception):
    """El proveedor reporto un cap diario/mensual (no transitorio) -- frenar el run."""


@dataclass
class Completion:
    text: str
    token_ids: Optional[List[int]] = None  # solo disponible con backend hf_local


@runtime_checkable
class TargetClient(Protocol):
    has_tokenizer: bool

    def complete(self, prompt: str, max_new_tokens: int = 64, **kw) -> Completion: ...

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        **kw,
    ) -> str: ...


_RETRY_AFTER_MS = re.compile(r"try again in (\d+)m([\d.]+)s")
_RETRY_AFTER_S = re.compile(r"try again in ([\d.]+)s")


def _parse_retry_after(msg: str) -> float | None:
    """Misma logica que safe_ask() en DE-COP/DE_COP_BookTection.ipynb, generalizada."""
    if "429" not in msg and "rate_limit" not in msg.lower():
        return None
    m2 = _RETRY_AFTER_MS.search(msg)
    if m2:
        return int(m2.group(1)) * 60 + float(m2.group(2))
    m1 = _RETRY_AFTER_S.search(msg)
    if m1:
        return float(m1.group(1))
    return 20.0


def _is_daily_cap(msg: str) -> bool:
    return "per day" in msg.lower() or "TPD" in msg or "RPD" in msg


class RateLimitedAPITarget:
    """Cliente generico para un target API (Groq/OpenAI/Anthropic/Google), con
    throttling + retry/backoff compartido por DE-COP, SiMIA y (via adapter) DUALTEST.

    Thread-safe via un lock por instancia: cuando se paraleliza el pipeline (varios
    chunks en simultaneo repartidos sobre un TargetClientPool), mas de un chunk puede
    terminar compartiendo la MISMA key/cliente -- el lock serializa esas llamadas para
    que el throttle (_last_call_ts) no se pise entre threads, sin bloquear llamadas que
    caen en OTRO cliente del pool (otra key), que siguen corriendo en paralelo."""

    has_tokenizer = False

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str | None = None,
        max_new_tokens: int = 64,
        max_retries: int = 6,
        min_seconds_between_calls: float = 0.0,
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.min_seconds_between_calls = min_seconds_between_calls
        self._last_call_ts = 0.0
        self._lock = threading.Lock()
        self._client = self._build_client(provider, api_key)

    def _build_client(self, provider: str, api_key: str | None) -> Any:
        if provider == "groq":
            from groq import Groq

            return Groq(api_key=api_key)
        if provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=api_key)
        if provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=api_key)
        if provider == "google":
            from google import genai

            return genai.Client(api_key=api_key)
        raise ValueError(f"Unknown target provider: {provider!r}")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - elapsed)

    def _raw_chat_call(
        self, messages: list[dict], max_tokens: int, temperature: float, **kw
    ) -> str:
        if self.provider in ("groq", "openai"):
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kw,
            )
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            turns = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.model_name,
                system=system,
                messages=turns,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.content[0].text if resp.content else ""
        if self.provider == "google":
            prompt = "\n".join(m["content"] for m in messages)
            resp = self._client.models.generate_content(model=self.model_name, contents=prompt)
            return resp.text or ""
        raise ValueError(f"Unknown target provider: {self.provider!r}")

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        cache_sample_index: int | None = None,
        **kw,
    ) -> str:
        """Cachea toda llamada (request+response) en mia_common.cache antes de pegarle
        a la API real. `cache_sample_index` es obligatorio en la practica para
        cualquier caller que haga sampling (temperature>0, ej. SiMIA pidiendo N
        muestras del mismo prompt) -- sin el, la 2da..Nesima muestra pegarian todas
        contra el mismo cache hit de la 1ra, perdiendo la independencia de las muestras."""
        payload = {"messages": messages, "max_new_tokens": max_new_tokens, "temperature": temperature, **kw}
        cached = cache.get(self.provider, self.model_name, payload, cache_sample_index)
        if cached is not None:
            return cached

        with self._lock:  # serializa llamadas reales+throttle de ESTE cliente entre threads
            cached = cache.get(self.provider, self.model_name, payload, cache_sample_index)
            if cached is not None:
                return cached

            last_err: Exception | None = None
            for _ in range(self.max_retries):
                self._throttle()
                try:
                    self._last_call_ts = time.time()
                    text = self._raw_chat_call(messages, max_new_tokens, temperature, **kw)
                    cache.put(self.provider, self.model_name, payload, text, cache_sample_index)
                    return text
                except Exception as e:  # noqa: BLE001 -- proveedores tiran excepciones distintas
                    msg = str(e)
                    if _is_daily_cap(msg):
                        raise DailyCapError(msg) from e
                    wait = _parse_retry_after(msg)
                    if wait is None:
                        raise
                    last_err = e
                    time.sleep(wait + 1)
            raise RuntimeError("max_retries agotado en 429 transitorio") from last_err

    # DUALTEST necesita continuacion cruda de texto (RLB/ESB comparan la continuacion
    # del target contra la continuacion fuente token a token). Un modelo chat-tuned
    # sin instruccion explicita tiende a responder conversacionalmente en vez de
    # continuar el texto ("Parece que esto es de una novela...") -- esta instruccion
    # es necesaria para que el black-box test tenga sentido contra APIs instruct,
    # no solo contra modelos base (que es lo que usa el paper original).
    _CONTINUATION_SYSTEM_PROMPT = (
        "Continue the following text exactly as it would naturally continue in its "
        "original source. Output ONLY the continuation -- no commentary, no preamble, "
        "no explanation, no quotation marks."
    )

    def complete(self, prompt: str, max_new_tokens: int | None = None, **kw) -> Completion:
        text = self.chat(
            [
                {"role": "system", "content": self._CONTINUATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            **kw,
        )
        return Completion(text=text, token_ids=None)


def make_target_client(provider: str, model_name: str, **kw) -> TargetClient:
    """Factory unica: (provider, model_name) -> cliente listo. Punto de entrada que
    usan agents/tools/mia_tools.py, DE-COP/decop.py y SiMIA/simia.py."""
    if provider == "hf_local":
        from DUALTEST.target_model import HFLocalTarget

        return HFLocalTarget(model_name=model_name, device=kw.get("device"))
    return RateLimitedAPITarget(provider=provider, model_name=model_name, **kw)


def resolve_target_client(provider: str, model_name: str) -> TargetClient:
    """make_target_client() pero resolviendo la api_key correspondiente desde
    mia_common.settings -- usado por agents/orchestrator.py para construir el target
    elegido en la webapp (Fase 4) sin que cada caller tenga que repetir el mapeo
    provider->api_key. Para "groq" con varias keys en GROQ_API_KEYS usa la primera --
    el round-robin de TargetClientPool esta pensado para paralelizar chunks con
    ThreadPoolExecutor (ver scripts/run_pipeline_manual.py), no para un agente
    razonando secuencialmente sobre un chunk a la vez."""
    from mia_common.settings import settings

    if provider == "hf_local":
        return make_target_client(provider, model_name)

    api_key_by_provider = {
        "groq": next(iter(settings.groq_api_keys()), None),
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "google": settings.google_api_key,
    }
    if provider not in api_key_by_provider:
        raise ValueError(f"Unknown target provider: {provider!r}")
    api_key = api_key_by_provider[provider]
    if not api_key:
        raise ValueError(
            f"No hay API key configurada para target_provider={provider!r} (ver .env.example)"
        )
    return make_target_client(provider, model_name, api_key=api_key)


class TargetClientPool:
    """Pool de clientes (uno por API key) para paralelizar llamadas reales sin
    pisarse el rate limit de una sola key -- cada key mantiene su propio throttle
    independiente, asi que N keys ~ N x el throughput de una sola. `.get()` reparte
    clientes round-robin (thread-safe); pensado para asignar UN cliente por chunk/tarea
    en un pool de workers, no por llamada individual, asi el throttle de cada key sigue
    siendo coherente con la secuencia de llamadas de ese chunk."""

    def __init__(self, clients: list[TargetClient]):
        if not clients:
            raise ValueError("TargetClientPool necesita al menos un cliente")
        self._clients = clients
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._clients)

    def get(self) -> TargetClient:
        with self._lock:
            idx = next(self._counter) % len(self._clients)
        return self._clients[idx]


def make_target_client_pool(provider: str, model_name: str, api_keys: list[str], **kw) -> TargetClientPool:
    """Un RateLimitedAPITarget por key en `api_keys`. Ver TargetClientPool."""
    clients = [
        RateLimitedAPITarget(provider=provider, model_name=model_name, api_key=key, **kw) for key in api_keys
    ]
    return TargetClientPool(clients)


def as_dualtest_target(client: TargetClient, max_new_tokens: int = 64):
    """Adapter: envuelve un TargetClient como DUALTEST.target_model.APITarget, sin
    tocar el modulo DUALTEST original (fidelidad al paper)."""
    from DUALTEST.target_model import APITarget

    def call_fn(prefix_text: str, max_new_tokens: int = 64, **kw) -> str:
        # DUALTEST.scoring.score_texts siempre pasa do_sample=False (parametro de
        # generate() de HuggingFace) -- no existe equivalente en una API de chat
        # completions (Groq/OpenAI/etc usan temperature), se descarta aca.
        kw.pop("do_sample", None)
        return client.complete(prefix_text, max_new_tokens=max_new_tokens, **kw).text

    return APITarget(call_fn=call_fn, max_new_tokens=max_new_tokens)
