"""
spend_guard.py — limite duro de gasto en USD para APIs facturadas por uso (Vertex AI,
de momento). Trackeado en disco (no en memoria) porque cada invocacion de un script es
un proceso nuevo -- mismo principio que DailyCapError en mia_common/target_client.py
para Groq: frenar ANTES de pasarse del limite, no enterarse despues por la factura.

Uso tipico: no se llama a mano. agents/orchestrator.py le pasa VertexSpendGuardCallback
al chat model de agent_model cuando es un modelo de Vertex AI.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from mia_common.settings import settings

_LOCK = threading.Lock()

# Precios publicados de Vertex AI (USD por 1M tokens, contexto <=200k). El output se
# redondea para arriba a proposito: Gemini 2.5 cobra "thinking tokens" como output y
# son faciles de subestimar -- mejor frenar antes de lo necesario que pasarse del limite
# real por una estimacion optimista. Si el modelo no esta en la tabla, se usa el peor
# caso conocido (_DEFAULT_PRICING), no se asume que es gratis.
GEMINI_VERTEX_PRICING_USD_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 12.0),  # publicado: input 1.25, output 10.0
    "gemini-2.5-flash": (0.30, 3.0),  # publicado: input 0.30, output 2.5
}
_DEFAULT_PRICING = (2.0, 12.0)


class SpendCapExceededError(Exception):
    """Se alcanzaria el limite de gasto configurado con esta llamada -- no se ejecuta."""


def _tracker_path(account: str) -> Path:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    return settings.runs_dir / f"_spend_{account}.json"


def get_spent_usd(account: str) -> float:
    path = _tracker_path(account)
    if not path.exists():
        return 0.0
    return json.loads(path.read_text()).get("spent_usd", 0.0)


def reserve(account: str, max_usd: float, estimated_usd: float) -> None:
    """Llamar ANTES de una llamada facturada. Si gastado+estimado supera max_usd,
    levanta SpendCapExceededError SIN reservar nada -- la llamada nunca se hace."""
    with _LOCK:
        spent = get_spent_usd(account)
        if spent + estimated_usd > max_usd:
            raise SpendCapExceededError(
                f"Limite de gasto de ${max_usd:.2f} para '{account}' alcanzado "
                f"(gastado ${spent:.4f}, esta llamada estimada en ${estimated_usd:.4f}). "
                f"Ver runs/_spend_{account}.json. Subir el limite es una decision "
                f"humana (editar mia_common.settings.agent_model_spend_cap_usd), no "
                f"algo que el agente deba resolver solo."
            )
        _write_spent(account, spent + estimated_usd)


def adjust_to_actual(account: str, estimated_usd: float, actual_usd: float) -> None:
    """Corrige la reserva de reserve() con el costo real (segun usage_metadata de la
    respuesta, o 0 si la llamada fallo) -- se llama despues de que la llamada termino."""
    with _LOCK:
        spent = get_spent_usd(account)
        _write_spent(account, max(0.0, spent - estimated_usd + actual_usd))


def _write_spent(account: str, spent_usd: float) -> None:
    _tracker_path(account).write_text(json.dumps({"spent_usd": spent_usd}, indent=2))


class VertexSpendGuardCallback(BaseCallbackHandler):
    """Frena (SpendCapExceededError) ANTES de cada llamada a un modelo Gemini de Vertex
    AI si el costo estimado de esta llamada haria superar `max_usd` acumulado para
    `account`. `raise_error = True` es necesario -- sin eso, langchain loguea la
    excepcion de un callback y sigue como si nada, la llamada se haria igual.

    Estimacion ANTES de la llamada (todavia no se sabe cuanto va a generar el modelo):
    caracteres de los mensajes de entrada / 4 como proxy de tokens de input, y se asume
    el peor caso de output (`assumed_max_output_tokens`). DESPUES de la llamada se
    ajusta al costo real con `AIMessage.usage_metadata` (input_tokens/output_tokens)."""

    raise_error = True

    def __init__(
        self,
        account: str,
        model_name: str,
        max_usd: float,
        assumed_max_output_tokens: int = 32_768,
    ) -> None:
        # Default alto a proposito: una pregunta TRIVIAL ("cual es la capital de
        # Argentina?") ya midio 6749 tokens de "reasoning" reales en Gemini 2.5 Pro --
        # una tarea compleja de un subagent (curacion, scoring) puede usar bastantes
        # mas. Esto es la cota PRE-llamada (antes de saber el real); mejor sobrar
        # margen aca, total se corrige con el costo real despues (on_llm_end).
        self.account = account
        self.max_usd = max_usd
        self.input_price, self.output_price = GEMINI_VERTEX_PRICING_USD_PER_1M_TOKENS.get(
            model_name, _DEFAULT_PRICING
        )
        self.assumed_max_output_tokens = assumed_max_output_tokens
        self._pending_estimates: dict[UUID, float] = {}

    def _estimate_usd(self, n_input_chars: int) -> float:
        estimated_input_tokens = n_input_chars / 4  # proxy grueso, ver docstring de la clase
        return (
            estimated_input_tokens / 1_000_000 * self.input_price
            + self.assumed_max_output_tokens / 1_000_000 * self.output_price
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        n_chars = sum(len(str(m.content)) for batch in messages for m in batch)
        estimated_usd = self._estimate_usd(n_chars)
        reserve(self.account, self.max_usd, estimated_usd)
        self._pending_estimates[run_id] = estimated_usd

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        estimated_usd = self._pending_estimates.pop(run_id, 0.0)
        actual_usd = estimated_usd  # fallback si la respuesta no trae usage_metadata
        for gen_list in response.generations:
            for gen in gen_list:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    input_tokens = usage.get("input_tokens", 0)
                    # OJO: usage_metadata["output_tokens"] de Gemini 2.5 (thinking) NO
                    # incluye los "reasoning tokens" (usage["output_token_details"]
                    # ["reasoning"]) -- Google factura esos como output igual.
                    # Confirmado en vivo: una pregunta trivial uso output_tokens=3 pero
                    # reasoning=6749 (total_tokens=6766) -- contar solo output_tokens
                    # subestima el costo real en ordenes de magnitud. total_tokens -
                    # input_tokens captura output + reasoning sin depender de como cada
                    # provider nombre el sub-campo.
                    billable_output_tokens = usage.get("total_tokens", 0) - input_tokens
                    actual_usd = (
                        input_tokens / 1_000_000 * self.input_price
                        + billable_output_tokens / 1_000_000 * self.output_price
                    )
        adjust_to_actual(self.account, estimated_usd, actual_usd)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        # La llamada fallo por algo ajeno a este guard (error de red, auth, etc) --
        # liberar la reserva, no se gasto nada real.
        estimated_usd = self._pending_estimates.pop(run_id, 0.0)
        if estimated_usd:
            adjust_to_actual(self.account, estimated_usd, 0.0)
