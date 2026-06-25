"""
settings.py — configuracion centralizada para la capa de ingenieria (agents/, webapp/,
mia_common/). El resto del repo (SAGE/, DUALTEST/) sigue leyendo sus propias env vars
sueltas via os.environ.get -- esto es additivo, no las reemplaza.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mismo patron que DUALTEST/experiment_utils.py


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM que razona dentro de los agentes (orquestador + subagentes), separado del
    # modelo target bajo test. Formato 'provider:model' (init_chat_model de LangChain).
    agent_model: str = "google_genai:gemini-2.0-flash"

    # Modelo "black box" target al que se le hace el MIA. Configurable por run desde la
    # webapp; estos son solo los defaults.
    target_provider: str = "groq"  # "groq" | "openai" | "anthropic" | "google" | "hf_local"
    target_model_name: str = "llama-3.1-8b-instant"

    # Reference model de DUALTEST: siempre local/white-box, nunca el target.
    reference_model_name: str = "Qwen/Qwen2.5-0.5B"

    # API keys (sin default -- deben venir de env/.env)
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    tavily_api_key: str | None = None
    google_cloud_project: str | None = None
    hf_token: str | None = None

    # Rate limiting / reintentos para clientes API (Groq free tier es el caso mas chico)
    target_min_seconds_between_calls: float = 2.1
    target_max_retries: int = 6

    # Cuantos chunks por texto entran al pipeline costoso (SAGE + 3 metodos MIA).
    # Control de costo/computo, no de cuantos chunks existen en el dataset -- eso lo
    # define MAX_CHUNKS_PER_PAGE en scrape_clean_chunk.ipynb al armar el CSV. Pensado
    # para poder bajarlo/subirlo facil sin tocar codigo (ver scripts/run_pipeline_manual.py
    # y --chunks-per-text, que sigue pudiendo overridear esto por run).
    chunks_per_text: int = 10
    chunk_sample_seed: int = 42

    # DUALTEST contra un target sin tokenizer (API): split_by_words necesita
    # prefijo+continuacion en PALABRAS que entren dentro de un chunk de ~128 tokens
    # (mediana ~80-85 palabras, ver processRawText/Datasets/dataset_len128.csv). Con
    # los defaults del paper (50 palabras de prefijo) la continuacion quedaba vacia o
    # casi vacia para la mitad de los chunks. Bajado para que la mayoria de los chunks
    # tengan continuacion no-trivial.
    dualtest_prefix_len: int = 40
    dualtest_continuation_len: int = 24
    dualtest_max_new_tokens: int = 24

    # Thresholds de curacion (ver agents/tools/curator_tools.py) -- viven aca, no
    # hardcodeados en los prompts, para que la skill persistente pueda ajustarlos.
    authorship_min_confidence: float = 0.6
    authorship_review_band: tuple[float, float] = (0.4, 0.6)
    voice_min_distinctiveness: float = 0.55

    # QA de SAGE (ver agents/tools/sage_tools.py)
    sage_min_sps: float = 0.7
    sage_min_length_ratio: float = 0.75

    # Paths
    runs_dir: Path = PROJECT_ROOT / "runs"
    skill_dir: Path = PROJECT_ROOT / "agents" / "skills" / "pipeline-learnings"
    ensemble_weights_path: Path = PROJECT_ROOT / "agents" / "ensemble" / "weights.yaml"


settings = Settings()

# Algunas librerias de terceros leen credenciales directo de os.environ, no de este
# objeto Settings (huggingface_hub/transformers via HF_TOKEN para modelos gated como
# google/gemma-2b en SAGE/sps.py; SAGE/paraphraser.py via GOOGLE_CLOUD_PROJECT). Sin
# este bridge, completar .env no alcanzaba para esos casos -- se detecto al intentar
# descargar Gemma-2B. No se pisa nada que el usuario ya haya exportado en su shell.
_ENV_BRIDGE = {
    "HF_TOKEN": settings.hf_token,
    "GOOGLE_CLOUD_PROJECT": settings.google_cloud_project,
    "GROQ_API_KEY": settings.groq_api_key,
    "OPENAI_API_KEY": settings.openai_api_key,
    "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    "GOOGLE_API_KEY": settings.google_api_key,
    "TAVILY_API_KEY": settings.tavily_api_key,
}
for _env_key, _env_value in _ENV_BRIDGE.items():
    if _env_value and not os.environ.get(_env_key):
        os.environ[_env_key] = _env_value
