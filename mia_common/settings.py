"""
settings.py — configuracion centralizada para la capa de ingenieria (agents/, webapp/,
mia_common/). El resto del repo (SAGE/, DUALTEST/) sigue leyendo sus propias env vars
sueltas via os.environ.get -- esto es additivo, no las reemplaza.
"""

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
