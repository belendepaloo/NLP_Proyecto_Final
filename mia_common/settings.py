"""
settings.py — configuracion centralizada para la capa de ingenieria (agents/, webapp/,
mia_common/). El resto del repo (SAGE/, DUALTEST/) sigue leyendo sus propias env vars
sueltas via os.environ.get -- esto es additivo, no las reemplaza.
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mismo patron que DUALTEST/experiment_utils.py


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM que razona dentro de los agentes (orquestador + subagentes), separado del
    # modelo target bajo test. Formato 'provider:model' (init_chat_model de LangChain).
    # gemini-2.0-flash da 429 (limit: 0 en el free tier) en keys nuevas de Cloud Console;
    # gemini-2.5-flash si tiene cuota free-tier disponible -- verificado contra la API real.
    # gemini-2.5-flash genera tool calls malformados con cierta frecuencia para
    # write_todos (deepagents) -- confirmado en vivo (finish_reason=MALFORMED_FUNCTION_CALL,
    # el mensaje queda vacio y el orquestador se frena en silencio, sin error visible).
    # gemini-2.5-pro no mostro este problema en las mismas pruebas.
    agent_model: str = "google_vertexai:gemini-2.5-pro"

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
    google_application_credentials: str | None = None
    hf_token: str | None = None

    # Limite duro de gasto (USD) para agent_model cuando es un modelo facturado por uso
    # (Vertex AI) -- ver mia_common/spend_guard.py. El calculo de costo ANTES de cada
    # llamada es una estimacion (no se sabe cuanto va a generar el modelo todavia).
    # Tope en $15.00 (usuario agrego $15 de credito nuevo y pidio el tope exacto en
    # ese valor, sin margen adicional esta vez).
    agent_model_spend_cap_usd: float = 40.0

    # Lista de keys de Groq separadas por coma (GROQ_API_KEYS=key1,key2,key3), para
    # paralelizar con un pool de clientes (mia_common.target_client.TargetClientPool)
    # en vez de quedar atado al rate limit de una sola key. Numero de keys no fijo --
    # se agregan las que se tengan. Si no esta seteada, cae a [groq_api_key].
    groq_api_keys_csv: str | None = Field(default=None, alias="GROQ_API_KEYS")

    def groq_api_keys(self) -> list[str]:
        if self.groq_api_keys_csv:
            return [k.strip() for k in self.groq_api_keys_csv.split(",") if k.strip()]
        return [self.groq_api_key] if self.groq_api_key else []

    # Rate limiting / reintentos para clientes API (Groq free tier es el caso mas chico)
    target_min_seconds_between_calls: float = 2.1
    target_max_retries: int = 6

    # Tope de caracteres por documento que bibliography_agent puede chunkear (~15
    # paginas de un libro tipico, ~2000 caracteres/pagina) -- el libro completo NUNCA
    # se procesa ni se guarda entero, ni en disco ni en el contexto de un LLM, sin
    # importar cuanto mida la fuente real (se vio en vivo: Gutenberg/archive.org dan
    # libros de ~280.000-700.000 caracteres). De sobra para los
    # curator_target_chunks_per_text chunks curados que se necesitan por texto.
    bibliography_max_chars_per_document: int = 30_000

    # Cuantas rondas de "buscar candidatos de reemplazo" intenta el orquestador cuando
    # curator_agent descarta un candidato (autoria=drop, o 0 chunks "keep" tras voz) --
    # acotado para no generar un loop de costo si el autor simplemente no tiene mas
    # textos disponibles online.
    bibliography_max_replacement_rounds: int = 2

    # Cuantos chunks por texto entran al pipeline costoso (SAGE + 3 metodos MIA).
    # Control de costo/computo, no de cuantos chunks existen en el dataset -- eso lo
    # define MAX_CHUNKS_PER_PAGE en scrape_clean_chunk.ipynb al armar el CSV. Pensado
    # para poder bajarlo/subirlo facil sin tocar codigo (ver scripts/run_pipeline_manual.py
    # y --chunks-per-text, que sigue pudiendo overridear esto por run).
    chunks_per_text: int = 10
    chunk_sample_seed: int = 42

    # Cuantos chunks CURADOS (que pasaron el filtro de voz) busca curator_agent por
    # documento en la Fase 2 agentica -- distinto de chunks_per_text de arriba (esa es
    # una muestra de chunks CRUDOS, sin curar, que usa la Fase 1 manual). Subido de 5 a
    # 20 a pedido del usuario (mas chunks = estadistica mas robusta para la
    # probabilidad final) -- OJO: el juicio de voz es UNA llamada "thinking" de
    # gemini-2.5-pro POR CHUNK, el driver de costo mas grande del pipeline (medido en
    # vivo: juzgar la voz de 51 chunks crudos consumio la mayoria de un presupuesto de
    # $4.50). Subir esto a 20 multiplica ese costo ~4x respecto del default anterior
    # (5) -- vigilar agent_model_spend_cap_usd. curator_agent sigue pidiendo de a poco
    # (un batch chico, despues de a uno mas si algo se descarta) hasta llegar a este
    # numero, no juzga el documento entero de una.
    curator_target_chunks_per_text: int = 20
    curator_initial_batch_size: int = 21  # target + margen chico, para no tener que pedir de a uno desde el principio

    # DUALTEST contra un target sin tokenizer (API): split_by_words necesita
    # prefijo+continuacion en PALABRAS que entren dentro de un chunk de ~128 tokens
    # (mediana ~80-85 palabras, ver processRawText/Datasets/dataset_len128.csv). Con
    # los defaults del paper (50 palabras de prefijo) la continuacion quedaba vacia o
    # casi vacia para la mitad de los chunks. Bajado para que la mayoria de los chunks
    # tengan continuacion no-trivial.
    dualtest_prefix_len: int = 40
    dualtest_continuation_len: int = 24
    dualtest_max_new_tokens: int = 24

    # SiMIA (ver SiMIA/simia.py) -- parametros portados de
    # simmia_decop/notebooks/simMIA.ipynb (commit a63ec742, PR #2 feature/decop_simmia):
    # N=3 samples por posicion y un prefijo non-member FIJO (no aleatorio) de varios
    # cientos de caracteres.
    simia_n_samples: int = 3
    simia_calibration_chars: int = 600

    # Apagado temporal de SiMIA en el pipeline agentico (Fase 2) -- decision del
    # usuario: "todavia no esta terminado", validar primero que el resto del pipeline
    # (orquestador, agentes, webapp) funciona bien de punta a punta con DE-COP +
    # DUALTEST, sumar SiMIA de vuelta despues. NO afecta a la Fase 1 manual
    # (scripts/run_pipeline_manual.py sigue corriendo los 3 metodos) -- esto es
    # especifico de agents/subagents/mia_agent.py.
    simia_enabled: bool = False

    # Thresholds de curacion (ver agents/tools/curator_tools.py) -- viven aca, no
    # hardcodeados en los prompts, para que la skill persistente pueda ajustarlos.
    authorship_min_confidence: float = 0.6
    authorship_review_band: tuple[float, float] = (0.4, 0.6)
    voice_min_distinctiveness: float = 0.55

    # QA de SAGE (ver agents/tools/sage_tools.py)
    sage_min_sps: float = 0.7
    sage_min_length_ratio: float = 0.75

    # Cuantos candidatos de paraphrase genera SAGE por segmento, y cuantos de esos
    # se quedan (los de mejor final_score = sps - wordsim) antes de exponerlos a
    # DE-COP como paraphrase_candidates -- generar de mas y filtrar da mejor calidad
    # que generar exactamente los que se necesitan. DE-COP necesita >=3 para no
    # skippear el chunk, por eso sage_n_candidates_kept no deberia bajar de 3.
    sage_n_candidates_generated: int = 4
    sage_n_candidates_kept: int = 3

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
    "GOOGLE_APPLICATION_CREDENTIALS": settings.google_application_credentials,
    "GROQ_API_KEY": settings.groq_api_key,
    "OPENAI_API_KEY": settings.openai_api_key,
    "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    "GOOGLE_API_KEY": settings.google_api_key,
    "TAVILY_API_KEY": settings.tavily_api_key,
}
for _env_key, _env_value in _ENV_BRIDGE.items():
    if _env_value and not os.environ.get(_env_key):
        os.environ[_env_key] = _env_value
