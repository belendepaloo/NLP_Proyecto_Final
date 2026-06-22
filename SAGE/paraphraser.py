import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Imports de Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    _VERTEXAI_AVAILABLE = True
except ImportError:
    _VERTEXAI_AVAILABLE = False

#Para DeepSeek
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

VERTEX_MODEL_IDS = {
    "gemini": "gemini-2.5-flash",
    "grok": "xai/grok-4.1-fast-non-reasoning",
}

VERTEX_MAAS_IDS = {
    "deepseek": "deepseek-ai/deepseek-v3.2-maas",
}

ALL_VERTEX_ALIASES = {**VERTEX_MODEL_IDS, **VERTEX_MAAS_IDS}

# _PARAPHRASE_SYSTEM = (
#     "You are a paraphrasing assistant. "
#     "Rewrite the given text keeping the same meaning but using different wording. "
#     "Return ONLY the paraphrased text, no explanations, no numbering."
# )

_PARAPHRASE_SYSTEM = """
You are a paraphrasing assistant.

Rewrite the COMPLETE input text.

Requirements:
- Preserve all information.
- Do not summarize.
- Do not omit details.
- Maintain approximately the same length.
- Keep names, dates, locations and numbers unchanged.
- Rewrite sentence structure and wording.
- Return only the rewritten text.
"""


class Paraphraser:
    """
    Soporta dos providers:

      provider="local"
          T5 local via HuggingFace.

      provider="vertex_ai"
          Tres modelos vía Vertex AI:
          - "gemini" → gemini-2.5-flash (GenerativeModel)
          - "grok" → grok-4.1-fast-non-reasoning (GenerativeModel)
          - "deepseek" → deepseek-v3.2 MaaS (OpenAI-compatible endpoint)

    Parámetros
    ----------
    provider : str
        "local" | "vertex_ai"
    model_name : str
        local → HF model id
        vertex_ai → alias: "gemini" | "grok" | "deepseek"
    device : str | None
        Solo para provider="local".
    gcp_project : str | None
        GCP project id. Si None, se lee de GOOGLE_CLOUD_PROJECT.
    gcp_location : str
        Región Vertex AI. Default "us-central1".
    temperature : float
        Temperatura de generación para vertex_ai. Default 0.9.
    """
    def __init__(
        self,
        provider: str = "local",
        model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
        device: str | None = None,
        gcp_project: str | None = None,
        gcp_location: str = "us-central1",
        temperature: float = 0.9,
    ):
        self.provider = provider
        self.temperature = temperature
        self._model_alias = model_name
        self.usage_stats = {"calls": 0, "retries": 0, "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0,}

        # local (T5)                                                           
        if provider == "local":
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            print(f"[Paraphraser] Loading {model_name} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = (
                AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            )
            self.model.eval()
            print("[Paraphraser] Model loaded.")

        # VERTEX AI                                                            
        elif provider == "vertex_ai":
            if not _VERTEXAI_AVAILABLE:
                raise ImportError(
                    "google-cloud-aiplatform no está instalado.\n"
                    "  pip install google-cloud-aiplatform"
                )

            project = gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError(
                    "Especificá el proyecto GCP con gcp_project= "
                    "o con la variable GOOGLE_CLOUD_PROJECT."
                )

            self._gcp_project = project
            self._gcp_location = gcp_location

            vertexai.init(project=project, location=gcp_location)

            # GenerativeModels (Gemini, Grok)
            if model_name in VERTEX_MODEL_IDS:
                vertex_id = VERTEX_MODEL_IDS[model_name]
                print(
                    f"[Paraphraser] Vertex AI (GenerativeModel)"
                    f"proyecto={project}, región={gcp_location}, modelo={vertex_id}"
                )
                self._vertex_client = GenerativeModel(vertex_id)
                self._gen_config = GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=512,
                )
                self._vertex_type = "generative"

            # OpenAI-compatible (DeepSeek)
            elif model_name in VERTEX_MAAS_IDS:
                if not _OPENAI_AVAILABLE:
                    raise ImportError(
                        "openai no está instalado (necesario para DeepSeek MaaS).\n"
                        "  pip install openai"
                    )
                vertex_id = VERTEX_MAAS_IDS[model_name]
                endpoint_url = (
                    f"https://{gcp_location}-aiplatform.googleapis.com/v1beta1/"
                    f"projects/{project}/locations/{gcp_location}/endpoints/openapi"
                )
                print(
                    f"[Paraphraser] Vertex AI (MaaS/OpenAI-compat) – "
                    f"proyecto={project}, región={gcp_location}, modelo={vertex_id}"
                )
                # Autenticación via Application Default Credentials
                import google.auth
                import google.auth.transport.requests
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                auth_req = google.auth.transport.requests.Request()
                credentials.refresh(auth_req)

                self._maas_client = OpenAI(
                    base_url=endpoint_url,
                    api_key=credentials.token,
                )
                self._maas_model_id = vertex_id
                self._vertex_type = "maas"

                #guardo credentials por si expira el token
                self._gcp_credentials = credentials
                self._gcp_auth_req = auth_req

            else:
                raise ValueError(
                    f"Alias de modelo desconocido: '{model_name}'. "
                    f"Opciones: {list(ALL_VERTEX_ALIASES.keys())}"
                )

            print("[Paraphraser] Vertex AI listo.")

        else:
            raise ValueError(
                f"Provider no soportado: '{provider}'. Usá 'local' o 'vertex_ai'."
            )

    def generate_candidates(self, text: str, n: int = 3, min_length_ratio: float = 0.75) -> list[str]:
        if not text or not text.strip():
            return []

        if self.provider == "local":
            return self._generate_local(text, n=n)

        if self.provider == "vertex_ai":
            if self._vertex_type == "generative":
                return self._generate_generative(text, n=n, min_length_ratio=min_length_ratio)
            if self._vertex_type == "maas":
                return self._generate_maas(text, n=n)

        raise ValueError(f"Provider no soportado: {self.provider}")

    def _generate_local(self, text: str, n: int = 3) -> list[str]:
        prompt = f"paraphrase: {text.strip()}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=max(6, n * 2),
                num_return_sequences=n,
                temperature=1.2,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        candidates = []
        for output in outputs:
            candidate = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        return candidates[:n]
    
    def _generate_generative(self, text, n=3, min_length_ratio=0.75, max_retries=2):
        """n llamadas independientes para obtener variedad."""
        candidates = []
        stripped = text.strip()
        target_len = len(stripped)
        min_len = target_len * min_length_ratio

        base_prompt = (
            f"{_PARAPHRASE_SYSTEM}\n\n"
            f"The original text is {target_len} characters long. "
            f"Your rewritten text must be at least {int(min_len)} characters long.\n\n"
            f"Text to paraphrase:\n{stripped}"
        )

        for slot in range(n):
            prompt = base_prompt
            draft = ""
            for attempt in range(max_retries + 1):
                try:
                    response = self._vertex_client.generate_content(prompt, generation_config=self._gen_config)
                    draft = response.text.strip()
                    self.usage_stats["calls"] += 1
                    if attempt > 0:
                        self.usage_stats["retries"] += 1
                    usage = getattr(response, "usage_metadata", None)
                    if usage is not None:
                        self.usage_stats["prompt_tokens"] += usage.prompt_token_count
                        self.usage_stats["candidates_tokens"] += usage.candidates_token_count
                        self.usage_stats["total_tokens"] += usage.total_token_count
                except Exception as e:
                    print(f"[Paraphraser][generative] Error: {e}")
                    continue

                if len(draft) >= min_len:
                    break

                print(f"[Paraphraser][generative] slot {slot+1} intento {attempt+1}: {len(draft)}/{target_len} chars, reintentando...")
                prompt = (
                    f"{base_prompt}\n\nYour previous attempt was too short "
                    f"({len(draft)} characters, target is {int(min_len)}+ characters). "
                    f"Rewrite again, expanding the phrasing while preserving all "
                    f"information, to reach the required length. Previous attempt:\n{draft}"
                )

            if draft and draft not in candidates:
                candidates.append(draft)

        return candidates[:n]


    def _generate_maas(self, text: str, n: int = 3) -> list[str]:
        """
        Usa el endpoint OpenAI-compatible de Vertex AI para DeepSeek.
        Refresca el token ADC si expiró antes de cada lote de llamadas.
        """
        # Refrescar token si expiró
        import google.auth.transport.requests
        if not self._gcp_credentials.valid:
            self._gcp_credentials.refresh(self._gcp_auth_req)
            self._maas_client.api_key = self._gcp_credentials.token

        candidates = []

        for _ in range(n):
            try:
                response = self._maas_client.chat.completions.create(
                    model=self._maas_model_id,
                    messages=[
                        {"role": "system", "content": _PARAPHRASE_SYSTEM},
                        {"role": "user",   "content": f"Text to paraphrase:\n{text.strip()}"},
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                candidate = response.choices[0].message.content.strip()
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            except Exception as e:
                print(f"[Paraphraser][maas/deepseek] Error: {e}")

        return candidates[:n]