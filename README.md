# Membership Inference Attack en LLMs

Proyecto final de Procesamiento del Lenguaje Natural (UdeSA) — Isabel Castañeda, Belén
Depalo, Francesca Ragonesi, Isidro Valeriano, Florencia Zoffi.

**Membership Inference Attack (MIA)**: dado un texto y acceso de caja negra a un LLM,
estimar si ese texto formó parte de su corpus de entrenamiento. El proyecto combina
múltiples métodos complementarios (preferencia por paráfrasis, señales de
memorización, likelihood) para construir una estimación probabilística robusta, y
expone ese pipeline detrás de una interfaz web donde alcanza con tipear el nombre de
un autor.

El proyecto tiene dos partes:

- **Investigación**: los métodos de scoring MIA en sí (`SAGE/`, `DUALTEST/`, `SiMIA/`,
  `DE_COP/`) y su evaluación sobre benchmarks (WikiMIA, BookTection, artículos
  periodísticos). Ver `survey/main.tex` para el marco teórico completo.
- **Ingeniería**: el pipeline agéntico (`agents/`, `mia_common/`, `webapp/` cuando
  exista) que orquesta scraping → curación → chunking → paraphraseo → scoring →
  ensemble de punta a punta. Está en desarrollo activo en `feature/agentes` — ver
  [README_feature_agentes.md](README_feature_agentes.md) para el estado actual de esa
  parte.

## Estructura del repo

```
SAGE/              Paraphraser (T5 local o Vertex AI Gemini/Grok/DeepSeek) + SPS
                    (Semantic Persistence Score, Gemma-2B+SAE) + WordSim
DUALTEST/           Memorizacion via RLB/ESB: completion del target vs. probabilidad
                    de un modelo de referencia chico (Qwen2.5-0.5B)
SiMIA/               Next-word black-box ratio test (senal debil, AUC~0.51 medido)
DE_COP/              Multiple-choice: identificar el pasaje verbatim entre N paraphrases
mia_common/          Infraestructura compartida: cliente target unificado
                    (Groq/OpenAI/Anthropic/Google/HF-local) y settings centralizadas
agents/              Pipeline de ingenieria: tools que envuelven los metodos de arriba
                    + logica de ensemble (ver README_feature_agentes.md)
processRawText/      Scraping + limpieza (trafilatura) + chunking por tokens
build_all_datasets.py, check_dataset.py
                    Descarga/normalizacion de datasets (WikiMIA, BookTection, etc.)
                    y validacion de su metadata
dataset/             Metadata + datasets sampleados + outputs de SAGE por dataset
notebooks/           Notebooks de desarrollo/experimentacion (SAGE, DUALTEST, SiMIA)
results/             CSVs de resultados de experimentos de DUALTEST
papers/              Papers de referencia (SAGE, DUALTEST)
survey/              Marco teorico / survey del proyecto (LaTeX)
scripts/             Scripts ejecutables de la capa de ingenieria (ver README_feature_agentes.md)
```

## Setup

Python 3.12. Instalar dependencias:

```bash
pip install -r requirements.txt
```

`transformer_lens`/`sae_lens` (usados por `SAGE/sps.py` para el Semantic Persistence
Score) son pesados — instalarlos aparte si se va a correr SAGE de verdad:

```bash
pip install transformer_lens sae_lens
```

Copiar `.env.example` a `.env` y completar las API keys que se vayan a usar (Groq,
OpenAI, Anthropic, Google, Tavily, HuggingFace). `mia_common/settings.py` las lee
automaticamente.

## Métodos de scoring MIA

| Método | Estado | Señal medida |
|---|---|---|
| DUALTEST | Completo (RLB + ESB, calibración de dos etapas) | Memorización verbatim/near-duplicate |
| DE-COP | Completo (refactor en `DE_COP/decop.py`) | Identificación del pasaje verbatim entre paráfrasis |
| SAGE | Completo (paraphraser + SPS + WordSim) | Insumo para DE-COP, no es un MIA score en sí |
| SiMIA | Completo pero señal débil (AUC≈0.51 en BookTection) | Ratio de predicción de siguiente palabra |

Cada método puede correr contra cualquier backend de modelo target a través de
`mia_common/target_client.py` (Groq, OpenAI, Anthropic, Google, o un modelo HF local).

## Más información

- Marco teórico completo: `survey/main.tex`
- Estado de la capa de ingeniería (en desarrollo): [README_feature_agentes.md](README_feature_agentes.md)
