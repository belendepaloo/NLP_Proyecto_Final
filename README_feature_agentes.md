# Branch `feature/agentes` — capa de ingeniería del pipeline MIA

Esta branch construye la parte de **ingeniería** del proyecto sobre los métodos de
investigación ya existentes (`SAGE/`, `DUALTEST/`, `SiMIA/`, `DE_COP/`): un pipeline
agéntico que, a partir del nombre de un autor, scrapea su bibliografía, la cura, la
chunkea, la parafrasea y corre los métodos de MIA, ponderando sus probabilidades en una
sola estimación final mostrada en una interfaz web. El orquestador (deepagents) simula
la presencia de un humano en el flujo: puede debuggear cada etapa, invocar
sub-agentes que chequean que todo salga bien, y persistir lo que aprende en una skill
que se enriquece run a run.

Ver el README general (`README.md`) para el proyecto completo. Este documento es el
estado específico de esta branch.

## Arquitectura

```
mia_common/          Cliente target unificado (Groq/OpenAI/Anthropic/Google/HF-local,
                    con retry/backoff) + settings centralizadas (pydantic-settings)
agents/
  tools/              Adapters delgados sobre SAGE/DUALTEST/SiMIA/DE_COP/text_pipeline,
                      hoy funciones planas (se decoran como @tool en la Fase 2)
  ensemble/           combine.py (promedio pesado de los 3 metodos) + weights.yaml
  subagents/          (Fase 2, no existe todavia)
  skills/             (Fase 3, no existe todavia)
webapp/               (Fase 4, no existe todavia)
scripts/              Scripts ejecutables para probar cada fase sin esperar al resto
runs/                 Artifacts por run (gitignored)
```

Principio: `agents/tools/*.py` importan y envuelven los módulos de investigación
existentes — no se reimplementa lógica de scoring/paraphraseo, solo se adapta a una
interfaz común.

## Estado por fase

- ✅ **Fase 0 — refactor a módulos importables.** `DUALTEST` es paquete importable
  (`__init__.py` + imports compatibles con los notebooks viejos). `SiMIA/simia.py` y
  `DE_COP/decop.py` son versiones importables de los notebooks originales, recibiendo
  un `TargetClient` inyectado en vez de hardcodear un cliente de Groq. `mia_common/`
  centraliza ese cliente target y la config. `requirements.txt` (no existía ninguno).
- ✅ **Fase 1 — pipeline determinista end-to-end, sin agentes.**
  `scripts/run_pipeline_manual.py` corre limpieza/chunking → SAGE → DE-COP/SiMIA/DUALTEST
  → ensemble sobre texto real (3 novelas de Dickens + 1 artículo de Wikipedia,
  `processRawText/Datasets/dataset_len128.csv`), sin scraping ni curación automática
  todavía (textos fijos).
- ⬜ **Fase 2 — orquestador deepagents + sub-agentes + human-in-the-loop.** Pendiente.
  `deepagents`/`langgraph`/`tavily-python`/`langchain-google-genai` ya están en
  `requirements.txt` y se confirmó que `create_deep_agent(model=, tools=, subagents=,
  system_prompt=, skills=, interrupt_on=, store=...)` tiene la forma que el plan
  asume. Falta: `bibliography_agent` (Tavily), `curator_agent` (los dos LLM-judges de
  autoría/voz característica — no tienen precedente en el repo, se diseñan desde cero),
  `sage_qa_agent`, `mia_agent`, `flow_checker_agent`, y el `interrupt_on` para que el
  usuario pueda revisar/agregar textos antes de seguir.
- ⬜ **Fase 3 — skill persistente + flow-checkers en cada etapa.** Pendiente. La idea
  es `agents/skills/pipeline-learnings/SKILL.md` + `learnings.jsonl` +
  `calibration_history.csv`, cargada automáticamente por deepagents (`skills=[...]`) en
  cada run y actualizada al final de cada uno (éxito o falla parcial).
- ⬜ **Fase 4 — interfaz web.** Pendiente. FastAPI + Jinja2 + JS mínimo (fetch/EventSource),
  sin SPA. `fastapi`/`uvicorn`/`Jinja2` ya están en `requirements.txt`.

## Cómo correr lo que existe hoy

```bash
pip install -r requirements.txt
cp .env.example .env   # completar GROQ_API_KEY como minimo
```

**Prueba de humo del cliente target unificado** (DE-COP + SiMIA + DUALTEST contra el
mismo cliente Groq, sin pipeline ni agentes):

```bash
python scripts/verify_phase0_target_client.py
```

**Pipeline manual completo** (Fase 1) sobre los chunks ya preparados de Dickens +
Wikipedia:

```bash
python scripts/run_pipeline_manual.py --chunks-per-text 10
```

`--chunks-per-text` controla cuántos chunks por libro entran al pipeline costoso
(default: `mia_common.settings.chunks_per_text`, hoy 10) — pensado para subir/bajar sin
tocar código a medida que el dataset crezca (ej. ~20 libros x 10 chunks). Otras flags:
`--seed` (muestreo reproducible) y `--no-sage` (saltea SAGE si no están instalados
`transformer_lens`/`sae_lens`).

Sin `GROQ_API_KEY` configurada, ambos scripts corren igual pero muestran `[SKIP ...]`
en cada paso que necesita el modelo target, en vez de fallar.

## Limitaciones conocidas / próximos riesgos a resolver

- `processRawText.text_pipeline.chunk_text` (pysbd) escala mal sobre un libro entero de
  una sola vez (~5 min medido sobre "A Tale of Two Cities"). El pipeline manual usa el
  dataset ya chunkeado para evitar esto; si el scraping de la Fase 2 trae libros
  completos, va a necesitar chunkear por capítulo/página, no el libro entero junto.
- `DE_COP/` se nombra con guión bajo (no `DE-COP` con guión medio, como en la branch
  `feature/decop`) porque un guión medio no es válido en un nombre de paquete Python.
  Al mergear `feature/decop`, el notebook original cae en una carpeta `DE-COP/` (con
  guión) que queda solo como referencia de evaluación contra BookTection — no colisiona.
- No se probó todavía contra un `GROQ_API_KEY` real en este entorno de desarrollo (no
  había una key disponible) — la lógica de retry/backoff/`DailyCapError` de
  `mia_common/target_client.py` está escrita pero no ejercitada contra rate limits
  reales.
- `transformer_lens`/`sae_lens` (SPS de SAGE) no están instalados en este entorno por
  ser pesados — `agents/tools/sage_tools.py` lo detecta y lo reporta como `[SKIP SAGE]`
  en vez de romper, pero el paraphraseo real de SAGE no se probó end-to-end todavía.
- Los dos LLM-judges de curación que necesita la Fase 2 (¿es texto del autor o una
  reseña/resumen?, ¿es una pasaje característico de su voz o boilerplate genérico?) no
  tienen benchmark etiquetado para validar — van a quedar configurables y con revisión
  humana en casos borderline, no completamente automatizados.
