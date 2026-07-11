# MIA Pipeline — App web + orquestador agéntico

Interfaz web y pipeline de punta a punta para **Membership Inference Attack (MIA)** sobre
LLMs: tipeás el nombre de un autor y el sistema busca su bibliografía, la cura, la
parafrasea y corre varios métodos de MIA para estimar si sus textos formaron parte del
corpus de entrenamiento del modelo target — con un humano en el loop antes de avanzar.

Proyecto final de NLP (UdeSA) — Isabel Castañeda, Belén Depalo, Francesca Ragonesi,
Isidro Valeriano, Florencia Zoffi.

---

## ¿Qué hace?

Dado un autor, corre este pipeline agéntico de principio a fin:

```
   Autor (texto en un form)
        │
        ▼
 ┌──────────────────┐
 │ 1. Bibliografía  │  bibliography_agent: web search (Tavily) + scraping →
 │                  │  propone N textos candidatos del autor
 └────────┬─────────┘
          │  ⏸  PAUSA — revisión humana (aprobar / editar / rechazar)
          ▼
 ┌──────────────────┐
 │ 2. Curación      │  curator_agent: descarga los textos (Gutenberg, etc.),
 │  · Autoría       │  segmenta en chunks (~500 palabras), verifica autoría y
 │  · Voz           │  selecciona los pasajes más característicos de la voz del autor
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │ 3. SAGE          │  sage_qa_agent: parafrasea cada chunk (T5 local o Vertex AI)
 │  (paraphraser)   │  guiado por features SAE (Semantic Persistence Score) + QA
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │ 4. Scoring MIA   │  mia_agent corre 3 métodos por chunk contra el modelo target:
 │  · DE-COP        │   → multiple-choice: ¿identifica el pasaje verbatim entre paráfrasis?
 │  · DUALTEST      │   → memorización verbatim/near-dup (RLB + ESB)
 │  · SiMIA         │   → next-word black-box ratio test (señal débil)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │ 5. Ensemble      │  combina los 3 métodos en una probabilidad final por chunk y
 │                  │  por autor (pesos en agents/ensemble/weights.yaml)
 └──────────────────┘
```

**Modelo target** (el "black box" bajo test): se elige en el form. Opciones y modelos:

| Provider | Modelo target |
|---|---|
| `groq` | `llama-3.1-8b-instant` |
| `openai` | `gpt-4o-mini` |
| `anthropic` | `claude-haiku-4-5` |
| `google` | `gemini-2.5-flash` |

**Pesos del ensemble** (`agents/ensemble/weights.yaml`): `dualtest 0.50`, `decop 0.35`,
`simia 0.15`. SiMIA es deliberadamente bajo (AUC medido ≈0.51 en BookTection, casi azar).

---

## Arquitectura

```
webapp/               Interfaz web (FastAPI + Jinja2, server-rendered, sin SPA)
  main.py             Rutas: GET / (form), POST /runs, GET /runs/{id}, SSE /stream,
                      POST /runs/{id}/decide (revisión humana), endpoints de demo
  run_manager.py      Corazón del backend: RunHandle, threads de background, replay
                      mode, demo mode, caches, donor runs
  progress.py         Reconstruye el progreso del pipeline leyendo artifacts de disco
  results.py          Arma la vista de resultados finales
  templates/          index.html (form), run.html (pantalla del run), base.html
  static/style.css

agents/               Pipeline de ingeniería (orquestador agéntico)
  orchestrator.py     create_deep_agent con los 5 subagentes + tools deterministicos
  subagents/          bibliography, curator, sage_qa, mia, flow_checker
  tools/              tools que envuelven cada método (bibliography, chunk, curator,
                      mia, sage) + fs_tools (artifacts en runs/<run_id>/)
  ensemble/           combine.py + weights.yaml

SAGE/ DUALTEST/ SiMIA/ DE_COP/    Los métodos de scoring MIA en sí
mia_common/           Infraestructura compartida: target_client unificado + settings
processRawText/       Scraping + limpieza (trafilatura) + chunking por tokens

runs/                 Salida de cada run: runs/<run_id>/ con sus artifacts por etapa
  _api_cache/         Cache cross-run de llamadas a la API (SHA256 del payload)
  _sage_cache/        Cache cross-run de paráfrasis de SAGE
  _checkpoints.sqlite Checkpoints de LangGraph (para retomar runs)
```

Cada método corre contra cualquier backend de modelo target a través de
`mia_common/target_client.py` (Groq / OpenAI / Anthropic / Google / HF local).

### Estado en memoria vs. disco

El estado de los runs (`RunHandle`) vive **en memoria del proceso de uvicorn** — no
sobrevive un reinicio del server. Pero los artifacts reales quedan en `runs/<run_id>/`,
así que si el server se reinicia (p.ej. con `--reload` al editar código), la pantalla de
un run viejo se reconstruye en modo degradado desde disco en vez de dar 404.

---

## Setup

Requiere **Python 3.12** (langgraph/deepagents pueden ser incompatibles con el Python del
sistema).

```bash
conda create -n mia-agentes python=3.12
conda activate mia-agentes
pip install -r requirements.txt
cp .env.example .env        # completar las keys que se vayan a usar
```

`transformer_lens` / `sae_lens` (usados por `SAGE/sps.py` para el Semantic Persistence
Score) son pesados; ya están en `requirements.txt` pero se pueden saltear si no vas a
correr SAGE de verdad.

### Variables de entorno (`.env`)

```bash
# Modelo target (elegir al menos uno según el provider que uses en el form)
GROQ_API_KEY=          # llama-3.1-8b-instant  (el default más barato)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# LLM de los agentes (orquestador + subagentes)
GOOGLE_CLOUD_PROJECT=   # Vertex AI para el agent_model

# Bibliography agent
TAVILY_API_KEY=         # web search

# Modelos locales (reference model de DUALTEST, etc.)
HF_TOKEN=
```

Para **paralelizar** las llamadas a Groq, agregar (no está en `.env.example`):

```bash
GROQ_API_KEYS=key1,key2,key3
```

`mia_common.settings.groq_api_keys()` reparte los chunks entre las keys del pool; cae a
`GROQ_API_KEY` única si esto no está.

Sin keys, todo corre igual pero cada paso que necesita el modelo target muestra
`[SKIP ...]` en vez de fallar.

---

## Cómo correr la app web

```bash
uvicorn webapp.main:app --reload
```

Abrir **http://127.0.0.1:8000/**.

> Si `uvicorn` no está en el PATH del entorno, usar la ruta completa del Python 3.12,
> p.ej. `~/Library/Python/3.12/bin/uvicorn webapp.main:app --reload`.
>
> Si el puerto ya está ocupado: `lsof -ti :8000 | xargs kill -9`.

Flujo normal en la interfaz:

1. Tipeás el autor, elegís cuántos textos candidatos pedir y el modelo target.
2. El pipeline arranca en un thread de background; la pantalla se refresca vía SSE.
3. Cuando el bibliography_agent propone candidatos, el run **pausa** para revisión
   humana: aprobás, editás la lista o la rechazás.
4. Sigue con curación → SAGE → scoring MIA → ensemble, y muestra la probabilidad final.

---

## Modo Demo 🎬

Pensado para **presentar el pipeline en vivo** sobre un autor que ya fue procesado antes,
sin depender de la API ni esperar horas: reproduce el run original **paso a paso**,
avanzando con un botón **"Siguiente"**.

### Cómo activarlo

En el form de inicio, tildar el checkbox **🎬 Modo Demo** antes de arrancar el run.
Requiere que exista un run previo **completo** del mismo autor (el "donor").

### Qué pasa

- Al arrancar, el sistema busca un **donor run**: el run previo más reciente del mismo
  autor que tenga curación completa **y** scores MIA completos (`require_mia_scores`).
- Reproduce el pipeline entero — bibliografía, revisión humana, curación, SAGE y los 3
  métodos MIA — pero **cada paso pausa** hasta que hacés click en "▶ Siguiente".
- Los scores **no se recalculan**: se leen de los artifacts del donor
  (`mia_scores`, `sage`, `curation`). Por eso el modo demo es **determinístico** — una
  vez cacheado, siempre muestra exactamente los mismos resultados, sin llamar a la API.
- El diagrama del pipeline arriba se **sincroniza** con el paso actual (nodo activo,
  nodos completados con ✓), y el log de agentes va apareciendo en vivo.
- La granularidad es **por chunk**: dentro del scoring MIA, cada chunk pasa por SAGE →
  DE-COP → DUALTEST → SiMIA → probabilidad, con un click entre cada uno.

### Cómo funciona por dentro

- `RunHandle.demo_mode=True` y `donor_run_id` apuntando al run donante.
- `_demo_pause(handle)` bloquea el thread de background con un `threading.Event.wait()`
  hasta que el usuario hace click; el endpoint `POST /runs/{id}/demo-next` lo libera.
- El SSE es **dual-canal**: `event: agentlog` (append al log sin recargar),
  `event: demopause` (mostrar/ocultar el botón), y `data: reload` (recarga completa
  cuando cambian artifacts o el status).
- **Fallback de polling**: como copiar la curación del donor puede escribir muchos
  archivos y disparar varios `reload` seguidos (posible carrera que se "come" un evento
  `demopause`), la pantalla también consulta `GET /runs/{id}/demo-state` cada 1.5s. Si el
  botón está oculto pero el server dice `paused=true`, lo muestra igual. Así el botón
  "Siguiente" aparece de forma confiable en cada pausa.

### Endpoints de demo

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/runs/{id}/demo-next` | Libera la pausa actual → avanza al próximo paso |
| `GET`  | `/runs/{id}/demo-state` | Devuelve `{paused, event_count}` (polling fallback) |

---

## Replay mode (fuera de la demo)

Independiente del modo demo: si arrancás un run normal para un autor que **ya fue
procesado**, el sistema entra en **replay** automáticamente — copia los artifacts de
bibliografía y curación del donor y salta directo a SAGE + MIA desde cache (segundos en
vez de horas). Si no hay donor, corre el pipeline completo con LangGraph desde cero.

La diferencia con la demo: replay corre de corrido (sin pausas ni botón) y **puede**
recalcular scores si no están cacheados; la demo pausa en cada paso y **siempre** lee del
donor.

---

## Scripts de línea de comandos

Además de la app web, el pipeline se puede correr desde scripts:

```bash
# Prueba de humo del cliente target (DE-COP + SiMIA + DUALTEST, sin pipeline ni agentes)
python scripts/verify_phase0_target_client.py

# Pipeline manual completo sobre chunks ya preparados
python scripts/run_pipeline_manual.py --chunks-per-text 10

# Pipeline agéntico completo (mismo orquestador que la web) por CLI
python scripts/run_pipeline_agentic.py --run-id <id>

# Retomar un run que quedó a medias desde su checkpoint
python scripts/resume_mia_direct.py --run-id <id>
```

Flags útiles de `run_pipeline_manual.py`: `--seed` (muestreo reproducible), `--no-sage`
(saltea SAGE si no están `transformer_lens`/`sae_lens` o la licencia gated de
`google/gemma-2b`), `--workers` (chunks en paralelo, default = cantidad de keys en el
pool). **Ojo con el costo**: con `simia_n_samples=10` cada chunk puede hacer ~180
llamadas a Groq solo para SiMIA (~5-6 min/chunk con una sola key; dividir por N keys en
paralelo).

---

## Métodos de scoring MIA

| Método | Señal medida | Estado |
|---|---|---|
| **DUALTEST** | Memorización verbatim / near-duplicate (RLB + ESB, calibración de 2 etapas) | Completo |
| **DE-COP** | Identificación del pasaje verbatim entre paráfrasis (multiple-choice) | Completo |
| **SAGE** | Paraphraser + SPS + WordSim — insumo de DE-COP, no es un score MIA en sí | Completo |
| **SiMIA** | Ratio de predicción de siguiente palabra (señal débil, AUC≈0.51) | Completo |

---

## Caches y determinismo

- **`runs/_api_cache/`** — cachea toda llamada a la API (input + output) por SHA256 del
  payload, cross-run. Volver a correr el mismo autor no vuelve a pagar la API.
- **`runs/_sage_cache/`** — cachea las paráfrasis de SAGE, cross-run.
- **`runs/_checkpoints.sqlite`** — checkpoints de LangGraph para retomar runs.

Gracias a estos caches, el replay y el modo demo son rápidos y determinísticos.

---

## Más información

- Marco teórico completo del proyecto: `survey/main.tex`
- Detalle histórico de la capa de ingeniería (fases, decisiones): `README_feature_agentes.md`
- Overview general del repo (investigación + ingeniería): `README.md`
