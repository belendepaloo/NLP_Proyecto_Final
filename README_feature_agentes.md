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
estado específico de esta branch — pensado como **handoff**: si retomás esto (o se lo
pasás a otra persona/instancia de Claude), empezá leyendo este archivo entero antes de
tocar código.

## TL;DR para retomar

- **Fase 0 y Fase 1 están terminadas y validadas contra Groq real** (no es teoría, se
  corrió de punta a punta varias veces). El ensemble discrimina member/non-member en
  promedio, aunque con superposición a nivel texto individual.
- **Fase 2 (orquestador deepagents + 5 sub-agentes + human-in-the-loop) está construida
  y compila/arranca**, pero la validación end-to-end real está **bloqueada esperando
  `GOOGLE_API_KEY`** (o `ANTHROPIC_API_KEY`) — ver el hallazgo de Groq como agent_model
  en la sección de esa fase, es importante leerlo antes de asumir que "no funciona".
- **Se agotó la cuota diaria (TPD) de Groq** probando todo esto (parece ser un límite
  por organización, no por key individual — las 4 keys no lo evitan, solo ayudan con el
  límite por minuto). El error es `mia_common.target_client.DailyCapError`; ya no tumba
  el proceso entero (se arregló para guardar progreso parcial), pero hasta que resetee
  la cuota no se puede seguir validando contra Groq real.
- Pendiente menor, deliberadamente no resuelto: el módulo se llama `SiMIA/` pero el
  método del paper que implementa se llama **"SimMIA"** (con dos emes: "Sim"+"MIA") —
  renombrar `SiMIA/` → `SimMIA/` (y `simia.py` → `simmia.py`) y actualizar los archivos
  que lo referencian (`grep -rn "SiMIA"`) es un buen primer paso de housekeeping.
- Última corrida grande de Fase 1 (antes de agotar la cuota):
  `runs/manual_phase1_smoke_test/` tiene resultados parciales en `chunks/` aunque
  `results/author_final.json` haya quedado del intento anterior (sin el fix de SiMIA).
- **Todas las llamadas a APIs externas se cachean** (`runs/_api_cache/`, ver
  `mia_common/cache.py`) — rerunear el mismo pipeline sobre el mismo texto no vuelve a
  gastar cuota. Esto es una regla de proyecto, no opcional: cualquier código nuevo que
  llame a una API externa tiene que pasar por `mia_common.target_client` (que ya cachea
  todo), no llamar al SDK del proveedor directo.

## Arquitectura

```
mia_common/
  target_client.py     Cliente target unificado (Groq/OpenAI/Anthropic/Google/HF-local),
                        con retry/backoff, cache transparente, y TargetClientPool para
                        repartir llamadas entre varias API keys en paralelo
  cache.py              Cache de llamadas a API (request+response) en runs/_api_cache/
  settings.py           Config centralizada (pydantic-settings) -- groq_api_keys(),
                        thresholds de SAGE/SiMIA/DUALTEST/curacion, todo en un lugar
agents/
  tools/                Adapters delgados sobre SAGE/DUALTEST/SiMIA/DE_COP/text_pipeline,
                        hoy funciones planas (se decoran como @tool en la Fase 2)
  ensemble/              combine.py (promedio pesado de los 3 metodos) + weights.yaml
  subagents/             (Fase 2, no existe todavia)
  skills/                (Fase 3, no existe todavia)
webapp/                  (Fase 4, no existe todavia)
scripts/                  Scripts ejecutables para probar cada fase sin esperar al resto
runs/                     Artifacts por run + cache de API (gitignored)
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

- ✅ **Fase 1 — pipeline determinista end-to-end, sin agentes.** Completa y validada
  contra Groq real en varias corridas. Detalle de lo que se fue encontrando y
  arreglando, en orden:

  1. **Pipeline base**: `scripts/run_pipeline_manual.py` corre limpieza/chunking → SAGE
     → DE-COP/SiMIA/DUALTEST → ensemble sobre texto real.
  2. **Dataset**: `processRawText/Datasets/dataset_len128.csv`, armado por
     `scripts/expand_dataset.py` — 5 novelas de Gutenberg como member (dominio público,
     seguro que están en cualquier training set) + 5 capítulos de novelas serializadas
     de Royal Road publicados en 2025/2026 como non-member (prosa narrativa real, no
     resúmenes — verificado vía el atributo `datetime` de cada página que son
     posteriores al cutoff de Llama 3.1 ~dic 2023). AO3 se intentó como fuente
     alternativa de non-members pero bloquea con error 525 (protección anti-scraping).
  3. **Bugs reales encontrados corriendo contra Groq** (no eran visibles sin probar con
     una key real):
     - `DUALTEST.target_model.APITarget.complete()` rompía con `max_new_tokens`
       duplicado.
     - `mia_common.target_client` reenviaba `do_sample=False` (param de HF) a APIs de
       chat completions, que no lo aceptan.
     - Un modelo chat-tuned sin instrucción explícita responde conversacionalmente en
       vez de continuar el texto — se agregó un system prompt de continuación en
       `RateLimitedAPITarget.complete()`.
     - `agents/ensemble/combine.normalize_dualtest` hacía `1 - p` directo sobre
       probabilidades que son productos de muchos tokens (~1e-17 a 1e-24) — colapsaba a
       una constante en floating point. Se normaliza por largo (media geométrica por
       token) antes de restar de 1.
     - `SAGE/sps.py` necesita el modelo gated `google/gemma-2b` de HuggingFace —
       requiere aceptar la licencia en huggingface.co + `HF_TOKEN`, y
       `mia_common/settings.py` necesitaba un "bridge" para propagar credenciales de
       `.env` a `os.environ` (librerías como `transformers` las leen directo de ahí, no
       del objeto `Settings`).
  4. **Validación con el dataset chico (1 member-set viejo)**: separación promedio
     member ~0.81-0.86 vs. non-member ~0.62.
  5. **Validación con el dataset ampliado (5M+5NM, 10 chunks/texto)**: promedio member
     0.806 vs. non-member 0.686 — direccionalmente correcto pero **con superposición a
     nivel texto individual** (un non-member puntuó 0.811, más alto que un member en
     0.761). Esperable: en esa corrida SiMIA todavía tenía el bug de abajo sin arreglar,
     y DUALTEST sigue sin el protocolo de calibración real corrido.
  6. **Verificación del paper de SimMIA** (`arXiv:2601.11314`, Yi & Li, CUHK,
     "Membership Inference on LLMs in the Wild", 2026 — el usuario lo señaló como
     supuesto nuevo SOTA en black-box, mejor que DE-COP): la fórmula de
     `SiMIA/simia.py` YA coincidía exactamente con el paper
     (`SimMIA(x) = -1/L * Σ s(xi|P⊕x<i)/s(xi|x<i)`, mismo embedder
     `all-MiniLM-L6-v2`). Pero el **pipeline llamaba a `simmia_score` con
     `non_member_prefix=""` (default)** — la perturbación P quedaba vacía, dando
     ratio≈1 siempre (señal constante). Y usaba `n_samples=1` contra el N=10 que
     recomienda el paper. Arreglado:
     - `scripts/build_simia_calibration_set.py` arma un prefijo non-member FIJO real
       (5 capítulos de Royal Road, **distintos** a los 5 usados en el dataset de
       evaluación, para no contaminar la comparación) → `SiMIA/non_member_calibration.txt`.
     - `SiMIA/simia.py` carga ese archivo por default y usa
       `mia_common.settings.simia_n_samples` (10) en vez de 1.
     - Costo real medido: con N=10, un chunk con ~9 posiciones de palabra hace hasta
       180 llamadas a Groq solo para SiMIA — **~5-6 minutos por chunk con una sola key**.
  7. **Cache de llamadas a API** (`mia_common/cache.py`, instrucción explícita del
     usuario: "toda llamada a una API se persiste, sin excepción"): cualquier llamada
     via `RateLimitedAPITarget.chat()` se cachea por contenido antes de pegarle a la
     API real. Para calls con sampling (temperature>0, el caso de SiMIA) hay que pasar
     `cache_sample_index` para que cada una de las N muestras tenga su propio slot — si
     no, la 2da..Nésima muestra pegarían todas contra el cache hit de la 1ra. Verificado:
     misma llamada, primera vez 44s, segunda vez 0.2s, mismo resultado.
  8. **Pool de múltiples API keys + paralelización** (dado el costo de (6)):
     `mia_common.settings.groq_api_keys()` lee `GROQ_API_KEYS` (CSV) o cae a
     `GROQ_API_KEY` única. `TargetClientPool`/`make_target_client_pool` en
     `mia_common/target_client.py` reparten clientes round-robin entre N keys, cada
     cliente con su propio lock/throttle (así dos chunks que comparten key no se pisan,
     pero chunks en keys distintas corren en paralelo de verdad).
     `scripts/run_pipeline_manual.py` ahora procesa chunks con un
     `ThreadPoolExecutor(max_workers=len(keys))`. SAGE y el reference model de DUALTEST
     son singletons locales de PyTorch (no garantizados thread-safe) — tienen sus
     propios locks (`agents/tools/sage_tools.py::_sage_lock`,
     `agents/tools/mia_tools.py::_dualtest_lock`) para serializar su uso sin bloquear
     las llamadas a la API (que sí corren en paralelo).
  9. **Persistencia de resultados crudos**: cada chunk ahora guarda en
     `runs/<run_id>/results/<titulo>.json` el resultado CRUDO de cada método
     (`decop`, `simia`, `dualtest`) además del score combinado del ensemble — no solo
     el número final.

- 🟡 **Fase 2 — orquestador deepagents + sub-agentes + human-in-the-loop.** Construida,
  **falta la validación end-to-end** (bloqueada por credenciales, ver abajo).
  - `agents/subagents/`: los 5 subagentes (`bibliography_agent`, `curator_agent`,
    `sage_qa_agent`, `mia_agent`, `flow_checker_agent`). En `curator_agent` el rubric de
    los dos LLM-judges (autoría / voz característica) vive en el system prompt del
    subagent — el agente mismo razona el veredicto; los tools
    (`agents/tools/curator_tools.py`) solo lo registran y aplican el threshold de
    `mia_common/settings.py` (no hardcodeado en el prompt).
  - `agents/orchestrator.py`: `create_deep_agent` con los 5 subagentes + tools propios
    de ensemble/artifacts. **Confirmado que compila y arranca** (con un modelo de
    prueba). `skills=` queda vacío a propósito (Fase 3).
  - `scripts/run_pipeline_agentic.py`: driver de CLI, maneja
    interrupt/`Command(resume=...)` por stdin (aprobar/editar/rechazar).
  - **No se decoran las funciones de `agents/tools/*.py` con `@tool`** — se confirmó
    (probando contra la API real instalada) que `create_deep_agent`/`SubAgent` aceptan
    callables planos directo (`tools: Sequence[BaseTool | Callable | dict]`), siempre
    que tengan type hints + docstring. Un bug real: `read_run_artifact` no tenía
    docstring y deepagents rechazaba armar el grafo entero por eso.
  - **Hallazgo importante, bloqueante**: probamos `groq:llama-3.3-70b-versatile` como
    `agent_model` (el LLM que razona DENTRO del orquestador, no el target de MIA) y la
    tool `task` que deepagents usa para delegar a subagentes le generaba argumentos mal
    formados — ese modelo no maneja con suficiente fidelidad el tool-calling anidado de
    deepagents. El nivel simple (una tool con interrupt directo en el orquestador, sin
    subagentes de por medio) sí funcionó perfecto con Groq. El plan original ya preveía
    Gemini como default (`settings.agent_model`) pero no había `GOOGLE_API_KEY`
    disponible en este entorno para probarlo — **pendiente de validar end-to-end en
    cuanto se consiga esa key** (o una de Anthropic, `langchain-anthropic` ya está
    instalado).
  - `TAVILY_API_KEY` tampoco está configurada — sin ella, `bibliography_agent` no puede
    buscar de verdad (el resto del pipeline se puede seguir probando con textos fijos
    mientras tanto).

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

Para paralelizar de verdad, agregar a `.env` (no está en `.env.example` actualmente,
ver nota abajo):
```
GROQ_API_KEYS=key1,key2,key3
```
(`mia_common.settings.groq_api_keys()` cae a `GROQ_API_KEY` única si esto no está.)

**Prueba de humo del cliente target unificado** (DE-COP + SiMIA + DUALTEST contra el
mismo cliente Groq, sin pipeline ni agentes):

```bash
python scripts/verify_phase0_target_client.py
```

**Si hace falta reconstruir el prefijo de calibración de SiMIA** (ya está commiteado en
`SiMIA/non_member_calibration.txt`, normalmente no hace falta correr esto):

```bash
python scripts/build_simia_calibration_set.py
```

**Pipeline manual completo** (Fase 1) sobre los chunks ya preparados (5 member +
5 non-member):

```bash
python scripts/run_pipeline_manual.py --chunks-per-text 10
```

`--chunks-per-text` controla cuántos chunks por libro entran al pipeline costoso
(default: `mia_common.settings.chunks_per_text`, hoy 10) — pensado para subir/bajar sin
tocar código a medida que el dataset crezca. Otras flags: `--seed` (muestreo
reproducible), `--no-sage` (saltea SAGE si no están instalados `transformer_lens`/
`sae_lens` o no se aceptó la licencia gated de `google/gemma-2b` en HuggingFace, ver
`.env.example` → `HF_TOKEN`), `--workers` (chunks en paralelo, default = cantidad de
keys en el pool). **Ojo con el costo**: con `simia_n_samples=10` (el default, fiel al
paper), cada chunk puede hacer hasta ~180 llamadas a Groq solo para SiMIA — con una
sola key son ~5-6 min/chunk. Con N keys en paralelo, dividir aproximadamente por N.

Sin `GROQ_API_KEY`/`GROQ_API_KEYS` configurada, los scripts corren igual pero muestran
`[SKIP ...]` en cada paso que necesita el modelo target, en vez de fallar.

**Para ampliar el dataset** con más libros member/non-member, ver
`scripts/expand_dataset.py` (agrega entradas a `NEW_SOURCES`, corre, regenera
`dataset_len128.csv`). Los non-member deben ser del mismo TIPO de texto que los member
(narrativo, no resúmenes/listas) para que la comparación tenga sentido.

## Limitaciones conocidas / próximos riesgos a resolver

- **Naming**: el módulo `SiMIA/` debería llamarse `SimMIA/` (y `simia.py` →
  `simmia.py`) para coincidir con el nombre real del método del paper — pendiente
  deliberadamente, ver TL;DR arriba.
- `.env.example` no documenta `GROQ_API_KEYS` (se agregó en sesión y después se sacó —
  posiblemente un linter o edición manual). El `.env` real del usuario sí la tiene y
  funciona; si hace falta volver a documentarla en el example, agregar:
  `GROQ_API_KEYS=` con un comentario explicando que es opcional, CSV, para paralelizar.
- `processRawText.text_pipeline.chunk_text` (pysbd) escala mal sobre un libro entero de
  una sola vez (~5 min medido sobre "A Tale of Two Cities"). Si el scraping de la
  Fase 2 trae libros completos, va a necesitar chunkear por capítulo/página, no el
  libro entero junto.
- `DE_COP/` se nombra con guión bajo (no `DE-COP` con guión medio, como en la branch
  `feature/decop`) porque un guión medio no es válido en un nombre de paquete Python.
  Al mergear `feature/decop`, el notebook original cae en una carpeta `DE-COP/` (con
  guión) que queda solo como referencia de evaluación contra BookTection — no colisiona.
- DUALTEST sigue siendo un proxy SIN CALIBRAR en el ensemble (ver
  `agents/ensemble/combine.normalize_dualtest`) — el bug de escala ya se arregló, pero
  el protocolo real de calibración de dos etapas (`DUALTEST/calibration.py`) no se
  corrió. La separación member/non-member observada es alentadora en promedio pero
  con superposición a nivel texto individual sobre muestras chicas (5-10 chunks/texto).
- Los dos LLM-judges de curación que necesita la Fase 2 no tienen benchmark etiquetado
  para validar — van a quedar configurables y con revisión humana en casos borderline.
- AO3 (Archive of Our Own) bloquea requests programáticos con error 525 — no se pudo
  usar como fuente de non-members narrativos, se usó Royal Road en su lugar.
- No se volvió a correr una validación grande (10 textos x 10 chunks) con el fix de
  SiMIA aplicado de punta a punta — la última corrida grande confirmada fue con
  `--chunks-per-text 5` y el fix recién aplicado; revisar
  `runs/manual_phase1_smoke_test/results/author_final.json` por el resultado mas
  reciente antes de sacar conclusiones definitivas.
