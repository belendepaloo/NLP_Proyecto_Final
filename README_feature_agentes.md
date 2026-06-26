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

- **Fase 0 y Fase 1 están terminadas y validadas contra Groq real**. El ensemble
  discrimina member/non-member en promedio, aunque con superposición a nivel texto
  individual.
- **Fases 2, 3 y 4 están construidas.** Esta es la branch de varias personas trabajando
  en paralelo (ver commits de `castaisa`/Isabel además de los de `belendepalo`) — si
  retomás esto, hacé `git pull` primero y `pip install -r requirements.txt` de nuevo
  (cada quien fue agregando dependencias nuevas: `langchain-google-vertexai`,
  `langgraph-checkpoint-sqlite`, `python-multipart`, etc.).
- **Se logró el primer run 100% real de la historia de este proyecto** (Tavily real +
  descarga real de Gutenberg + Groq real + Gemini real como `agent_model`): autor Edgar
  Allan Poe, candidato real aprobado, texto real persistido (39481 caracteres). Se cortó
  en la pausa post-aprobación por agotar cuota de Gemini, antes de llegar a
  `curator_agent` — pero la parte de bibliografía+aprobación humana ya está confirmada
  funcionando con datos reales, no solo mocks.
- **Bug de integridad serio, encontrado y arreglado en esta sesión** (ver Fase 2 para
  el detalle completo): el orquestador podía fabricar una lista de candidatos falsa y
  guardarla como "aprobada" cuando bibliography_agent no encontraba nada, sin que
  ninguna revisión humana real hubiera ocurrido. Arreglado anclando la persistencia del
  artifact al propio tool que la pausa humana protege (`propose_candidate_texts`), no a
  que el orquestador "se acuerde" de guardarlo después. Esto era más importante que
  cualquier otra cosa pendiente — un resultado de MIA sobre un libro que nunca se bajó
  de verdad invalida todo lo que viene después.
- **El cuello de botella real hoy es la cuota gratuita de Gemini como `agent_model`**:
  20 requests/día, **por modelo** (no por key) — `gemini-2.5-flash`, `gemini-2.5-pro`,
  `gemini-flash-latest` cada uno tiene su propio contador, así que probar con un alias
  distinto da algo de margen pero se agota rápido igual. Parece resetear a medianoche
  UTC (21hs Argentina). Opciones para destrabar esto en serio: habilitar billing en el
  proyecto de Google Cloud, configurar Vertex AI (`gcloud auth application-default
  login` — no viaja por `.env`, es por máquina) que es el default actual
  (`google_vertexai:gemini-2.5-pro`, decisión de Isabel), o conseguir `ANTHROPIC_API_KEY`
  (sigue vacía). La cuota de Groq como *target* no se volvió a agotar en esta sesión.
- **Credenciales en `.env` de esta máquina**: `GROQ_API_KEY`/`GROQ_API_KEYS` (4 keys),
  `TAVILY_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `HF_TOKEN` ya están.
  `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` siguen vacías.
- **`mia_agent` estaba roto para CUALQUIER target** (otro bug encontrado esta sesión,
  ver Fase 2): `client: TargetClient` expuesto crudo en una tool rompe la conversión a
  function-calling de Gemini. Arreglado con closures (`functools.partial` NO sirve para
  esto). Validado offline con el `TargetClient` real de Groq bindeado.
- Pendiente menor, deliberadamente no resuelto: renombrar `SiMIA/` → `SimMIA/` (el
  método del paper se llama con dos emes) — housekeeping, no urgente.
- **Todas las llamadas a APIs externas se cachean** (`runs/_api_cache/`) — regla de
  proyecto no opcional, cualquier código nuevo tiene que pasar por
  `mia_common.target_client`, nunca el SDK del proveedor directo.

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
  subagents/             Los 5 subagentes de Fase 2 (ver "Estado por fase")
  skills/pipeline-learnings/  SKILL.md + learnings.jsonl + calibration_history.csv
                        (Fase 3) -- backend scopeado aca, no al resto del repo
webapp/                  FastAPI + Jinja2 (Fase 4) -- run_manager.py es el puente
                        sincrono/async con el orquestador, ver "Estado por fase"
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
  - **Hallazgo bloqueante de la sesión anterior, ya VALIDADO**: probamos
    `groq:llama-3.3-70b-versatile` como `agent_model` (el LLM que razona DENTRO del
    orquestador, no el target de MIA) y la tool `task` que deepagents usa para delegar
    a subagentes le generaba argumentos mal formados — ese modelo no maneja con
    suficiente fidelidad el tool-calling anidado de deepagents. Con `GOOGLE_API_KEY`
    real conseguida en esta sesión, se confirmó que **Gemini sí delega bien**: invocar
    el orquestador con un mensaje real generó una llamada bien formada a `task` hacia
    `bibliography_agent`, que a su vez llamó bien a su propia tool `tavily_search` (paró
    ahí solo por `TAVILY_API_KEY` faltante, un `RuntimeError` claro y esperado, no un
    crash de tool-calling). Conclusión: usar Gemini (o Anthropic) como `agent_model`,
    nunca un Groq chat model para este rol.
  - **2 bugs reales encontrados validando esto contra la API real de Gemini** (no eran
    visibles sin probar — mismo patrón que en Fase 1):
    - `agents/tools/fs_tools.write_run_artifact` tenía `data: Any` como tipo del
      parámetro. `langchain-google-genai` convierte el schema de cada tool al formato
      de function-calling de Gemini, y un `Any` sin estructura generaba
      `properties.data: None` — `pydantic_core.ValidationError` apenas el orquestador
      arma su primer mensaje (rompía el grafo entero antes de llamar a ningún
      subagente, ni siquiera dependía del mensaje de entrada). Cambiado a
      `data: dict[str, Any]` (coincide con todos los call-sites reales, que ya pasaban
      dicts) — error desaparece.
    - El default `agent_model = "google_genai:gemini-2.0-flash"` daba
      `429 RESOURCE_EXHAUSTED` con `limit: 0` en el free tier para ESTE tipo de key
      (generada desde Google Cloud Console, no desde el botón "Get API key" de AI
      Studio — formato `AQ....` en vez de `AIzaSy...`). Probado directo contra la API
      (`generateContent`): `gemini-2.0-flash` y `gemini-2.0-flash-lite-001` → 429 (cuota
      0); `gemini-2.5-flash` → 200 OK. Cambiado el default a
      `google_genai:gemini-2.5-flash` en `mia_common/settings.py`. Si en algún momento
      se usa una key vieja tipo AI Studio (`AIzaSy...`), puede que vuelva a andar
      `gemini-2.0-flash` y convenga revisar cuál conviene por costo/latencia — esto no
      es una limitación de Gemini en general, es especifico de cuota por tipo de key.
    - **`mia_agent` estaba roto para CUALQUIER target, nunca se había ejercitado**:
      `run_decop_tool`/`run_simia_tool`/`run_dualtest_tool` tenían `client: TargetClient`
      expuesto directo como parámetro de la tool — un objeto Python real, no
      JSON-serializable. `convert_to_genai_function_declarations` rompe con
      `PydanticInvalidForJsonSchema` apenas el orquestador intenta delegar a
      `mia_agent`. `functools.partial` NO sirve para esconderlo (`inspect.signature`
      sigue mostrando el parámetro con el objeto bindeado como default, y
      `typing.get_type_hints` directamente rechaza un `partial`) — la solución real es
      un closure (`def` anidado) que capture `client`/`reference_model_name` sin
      exponerlos en su propia firma. `agents/subagents/mia_agent.py` ahora expone
      `build_mia_subagent(client, reference_model_name)` en vez de un dict estático;
      `agents/orchestrator.py` construye el `TargetClient` elegido
      (`mia_common.target_client.resolve_target_client`, nuevo) antes de armar ese
      subagent — por eso `mia_agent` ya no vive en `STATIC_SUBAGENTS`
      (`agents/subagents/__init__.py`). Validado offline (sin gastar API, con el
      `TargetClient` real de Groq bindeado): las 3 tools convierten bien al schema de
      Gemini.
  - **Selector de modelo target por run (Fase 4)**: la webapp deja elegir
    `target_provider` (groq/openai/anthropic/google, botones que se deshabilitan solos
    si falta la key en `.env`) antes de arrancar un run —
    `webapp/run_manager.TARGET_MODEL_CHOICES` mapea cada proveedor a un model_name
    default razonable. Como consecuencia, `build_orchestrator()` y el orquestador
    global compartido de `webapp/run_manager.py` ya no existen — **cada run construye
    su propio orquestador** (con su propio `mia_agent` bindeado al target elegido),
    porque `mia_agent` necesita un `TargetClient` concreto por run, no uno fijo para
    todo el proceso.
  - `GROQ_API_KEY`/`GROQ_API_KEYS`, `TAVILY_API_KEY` y `HF_TOKEN` ya están configuradas
    (esta sesión + la de Isabel). Con eso, **se consiguió la primera búsqueda 100% real
    de punta a punta** (Tavily real + descarga real + Gemini real, ver más abajo).
  - **Trabajo de Isabel (`castaisa`) en esta branch, commits `2d58e0c` y `7de17de`**:
    - `fetch_url` pasó de `requests` a `curl` via `subprocess` — Gutenberg bloquea el
      fingerprint TLS (JA3) de `requests`/`urllib3` aunque el User-Agent sea de
      navegador real; `curl` sí pasa. Encontró además que `--fail` combinado con
      HTTP/2 hace que un error HTTP devuelva exit 56 (error de red) en vez de exit 22
      (error HTTP claro) — se fuerza `--http1.1`. Reintentos con backoff,
      `fetch_url` ya no excepciona (devuelve `{"url","error"}`) para que una fuente
      caída no tire abajo el run.
    - `agent_model` default cambiado a `google_vertexai:gemini-2.5-pro` —
      `gemini-2.5-flash` generaba `MALFORMED_FUNCTION_CALL` intermitente en
      `write_todos` (deepagents) que frenaba el orquestador EN SILENCIO (sin
      excepción visible). Vertex AI además factura contra créditos de Cloud, no contra
      el free tier de 20 req/día de AI Studio — pero necesita
      `gcloud auth application-default login` (ADC) configurado por máquina, no viaja
      por `.env`. Sin ADC en esta máquina, las pruebas de esta sesión siguen usando
      `agent_model="google_genai:gemini-2.5-flash"` (o `gemini-flash-latest`) como
      override explícito.
    - Checkpointer cambiado de `InMemorySaver` a `SqliteSaver` persistente
      (`runs/_checkpoints.sqlite`) en `scripts/run_pipeline_agentic.py` — un crash a
      mitad de run se puede retomar con `--run-id <mismo run_id>` sin repetir las
      etapas ya completadas. Confirmado por Isabel con una prueba real entre procesos
      distintos.
    - **Bug de integridad real, encontrado por Isabel y arreglado por mí en esta
      sesión**: `bibliography_agent` descarga texto real pero esa conversación (y el
      texto crudo) desaparece en cuanto termina su subtarea — `curator_agent` no tenía
      forma de recuperarlo, y terminaba evaluando en base a lo que el modelo
      recordaba de su propio entrenamiento (literalmente inventó estar evaluando "El
      Aleph" de Borges en vez de los libros de Dickens que se habían bajado de
      verdad). Fix de Isabel: `bibliography_agent` ahora guarda cada texto descargado
      con `write_run_artifact(run_id, "bibliography", f"text_{document_id}", {...})`
      apenas lo baja, y `curator_agent` está obligado a recuperarlo con
      `read_run_artifact` antes de evaluar — nunca en base a memoria propia.
    - **Bug MÁS serio que destapó ese mismo fix, sin arreglar hasta esta sesión**: si
      `bibliography_agent` terminaba su subtarea SIN haber llamado a
      `propose_candidate_texts` (ej. no encontró nada para ese autor), el
      ORQUESTADOR podía fabricar él mismo una lista de candidatos plausible y
      guardarla con `write_run_artifact` como si un humano la hubiera aprobado — sin
      que ninguna pausa de revisión real hubiera ocurrido. El fix de `curator_agent`
      de arriba contenía el daño (no encontraba el texto fabricado y fallaba en vez de
      evaluarlo), pero el dato fabricado ya había pasado la etapa de bibliografía como
      "aprobado". **Fix real**: `propose_candidate_texts(run_id, candidates)`
      ahora persiste `candidates.json` ELLA MISMA, adentro del tool que el
      `interrupt_on` protege — el orquestador ya no llama a `write_run_artifact` para
      esto, ni puede. Como ese código solo corre como consecuencia de que un humano
      resuelva una pausa real (deepagents reemplaza los args por la versión
      aprobada/editada antes de ejecutar), el artifact en disco pasa a ser una prueba
      MECÁNICA de aprobación humana genuina, no algo que dependa del juicio del
      orquestador. Defensa en profundidad adicional: `curator_agent` ahora lee
      `candidates.json` él mismo (`read_run_artifact`) en vez de confiar en la lista
      que el orquestador le pasa en el mensaje de la tarea — así un eventual error de
      fabricación del orquestador ya no se le puede contagiar.
    - **Validado con el primer run 100% real de la historia de este proyecto**: autor
      "Edgar Allan Poe", Tavily real, Groq real (`llama-3.1-8b-instant`),
      `agent_model="google_genai:gemini-flash-latest"` (override, ver nota de ADC
      arriba). El interrupt llegó con un candidato genuino ("The Fall of the House of
      Usher", URL real de Gutenberg), se aprobó, y quedaron persistidos
      `runs/real_e2e_test_1/bibliography/candidates.json` Y
      `text_the_fall_of_the_house_of_usher.json` (39481 caracteres, texto real de
      Gutenberg, verificado). El run se cortó ahí por agotar la cuota de Gemini (20
      req/día, **es por modelo, no por key/proyecto** — `gemini-2.5-flash`,
      `gemini-2.5-pro` y `gemini-flash-latest` cada uno tiene su propio contador) antes
      de llegar a `curator_agent`. **Pendiente real**: re-correr el mismo test cuando
      resetee alguna cuota (parece resetear a medianoche UTC, 21hs Argentina) o se
      configure ADC de Vertex, para validar curator_agent → SAGE → mia_agent → ensemble
      con datos reales de punta a punta.
    - **Segundo intento (reusando el texto de Poe ya descargado, sin repetir Tavily)**:
      delegar directo a `curator_agent` con `gemini-flash-lite-latest` pegó dos veces
      seguidas contra `GenerateContentInputTokensPerModelPerMinute-FreeTier` (un límite
      POR MINUTO, no diario) — el texto completo (~40K caracteres) más el system prompt
      de deepagents se acerca al tope de tokens/minuto del free tier en una sola
      llamada. Esperar ~75s entre intentos no alcanzó para que se liberara. No vale la
      pena seguir reintentando a ciegas contra el free tier — el plan es retomar esto
      con una key con billing habilitado (sin cap), que el usuario está gestionando.
    - **Resuelto: proyecto nuevo de Vertex AI con crédito estudiantil, sin el cap del
      free tier.** Setup real (con varios intentos fallidos por nombres de roles
      confusos en la consola en español, ver `learnings.jsonl` para el detalle
      paso a paso): service account nuevo en un proyecto propio (`nlpagentes`),
      `GOOGLE_APPLICATION_CREDENTIALS` apuntando al JSON descargado (no hace falta
      `gcloud auth login`, las credenciales de service account son no-interactivas),
      `aiplatform.googleapis.com` habilitada, rol `Editor` otorgado al service account
      (el rol específico "Vertex AI User" no aparecía de forma confiable en el buscador
      de la consola — `Editor` es más permiso del necesario pero aceptable en un
      proyecto personal de prueba).
    - **Nuevo: `mia_common/spend_guard.py`** — tope duro de gasto en USD
      (`settings.agent_model_spend_cap_usd`, default $4.50, deliberadamente por debajo
      de los $5 que pidió el usuario por margen de estimación) para `agent_model`
      cuando es un modelo de Vertex AI (factura por uso real, a diferencia del free
      tier de AI Studio). `agents/orchestrator.py::_resolve_agent_model` envuelve el
      chat model con un `VertexSpendGuardCallback` (trackeado en disco,
      `runs/_spend_<project>.json`, sobrevive entre procesos) que frena ANTES de cada
      llamada si se superaría el límite, y ajusta al costo real despues con
      `usage_metadata`.
    - **Bug real encontrado validando el spend guard con una llamada real**: Gemini 2.5
      (modelos "thinking") factura los tokens de "reasoning" igual que el output, pero
      `usage_metadata.output_tokens` NO los incluye — medido en vivo, una pregunta
      trivial usó `output_tokens=3` pero `reasoning=6749` (`total_tokens=6766`). Contar
      solo `output_tokens` subestimaba el costo real en ~1260x. Arreglado usando
      `total_tokens - input_tokens` como output facturable (no depende de como cada
      provider nombre el sub-campo de reasoning). Validado offline reproduciendo los
      números reales de esa llamada.
    - **Primera llamada real a Vertex AI confirmada**: "¿Cuál es la capital de
      Argentina?" → "Buenos Aires", costo real ~$0.081 (con el fix de reasoning
      tokens).
    - **Bug real más, encontrado armando esto**: pasarle al chat model resuelto
      `.with_config({"callbacks": [...]})` para adjuntar el `VertexSpendGuardCallback`
      rompe `create_deep_agent` — envuelve el modelo en un `RunnableBinding`, que
      `deepagents._models.resolve_model()` no reconoce como `isinstance(_,
      BaseChatModel)` y lo trata como si fuera un string de nuevo
      (`AttributeError: '<Modelo>' object has no attribute 'count'`). Fix: mutar
      `model.callbacks = [callback]` directo en la instancia (`callbacks` es un campo
      real de `BaseChatModel`) en vez de `.with_config()`.
    - **Primer test real de `curator_agent` → SAGE → `mia_agent` con Vertex AI
      (`gemini-2.5-pro`)**: llegó mucho más lejos que con el free tier — autoría real
      (Poe, confianza 1.0, razonamiento correcto) + 8 chunks con voice score real
      (0.9 de distintividad, identificó bien el estilo gótico). `flow_checker_agent`
      detectó de verdad una inconsistencia real en el naming de los artifacts de voz
      (`voice_<chunk_id>.json` vs `voice_<document_id>_<chunk_id>.json`) y la marcó
      `severity="warning"`, `recommended_action="continue"` — juicio correcto, no era
      grave. **Pero el run se cortó ahí**, sin delegar a `sage_qa_agent`/`mia_agent`, y
      el mensaje final vino vacío — no investigado a fondo todavía (corta un test más
      caro). **Costo real: ~$1.11** solo por esta etapa (curación de 1 texto, 8
      chunks) — bastante más de lo esperado, ojo con el presupuesto para corridas
      reales completas.
    - **`gemini-2.5-flash` en Vertex AI: mismo bug que ya documentó Isabel en AI
      Studio, confirmado que NO es específico de AI Studio.** Leyó bien la skill (3
      `read_file` correctos) y el turno siguiente vino con `content` vacío y SIN tool
      calls — el grafo terminó sin delegar nunca a `curator_agent`, sin excepción ni
      error visible. Costó casi nada (~$0.007) pero no sirvió para nada. **No usar
      `gemini-2.5-flash` como `agent_model` bajo ninguna circunstancia** (ni AI Studio
      ni Vertex) — `gemini-2.5-pro` sigue siendo el único modelo Gemini validado para
      este rol, a pesar de costar bastante más.
    - **Segundo intento real con `gemini-2.5-pro` (run `real_e2e_test_2`, mismo texto
      de Poe)**: esta vez SÍ llegó a delegar a `sage_qa_agent` — se vio
      `[SPS] Loading Gemma model on cpu...` en los logs. Pero `google/gemma-2b`
      (modelo gated que usa SAGE para el Semantic Persistence Score) tardó **25+
      minutos** en descargarse la primera vez (no estaba cacheado en esta máquina) y
      la tarea en background se mató por superar el tiempo disponible — **no fue un
      problema de plata ni del código del agente**. El modelo quedó cacheado
      (`~/.cache/huggingface/hub`, 4.7GB) para la próxima vez. Costo real de esta
      corrida (curación de nuevo, ya que cada proceso es una conversación nueva sin
      memoria): ~$1.33.
    - **Revisando el código antes de gastar más, encontré que el mismo patrón de bug
      se repetía un escalón más adelante**: `sage_qa_agent` y `mia_agent` esperaban
      recibir el texto del chunk (y `mia_agent` además los paraphrase candidates de
      SAGE) directo en el mensaje de la tarea del orquestador — pero el resumen final
      que un subagent le devuelve a su padre no incluye el contenido completo (a
      propósito, para no gastar tokens de más), así que el orquestador nunca iba a
      tener ese contenido para relayar. **Arreglado con el mismo patrón ya validado**
      (disco, no conversación): `curator_agent` ahora persiste cada chunk verbatim en
      `runs/<run_id>/curation/chunk_<chunk_id>`, `sage_qa_agent` lee eso y persiste
      sus paraphrase candidates en `runs/<run_id>/sage/paraphrase_<chunk_id>`,
      `mia_agent` lee ambos. De paso, se fijó `chunk_id = f"{document_id}_{i}"` como
      única convención (arregla la inconsistencia de naming que `flow_checker_agent`
      ya había detectado solo).
    - **Tercer intento (run limpio, fix de persistencia puesto, modelo de SAGE ya
      cacheado): validado en vivo, y el spend guard frenó exactamente como debía.**
      `curator_agent` curó los **51 chunks** de "The Fall of the House of Usher" con
      `chunk_id` consistente (`the_fall_of_the_house_of_usher_0` .. `_50`) y persistió
      el texto verbatim real de cada uno en `runs/real_e2e_test_2/curation/chunk_*`
      (confirmado leyendo el contenido — es el texto real de Poe) — el fix de
      persistencia entre subagentes funciona. El run **se frenó solo** cuando el costo
      acumulado llegó a $4.0844 y la siguiente llamada estimada ($0.443) hubiera
      superado el tope de $4.50: `SpendCapExceededError`, exactamente el
      comportamiento pedido por el usuario, no un fallo. No llegó a `sage_qa_agent`
      para estos 51 chunks dentro del presupuesto.
    - **Hallazgo de costo real importante**: curar 51 chunks (autoría una vez +
      voz **51 veces**, una por chunk) con `gemini-2.5-pro` consume la enorme mayoría
      de un presupuesto de $4.50 — antes de llegar siquiera a parafrasear con SAGE o
      puntuar con MIA. El chunking a ~128 tokens/chunk (el mismo default que usa la
      Fase 1 manual) genera demasiados chunks para que la curación agéntica con un
      modelo "thinking" sea barata.
    - **Arreglado (propuesta del usuario)**: `curator_agent` ya NO juzga la voz de
      todos los chunks de un documento. Ahora pide un lote inicial chico
      (`settings.curator_initial_batch_size`, default 6) y, si no llegó al objetivo
      (`settings.curator_target_chunks_per_text`, default 5 "keep" por documento),
      pide chunks adicionales **de a uno** (no en bloque) hasta llegar al objetivo o
      agotar el documento — paga solo por lo que efectivamente termina usando. Esto es
      distinto de `chunks_per_text` (esa es la muestra de chunks CRUDOS sin curar que
      usa la Fase 1 manual, antes de cualquier juicio LLM). Debería reducir ~80-90% el
      costo de la etapa de curación (6-10 juicios de voz en vez de 51 para un texto de
      este tamaño) — **validado que compila, todavía no probado en vivo** (sin
      presupuesto al momento de escribir esto, ver abajo).
    - **Presupuesto final de esta sesión: $4.08 de $4.50** (tope configurado
      deliberadamente por debajo de los $5 que pidió el usuario — el margen de
      seguridad cumplió su función). El guard de gasto (`mia_common/spend_guard.py`)
      quedó validado de punta a punta: reserva conservadora antes de cada llamada,
      ajuste al costo real después, y corte duro cuando corresponde.
    - **Auditoría completa pedida por el usuario antes de gastar más** (sin tocar
      ninguna API, todo offline): encontró 2 bugs más, ningún costo real.
      1. `read_run_artifact` lanza `FileNotFoundError` sin atrapar cuando el archivo
         no existe — por default LangGraph NO lo convierte en mensaje de error legible
         por el agente (`_default_handle_tool_errors` solo atrapa
         `ToolInvocationError`, cualquier otra excepción se relanza y frena el run
         ENTERO). Esto iba a explotar la primera vez que un chunk legítimamente no
         tuviera paraphrase de SAGE (descartado por QA) y `mia_agent` intentara leerlo
         — exactamente el tipo de caso que SÍ iba a pasar en un run real. Fix aplicado
         en toda la cadena (`curator_agent`, `sage_qa_agent`, `mia_agent`): siempre
         `list_run_artifacts()` primero para confirmar que el archivo está listado
         antes de leerlo, nunca asumir y atrapar la excepción después.
      2. `mia_agent` nunca persistía sus 3 resultados crudos a disco — el orquestador
         iba a necesitar extraerlos de su resumen de texto para llamar a
         `combine_scores`, el mismo patrón frágil que ya se arregló para
         bibliografía/curación/SAGE. Fix: `mia_agent` ahora persiste
         `runs/<run_id>/mia_scores/<chunk_id>` con los 3 resultados crudos; el
         orquestador lee de ahí, no del resumen.
      3. Reforzado en código (no solo en el prompt) el límite de chunks curados por
         documento: `record_voice_score` ahora cuenta cuántos "keep" hay para ese
         `document_id` y devuelve `decision="target_reached"` sin registrar nada si ya
         se llegó al objetivo — verificado offline (10 chunks simulados → exactamente
         5 "keep" + 5 "target_reached", sin gastar nada).
      - Todo lo de arriba: validado que compila y que el schema de TODAS las tools de
        TODOS los subagentes convierte bien a Gemini. **No probado en vivo todavía**
        (sin presupuesto) — pendiente para la próxima sesión (necesita más crédito o
        subir `settings.agent_model_spend_cap_usd`): completar
        `sage_qa_agent` → `mia_agent` → `combine_scores` → `aggregate_text_scores` con
        datos 100% reales.
    - **Dos ajustes a pedido del usuario, antes de la próxima corrida real**:
      - SAGE genera 4 candidatos de paraphrase por segmento y se queda con los 3 de
        mejor `final_score` (antes generaba exactamente 3, sin margen) —
        `settings.sage_n_candidates_generated`/`sage_n_candidates_kept`. `SAGE/sage.py`
        sigue sin depender de `mia_common` (los valores se pasan como argumentos desde
        `agents/tools/sage_tools.py`, respetando la separación investigación/ingeniería
        del proyecto). Validado offline (lógica de orden/filtro, sin cargar los
        modelos pesados de SAGE).
      - **SiMIA desactivado temporalmente** (`settings.simia_enabled = False`,
        default): el pipeline agéntico corre solo DE-COP + DUALTEST hasta validar el
        resto de punta a punta — SiMIA "todavía no está terminado". Implementado
        quitando `run_simia_tool` de la lista de tools de `mia_agent` cuando está
        desactivado (no alcanza con pedírselo solo al prompt). `combine_scores` ya
        manejaba `simia_raw=None` sin cambios. No afecta `scripts/run_pipeline_manual.py`
        (Fase 1), que sigue corriendo los 3 métodos siempre. Reactivar: una sola
        variable (`simia_enabled=True`), sin tocar más código.
    - **Bug real grave, encontrado en un run de la webapp con autores reales (Jane
      Austen)**: `bibliography_agent` encontró y descargó **novelas completas**
      (`Pride and Prejudice`, `Sense and Sensibility`, ~700.000 caracteres cada una —
      no fragmentos). `curator_agent` necesitaba mandarle ese texto entero como
      argumento de tool call a Gemini para chunkearlo — exactamente el riesgo que el
      README ya advertía ("va a necesitar chunkear por capítulo/página, no el libro
      entero junto"). El run costó **\$2.81** y nunca terminó de chunkear — los dos
      documentos pasaron autoría (`decision=keep`) pero cero chunks se generaron.
      `flow_checker_agent` no lo detectó ("no anomalies found"). **Fix de raíz**:
      nueva tool `agents/tools/chunk_tools.clean_and_chunk_document(run_id,
      document_id)` que lee el texto del disco, chunkea, y persiste CADA chunk
      server-side — el texto completo del documento ya NUNCA pasa por el contexto del
      LLM (ni como argumento ni como respuesta). `curator_agent` ahora lee cada chunk
      individualmente con `read_run_artifact` antes de juzgarlo, mismo patrón que
      `sage_qa_agent`/`mia_agent`. **Validado offline con el texto REAL de "Pride and
      Prejudice"** (gratis, sin LLM): 3.6 segundos para chunkear el libro completo en
      139 chunks — el chunking en sí nunca fue el problema de performance (la
      preocupación de ~5 min documentada antes parece ya no aplicar en este entorno);
      el problema 100% era el costo de mandarle el texto completo a un LLM. Reforzado
      `flow_checker_agent` para detectar este caso específico (documento con
      `authorship` keep pero cero chunks) como `severity=error`/`retry_stage`.
      **Todavía no validado en vivo** (el bug se encontró justo después del run que lo
      reveló, no se volvió a intentar).

- 🟡 **Fase 3 — skill persistente.** Construida y validada (sin necesitar
  Groq/Tavily). `agents/skills/pipeline-learnings/` tiene `SKILL.md` (instrucciones de
  cuándo leer/escribir), `learnings.jsonl` (log apendable, seedeado con los bugs reales
  de Fase 1/2 — SiMIA, DUALTEST, Groq-como-agent_model, etc.) y
  `calibration_history.csv` (seedeado con los números de separación member/non-member
  ya documentados en Fase 1).
  - `agents/orchestrator.py::build_orchestrator` ahora pasa
    `backend=FilesystemBackend(root_dir=settings.skill_dir.parent, virtual_mode=True)` +
    `skills=["/"]` a `create_deep_agent` — el orquestador (y, como efecto del backend
    compartido, los subagentes) ganan `read_file`/`ls` ahí, scopeados SOLO a
    `agents/skills/` (no al resto del repo — decisión deliberada: `FilesystemBackend`
    con un root amplio expondría `.env` y el resto del código a un agente con tools de
    red, ver el warning de seguridad en la documentación de deepagents).
  - `agents/tools/skill_tools.py`: `record_learning`/`record_calibration`, dos tools
    nuevas para que el orquestador escriba al final de cada run (paso 8 nuevo en
    `ORCHESTRATOR_SYSTEM_PROMPT`). Apendean, nunca reescriben.
  - **Validado en vivo** (un solo mensaje, sin delegar a ningún subagente, para no
    gastar Tavily/Groq): le pedí al orquestador que dijera qué modelo NO usar como
    `agent_model` según lo aprendido — leyó `/pipeline-learnings/SKILL.md` con
    `read_file` por su cuenta y contestó correctamente citando el bug real de Groq.
  - **Límite deliberado**: la skill registra observaciones de calibración pero NO
    ajusta sola los thresholds de `mia_common/settings.py` — eso sigue siendo decisión
    humana (mismo principio que los LLM-judges de curación: escalar, no autoaplicar).
  - **Pendiente real**: todavía no se validó `record_learning`/`record_calibration`
    siendo llamados por el agente DENTRO de un run completo de punta a punta (necesita
    `GROQ_API_KEY`/`TAVILY_API_KEY`, ver Fase 2) — solo se confirmó que el agente lee la
    skill correctamente, no que la escriba bien al cierre de un run real.

- 🟡 **Fase 4 — interfaz web.** Construida y validada de punta a punta (incluyendo el
  ciclo completo de human-in-the-loop por HTTP) con `tavily_search`/`fetch_url`
  mockeados — el resto del flujo (Gemini real, deepagents real) no está mockeado.
  - `webapp/main.py`: rutas FastAPI server-rendered (Jinja2) — `GET /` (formulario con
    el nombre del autor, eso es todo lo que pide la idea original), `POST /runs`
    (arranca el run), `GET /runs/{run_id}` (estado + revisión humana + artifacts),
    `POST /runs/{run_id}/decide` (aprobar tal cual / editar JSON / rechazar),
    `GET /runs/{run_id}/stream` (SSE de una sola línea de JS para refrescar la
    pantalla sola — sin SPA, sin polling manual en el cliente).
  - `webapp/run_manager.py`: el puente entre el orquestador (síncrono, bloqueante,
    `Command(resume=...)`) y FastAPI (async). Un `RunHandle` por run con un thread de
    background — cuando el orquestador pega un `__interrupt__`, el thread se duerme en
    un `threading.Event` hasta que `POST /runs/{run_id}/decide` lo despierta con la
    decisión. Mismo patrón que el loop de `scripts/run_pipeline_agentic.py`, pero la
    "pausa por stdin" de ese script ahora es una pantalla.
  - Faltaba `python-multipart` en `requirements.txt` (necesario para que FastAPI lea
    `Form(...)`, no estaba listado en la Fase 4 original) — agregado.
  - **Validado con un test que mockea SOLO `tavily_search`/`fetch_url`** (no el
    orquestador ni deepagents) manejando requests HTTP reales contra la app
    (`TestClient`, sin mockear nada de FastAPI/Jinja2): el ciclo completo
    `POST /runs` → pantalla de revisión humana con los candidatos reales que propuso
    bibliography_agent → `POST .../decide` (approve) → resume → resultado final,
    funcionó de punta a punta.
  - **Hallazgo real en el camino** (no un bug introducido por la webapp, ya existía en
    el orquestador de Fase 2, simplemente nadie lo había corrido lo bastante lejos
    para verlo): el orquestador delegaba a bibliography_agent y pausaba para revisión
    humana, pero nunca guardaba la lista de candidatos ya aprobada en
    `runs/<run_id>/` antes de pasar a `curator_agent`. `flow_checker_agent` (que ya
    existía, Fase 2) detectó correctamente que no había nada guardado para esa etapa y
    recomendó `escalate_to_human` — el orquestador frenó como tenía que frenar. Esto es
    justo lo que Fase 2 dejaba pendiente de validar ("que flow_checker_agent realmente
    frena/escala cuando corresponde"): confirmado, frena bien. Arreglado agregando una
    instrucción explícita en el paso 1 de `ORCHESTRATOR_SYSTEM_PROMPT`: guardar los
    candidatos aprobados con `write_run_artifact` antes de la etapa 2.
  - **No se pudo re-validar el fix de punta a punta**: al reintentar el mismo test
    después del fix, se agotó la cuota diaria gratuita de `gemini-2.5-flash`
    (**20 requests/día** en el free tier de esta key de Cloud Console — bastante más
    chico de lo que parecía con `gemini-2.0-flash` dando directamente cuota 0). Mismo
    patrón que el `DailyCapError` de Groq, ver `learnings.jsonl`. **Pendiente real**:
    re-correr el mismo test (`isolation`/mock de Tavily, real Gemini) cuando resetee la
    cuota o se agregue billing/otra key, para confirmar que el fix realmente hace que
    `flow_checker_agent` recomiende `continue` y el run avance a `curator_agent`.
  - **Rediseño visual** (a pedido del usuario, la v1 era deliberadamente mínima):
    paleta bordó/dorado, tipografía Playfair Display + Inter (Google Fonts), cards con
    sombra, botones primario/secundario/peligro diferenciados para
    aprobar/editar/rechazar, badges de estado con ícono. `webapp/templates/base.html`
    es el layout compartido (header/footer) — `index.html`/`run.html` solo llenan
    `{% block content %}`, así que si algo del header/footer no se ve bien hay que
    mirar `base.html`, no las otras dos. Sin venv con Playwright instalado en esta
    máquina (poco espacio en disco) — los cambios de CSS se validaron por estructura de
    HTML/clases, no con captura de pantalla real; usar `/run` o pedirle al usuario que
    confirme visualmente.
  - **Selector de modelo target** (`target_provider`, botones en `index.html`,
    deshabilitados solos si falta la key): ver el detalle completo y el bug real que
    encontró (`mia_agent` roto para cualquier target) en la sección de Fase 2 — está
    documentado ahí porque el bug vivía en `agents/subagents/mia_agent.py`, no en la
    webapp en sí.
  - No probado todavía con `TAVILY_API_KEY` real ni con autores/bibliografía reales —
    `GROQ_API_KEY` ya está configurada (ver Fase 2), pero sigue bloqueado por la cuota
    de Gemini como `agent_model` (no llega a delegar a `bibliography_agent`).

## Cómo correr lo que existe hoy

```bash
conda create -n mia-agentes python=3.12   # el python del sistema puede ser incompatible
conda activate mia-agentes
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

**Interfaz web** (Fase 4) sobre el mismo orquestador de Fase 2/3:

```bash
uvicorn webapp.main:app --reload
```

Abrir `http://127.0.0.1:8000/`. Necesita los mismos requisitos que
`scripts/run_pipeline_agentic.py` (`GOOGLE_API_KEY`/`ANTHROPIC_API_KEY` para el
`agent_model`, `GROQ_API_KEY`/`TAVILY_API_KEY` para un run real de punta a punta) — sin
ellas el run arranca igual y la pantalla muestra el error claro en vez de fallar en
silencio. El estado de los runs vive en memoria del proceso de uvicorn (no sobrevive un
reinicio del server), igual que `InMemorySaver` en el script de CLI.

**Para ampliar el dataset** con más libros member/non-member, ver
`scripts/expand_dataset.py` (agrega entradas a `NEW_SOURCES`, corre, regenera
`dataset_len128.csv`). Los non-member deben ser del mismo TIPO de texto que los member
(narrativo, no resúmenes/listas) para que la comparación tenga sentido.

## Limitaciones conocidas / próximos riesgos a resolver

- **Costo de la curación agéntica con un modelo "thinking" — ARREGLADO, no probado en
  vivo todavía**: con chunks de ~128 tokens, un texto narrativo normal genera 50+
  chunks. Medido en vivo: curar los 51 chunks de un cuento (un `record_voice_score`
  por chunk, sin límite) consumió la mayor parte de un presupuesto de $4.50 de Vertex
  AI sin llegar a SAGE/MIA. Fix (`agents/subagents/curator_agent.py`,
  `mia_common.settings.curator_target_chunks_per_text`/`curator_initial_batch_size`):
  `curator_agent` ahora pide un lote chico y completa de a un chunk extra hasta llegar
  al objetivo, en vez de juzgar el documento entero. **Pendiente**: validar cuánto
  ahorra esto de verdad en un run real (se espera ~80-90% menos llamadas en la etapa
  de curación) — no se pudo probar en esta sesión por falta de presupuesto.
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
