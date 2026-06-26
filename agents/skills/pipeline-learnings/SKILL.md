---
name: pipeline-learnings
description: Historial de bugs, calibraciones y decisiones de diseno del pipeline de Membership Inference Attack (SAGE, DE-COP, SiMIA, DUALTEST, bibliography/curation). Usar SIEMPRE al arrancar un run nuevo (antes de delegar a bibliography_agent) para no repetir errores ya conocidos, y al final de cada run (exito o falla parcial) para dejar registrado lo que se aprendio.
metadata:
  phase: "3"
---

# Pipeline learnings — orquestador MIA

Esta skill es la memoria entre runs del orquestador. No reemplaza a `runs/<run_id>/`
(eso es el detalle de UN run); esto es lo que se generaliza ENTRE runs.

## Al arrancar un run

1. Leer `learnings.jsonl` completo (es un JSONL chico, una linea = un aprendizaje con
   `{timestamp, run_id, stage, severity, learning}`). Prestar atencion especial a
   entradas `severity="critical"` — son cosas que rompieron un run entero antes.
2. Leer `calibration_history.csv` (`timestamp,run_id,method,metric,value,notes`) para
   tener una nocion de los rangos de separacion member/non-member ya observados, y no
   sorprenderse si un chunk individual cae fuera de rango (hay superposicion conocida a
   nivel texto individual, ver abajo).
3. Si algo en el run actual contradice fuerte una entrada de `learnings.jsonl` (ej. un
   metodo que antes daba senal constante ahora varia, o viceversa), tratalo como
   sospechoso y delega a flow_checker_agent en vez de asumir que esta bien.

## Al terminar un run (exito o falla parcial)

Llamar `record_learning(run_id, stage, learning, severity)` por cada cosa nueva que
valga la pena que un run futuro sepa (un bug, un patron de falla recurrente, un ajuste
que funciono). No hace falta forzar una entrada si no se aprendio nada nuevo.

Llamar `record_calibration(run_id, method, metric, value, notes)` con los numeros
agregados del run (ej. `aggregate_text_scores` por autor, separacion promedio
member/non-member si el run tiene ground truth) para que la serie historica en
`calibration_history.csv` siga creciendo.

**Importante**: estos tools APENDEAN — nunca reescriben ni borran lo que ya esta. Si un
aprendizaje pasado quedo obsoleto (ej. un bug ya arreglado), agregar una entrada NUEVA
que lo aclare en vez de editar la vieja (`learnings.jsonl` es un log, no un estado).

**Limite deliberado de esta skill (Fase 3)**: registra observaciones, pero NO ajusta
sola los thresholds de `mia_common/settings.py` (`authorship_min_confidence`,
`sage_min_sps`, etc.). Decidir si un threshold deberia cambiar en base a la calibracion
acumulada sigue siendo una decision humana — escalar la sugerencia, no aplicarla.

## Aprendizajes ya consolidados (no se repiten en learnings.jsonl, son estructurales)

- **`agent_model` (el LLM que razona en el orquestador) tiene que ser Gemini o
  Anthropic, nunca un chat model de Groq** — Groq (`llama-3.3-70b-versatile`) genera
  argumentos mal formados en la tool `task` que deepagents usa para delegar a
  subagentes. Esto es independiente de que Groq funcione perfecto como TARGET del MIA
  (son roles distintos, no confundir).
- **DUALTEST sigue siendo un proxy sin calibrar** en `agents/ensemble/combine.py` — el
  bug de escala (colapsaba a constante) esta arreglado, pero el protocolo de
  calibracion de dos etapas (`DUALTEST/calibration.py`) nunca se corrio. No tratar su
  score como tan confiable como DE-COP/SiMIA todavia.
- **SiMIA es una senal debil incluso arreglada** (AUC~0.51 medido en BookTection segun
  el README general) — no descartar un texto SOLO por su score de SiMIA.
- **Los dos LLM-judges de curacion (autoria, voz caracteristica) no tienen benchmark
  etiquetado** — los casos borderline (ver `authorship_review_band` en settings.py)
  tienen que ir a revision humana, no resolverse en silencio aunque el agente este
  "seguro".
- **Toda llamada a una API externa tiene que pasar por `mia_common.target_client`**
  (cachea automaticamente en `runs/_api_cache/`) — nunca llamar al SDK de un proveedor
  directo desde un tool nuevo, por mas que parezca mas simple para un caso puntual.
