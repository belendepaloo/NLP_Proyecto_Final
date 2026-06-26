"""
fs_tools.py — convencion de artifacts en disco por run, independiente de cualquier
filesystem virtual de deepagents (ver el plan: "debugging nunca depende de inspeccionar
internals de LangGraph"). Cada etapa del pipeline escribe aca lo que produjo.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from mia_common.settings import settings


def run_dir(run_id: str) -> Path:
    return settings.runs_dir / run_id


def stage_dir(run_id: str, stage: str) -> Path:
    d = run_dir(run_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _artifact_path(run_id: str, stage: str, name: str) -> Path:
    # El LLM a veces pasa `name` con ".json" ya incluido (ej. "candidates.json") aunque
    # el docstring pida el nombre sin extension -- normalizar en vez de fallar con un
    # path tipo "candidates.json.json", visto en vivo con gemini-2.5-pro via Vertex.
    name = name.removesuffix(".json")
    # Tools como record_authorship_verdict usan un id que el agente eligio (ej.
    # document_id) como parte de `name` -- visto en vivo que el agente paso un path
    # temporal completo ("/tmp/libro.txt") en vez de un identificador simple, lo que
    # generaba subdirectorios/paths invalidos. Sanitizar en vez de confiar en que el
    # agente siempre pase algo path-safe.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return stage_dir(run_id, stage) / f"{name}.json"


def write_run_artifact(run_id: str, stage: str, name: str, data: dict[str, Any]) -> str:
    """Escribe `data` (un dict JSON-serializable) como
    runs/<run_id>/<stage>/<name>.json (pasar `name` SIN extension). Devuelve el path
    escrito (como string)."""
    path = _artifact_path(run_id, stage, name)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    return str(path)


def read_run_artifact(run_id: str, stage: str, name: str) -> Any:
    """Lee de vuelta el artifact escrito por write_run_artifact en
    runs/<run_id>/<stage>/<name>.json (pasar `name` SIN extension)."""
    path = _artifact_path(run_id, stage, name)
    return json.loads(path.read_text())


def list_run_artifacts(run_id: str) -> dict[str, list[str]]:
    """Tree listing de runs/<run_id>/ -- usado por la pantalla de drill-down (Fase 4)
    y por flow_checker_agent para inspeccionar que se produjo en cada etapa."""
    base = run_dir(run_id)
    if not base.exists():
        return {}
    return {
        stage.name: sorted(p.name for p in stage.glob("*.json"))
        for stage in base.iterdir()
        if stage.is_dir()
    }


def flag_anomaly(
    run_id: str,
    stage: str,
    severity: Literal["info", "warning", "error"],
    message: str,
    recommended_action: Literal["continue", "retry_stage", "skip_item", "escalate_to_human"],
) -> dict:
    """Tool del flow_checker_agent: registra una anomalia detectada en `stage` (ej.
    "curacion descarto >90% de los candidatos", "DE-COP se skippeo en todos los chunks
    de este texto") en runs/<run_id>/flow_checks/. No decide nada por si solo -- el
    orquestador es quien actua sobre `recommended_action`."""
    record = {
        "stage": stage,
        "severity": severity,
        "message": message,
        "recommended_action": recommended_action,
    }
    write_run_artifact(run_id, "flow_checks", f"{stage}_{time.time_ns()}", record)
    return record
