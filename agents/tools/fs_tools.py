"""
fs_tools.py — convencion de artifacts en disco por run, independiente de cualquier
filesystem virtual de deepagents (ver el plan: "debugging nunca depende de inspeccionar
internals de LangGraph"). Cada etapa del pipeline escribe aca lo que produjo.
"""

from __future__ import annotations

import json
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


def write_run_artifact(run_id: str, stage: str, name: str, data: Any) -> str:
    """Escribe `data` (cualquier cosa JSON-serializable) como
    runs/<run_id>/<stage>/<name>.json. Devuelve el path escrito (como string)."""
    path = stage_dir(run_id, stage) / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    return str(path)


def read_run_artifact(run_id: str, stage: str, name: str) -> Any:
    """Lee de vuelta el artifact escrito por write_run_artifact en
    runs/<run_id>/<stage>/<name>.json."""
    path = stage_dir(run_id, stage) / f"{name}.json"
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
