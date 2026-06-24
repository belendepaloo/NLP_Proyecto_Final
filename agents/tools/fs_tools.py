"""
fs_tools.py — convencion de artifacts en disco por run, independiente de cualquier
filesystem virtual de deepagents (ver el plan: "debugging nunca depende de inspeccionar
internals de LangGraph"). Cada etapa del pipeline escribe aca lo que produjo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mia_common.settings import settings


def run_dir(run_id: str) -> Path:
    return settings.runs_dir / run_id


def stage_dir(run_id: str, stage: str) -> Path:
    d = run_dir(run_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifact(run_id: str, stage: str, name: str, data: Any) -> Path:
    """Escribe `data` (cualquier cosa JSON-serializable) como
    runs/<run_id>/<stage>/<name>.json. Devuelve el path escrito."""
    path = stage_dir(run_id, stage) / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    return path


def read_run_artifact(run_id: str, stage: str, name: str) -> Any:
    path = stage_dir(run_id, stage) / f"{name}.json"
    return json.loads(path.read_text())


def list_run_artifacts(run_id: str) -> dict[str, list[str]]:
    """Tree listing de runs/<run_id>/ -- usado por la pantalla de drill-down (Fase 4)
    y por flow_checker_agent (Fase 3) para inspeccionar que se produjo en cada etapa."""
    base = run_dir(run_id)
    if not base.exists():
        return {}
    return {
        stage.name: sorted(p.name for p in stage.glob("*.json"))
        for stage in base.iterdir()
        if stage.is_dir()
    }
