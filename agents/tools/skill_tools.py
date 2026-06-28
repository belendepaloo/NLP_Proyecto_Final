"""
skill_tools.py — tools del orquestador para la skill persistente de Fase 3
(agents/skills/pipeline-learnings/). El orquestador LEE el contenido de la skill
(SKILL.md, learnings.jsonl, calibration_history.csv) con los tools de filesystem que
le da deepagents (read_file/ls), scopeados a ese directorio via el backend de
agents/orchestrator.py -- estos dos tools son solo para ESCRIBIR al final de un run,
sin necesitar que el agente arme el JSON/CSV a mano.

record_learning/record_calibration APENDEAN, nunca reescriben -- la skill es un log
entre runs, no un estado (ver SKILL.md, seccion "Al terminar un run").
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from typing import Callable, Literal

from mia_common.settings import settings


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def record_learning(
    run_id: str,
    stage: str,
    learning: str,
    severity: Literal["info", "warning", "critical"],
) -> dict:
    """Agrega una linea a agents/skills/pipeline-learnings/learnings.jsonl -- algo que
    valga la pena que un run futuro sepa (un bug, un patron de falla recurrente, un
    ajuste que funciono). Llamalo al final de cada run, exito o falla parcial."""
    record = {
        "timestamp": _now(),
        "run_id": run_id,
        "stage": stage,
        "severity": severity,
        "learning": learning,
    }
    settings.skill_dir.mkdir(parents=True, exist_ok=True)
    path = settings.skill_dir / "learnings.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def record_calibration(
    run_id: str,
    method: Literal["sage", "decop", "simia", "dualtest", "ensemble"],
    metric: str,
    value: float,
    notes: str = "",
) -> dict:
    """Agrega una fila a agents/skills/pipeline-learnings/calibration_history.csv -- un
    numero agregado de este run (ej. separacion member/non-member promedio) para que la
    serie historica de calibracion siga creciendo entre runs. No reemplaza al resultado
    crudo en runs/<run_id>/, que ya queda completo via write_run_artifact."""
    settings.skill_dir.mkdir(parents=True, exist_ok=True)
    path = settings.skill_dir / "calibration_history.csv"
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "run_id", "method", "metric", "value", "notes"])
        writer.writerow([_now(), run_id, method, metric, value, notes])
    return {"run_id": run_id, "method": method, "metric": metric, "value": value, "notes": notes}


def make_run_scoped_skill_tools(run_id: str) -> dict[str, Callable]:
    """Devuelve record_learning/record_calibration con `run_id` ya fijo via closure --
    mismo motivo que make_run_scoped_fs_tools (agents/tools/fs_tools.py): el
    orquestador es el unico que llama a estos, y run_id es constante durante todo el
    run, asi que no hay razon para dejar que lo tipee de nuevo cada vez (y arriesgue
    contaminar el log de aprendizajes ENTRE runs con un run_id mal escrito)."""

    def record_learning_bound(stage: str, learning: str, severity: Literal["info", "warning", "critical"]) -> dict:
        """Agrega una linea a agents/skills/pipeline-learnings/learnings.jsonl -- algo
        que valga la pena que un run futuro sepa (un bug, un patron de falla
        recurrente, un ajuste que funciono). Llamalo al final de cada run, exito o
        falla parcial."""
        return record_learning(run_id, stage, learning, severity)

    def record_calibration_bound(
        method: Literal["sage", "decop", "simia", "dualtest", "ensemble"],
        metric: str,
        value: float,
        notes: str = "",
    ) -> dict:
        """Agrega una fila a agents/skills/pipeline-learnings/calibration_history.csv --
        un numero agregado de este run (ej. separacion member/non-member promedio)."""
        return record_calibration(run_id, method, metric, value, notes)

    record_learning_bound.__name__ = "record_learning"
    record_calibration_bound.__name__ = "record_calibration"

    return {"record_learning": record_learning_bound, "record_calibration": record_calibration_bound}
