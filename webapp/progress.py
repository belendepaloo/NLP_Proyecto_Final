"""
progress.py — calcula el estado estructurado del pipeline desde los artifacts en
runs/<run_id>/, para renderizar el diagrama de nodos y las stage-cards en run.html.
Lee archivos de disco; no accede al checkpoint de LangGraph.
"""

from __future__ import annotations

import os

from agents.tools.fs_tools import list_run_artifacts, read_run_artifact, run_dir
from mia_common.settings import settings

# Orden canónico de etapas para el diagrama de nodos.
STAGE_ORDER = ["bibliography", "waiting", "authorship", "voice", "sage", "mia", "result"]


def compute_pipeline_progress(run_id: str) -> dict:
    artifacts = list_run_artifacts(run_id)
    bib_files = artifacts.get("bibliography", [])
    curation_files = artifacts.get("curation", [])
    sage_files = artifacts.get("sage", [])
    mia_files = artifacts.get("mia_scores", [])
    flow_files = artifacts.get("flow_checks", [])

    # --- Bibliografía ---
    has_candidates = "candidates.json" in bib_files
    candidate_list: list[dict] = []
    if has_candidates:
        data = read_run_artifact(run_id, "bibliography", "candidates")
        if "error" not in data:
            candidate_list = data.get("candidates", [])

    # --- Curación: Autoría ---
    authorship_verdicts: list[dict] = []
    for f in curation_files:
        if not f.startswith("authorship_"):
            continue
        data = read_run_artifact(run_id, "curation", f.removesuffix(".json"))
        if "error" not in data:
            authorship_verdicts.append(data)

    auth_keep = sum(1 for v in authorship_verdicts if v.get("decision") == "keep")
    auth_drop = sum(1 for v in authorship_verdicts if v.get("decision") == "drop")
    auth_review = sum(1 for v in authorship_verdicts if v.get("decision") == "needs_human_review")

    # --- Chunks ---
    n_chunks_total = sum(1 for f in curation_files if f.startswith("chunk_"))

    # --- Curación: Voz ---
    voice_verdicts: list[dict] = []
    for f in curation_files:
        if not f.startswith("voice_"):
            continue
        data = read_run_artifact(run_id, "curation", f.removesuffix(".json"))
        if "error" not in data:
            voice_verdicts.append(data)

    voice_keep = sum(1 for v in voice_verdicts if v.get("decision") == "keep")
    voice_drop = sum(1 for v in voice_verdicts if v.get("decision") in ("drop", "target_reached"))

    # Agrupar voz por documento para el grid de chunks
    voice_by_doc: dict[str, list[dict]] = {}
    for v in voice_verdicts:
        doc = v.get("document_id", "?")
        voice_by_doc.setdefault(doc, []).append(v)

    # --- SAGE ---
    sage_results: list[dict] = []
    for f in sage_files:
        if not f.startswith("paraphrase_"):
            continue
        data = read_run_artifact(run_id, "sage", f.removesuffix(".json"))
        if "error" not in data:
            sage_results.append(data)

    sage_passed = sum(1 for r in sage_results if r.get("passed_qa", True))
    sage_failed = sum(1 for r in sage_results if not r.get("passed_qa", True))

    # --- MIA scoring ---
    mia_results: list[dict] = []
    for f in mia_files:
        data = read_run_artifact(run_id, "mia_scores", f.removesuffix(".json"))
        if "error" not in data:
            data["_chunk_id"] = f.removesuffix(".json")
            mia_results.append(data)

    decop_scored = sum(1 for r in mia_results if r.get("decop") is not None)
    decop_skipped = sum(1 for r in mia_results if r.get("decop") is None)
    dualtest_scored = sum(1 for r in mia_results if r.get("dualtest") is not None)
    dualtest_skipped = sum(1 for r in mia_results if r.get("dualtest") is None)

    # --- Anomalías ---
    anomalies: list[dict] = []
    for f in flow_files:
        data = read_run_artifact(run_id, "flow_checks", f.removesuffix(".json"))
        if "error" not in data:
            anomalies.append(data)

    # --- Timeline de actividad reciente ---
    timeline = _build_timeline(run_id, artifacts)

    return {
        "bibliography": {
            "n_candidates": len(candidate_list),
            "candidates": candidate_list,
            "done": has_candidates and len(candidate_list) > 0,
        },
        "authorship": {
            "verdicts": authorship_verdicts,
            "keep": auth_keep,
            "drop": auth_drop,
            "review": auth_review,
            "total": len(authorship_verdicts),
        },
        "chunks": {"total": n_chunks_total},
        "voice": {
            "verdicts": voice_verdicts,
            "keep": voice_keep,
            "drop": voice_drop,
            "total": len(voice_verdicts),
            "target": settings.curator_target_chunks_per_text,
            "by_doc": voice_by_doc,
        },
        "sage": {
            "results": sage_results,
            "done": len(sage_results),
            "passed": sage_passed,
            "failed": sage_failed,
            "target": voice_keep,
        },
        "mia": {
            "results": mia_results,
            "n_scored": len(mia_results),
            "decop_scored": decop_scored,
            "decop_skipped": decop_skipped,
            "dualtest_scored": dualtest_scored,
            "dualtest_skipped": dualtest_skipped,
            "target": len(sage_results),
        },
        "anomalies": anomalies,
        "timeline": timeline,
    }


def detect_active_stage(run_status: str, progress: dict) -> str:
    """Devuelve el id de la etapa activa según el estado del run y los artifacts."""
    if run_status == "done":
        return "result"
    if progress["mia"]["n_scored"] > 0:
        return "mia"
    if progress["sage"]["done"] > 0:
        return "sage"
    if progress["voice"]["total"] > 0:
        return "voice"
    if progress["authorship"]["total"] > 0:
        return "authorship"
    if run_status == "waiting_human":
        return "waiting"
    if progress["bibliography"]["done"]:
        return "authorship"
    return "bibliography"


def build_pipeline_nodes(run_status: str, progress: dict) -> list[dict]:
    """Construye la lista de nodos del diagrama con id, icon, label, status y stat."""
    active = detect_active_stage(run_status, progress)
    active_idx = STAGE_ORDER.index(active) if active in STAGE_ORDER else 0

    def node_status(stage_id: str) -> str:
        idx = STAGE_ORDER.index(stage_id)
        if run_status == "error" and idx == active_idx:
            return "error"
        if idx < active_idx or run_status == "done":
            return "done"
        if idx == active_idx:
            return "waiting" if run_status == "waiting_human" and stage_id == "waiting" else "active"
        return "pending"

    bib = progress["bibliography"]
    auth = progress["authorship"]
    voice = progress["voice"]
    sage = progress["sage"]
    mia = progress["mia"]

    return [
        {
            "id": "bibliography",
            "icon": "🔎",
            "label": "Búsqueda",
            "sublabel": "bibliography_agent",
            "status": node_status("bibliography"),
            "stat": f"{bib['n_candidates']} texto{'s' if bib['n_candidates'] != 1 else ''}" if bib["n_candidates"] else "",
        },
        {
            "id": "waiting",
            "icon": "✋",
            "label": "Revisión humana",
            "sublabel": "human-in-the-loop",
            "status": node_status("waiting"),
            "stat": "aprobado" if active_idx > STAGE_ORDER.index("waiting") else "",
        },
        {
            "id": "authorship",
            "icon": "👤",
            "label": "Autoría",
            "sublabel": "curator_agent",
            "status": node_status("authorship"),
            "stat": f"{auth['keep']} ✓ / {auth['drop']} ✗" if auth["total"] else "",
        },
        {
            "id": "voice",
            "icon": "🎭",
            "label": "Voz",
            "sublabel": "curator_agent",
            "status": node_status("voice"),
            "stat": f"{voice['keep']}/{voice['target']} chunks" if voice["total"] else "",
        },
        {
            "id": "sage",
            "icon": "✂️",
            "label": "SAGE",
            "sublabel": "sage_qa_agent",
            "status": node_status("sage"),
            "stat": f"{sage['done']}/{sage['target']}" if sage["target"] else "",
        },
        {
            "id": "mia",
            "icon": "📊",
            "label": "DE-COP / DUALTEST",
            "sublabel": "mia_agent",
            "status": node_status("mia"),
            "stat": f"{mia['n_scored']}/{mia['target']}" if mia["target"] else "",
        },
        {
            "id": "result",
            "icon": "🎯",
            "label": "Resultado",
            "sublabel": "ensemble",
            "status": node_status("result"),
            "stat": "",
        },
    ]


def _build_timeline(run_id: str, artifacts: dict[str, list[str]]) -> list[dict]:
    base = run_dir(run_id)
    entries: list[dict] = []
    for stage, files in artifacts.items():
        for fname in files:
            path = base / stage / fname
            if path.exists():
                entries.append({
                    "stage": stage,
                    "file": fname,
                    "mtime": os.path.getmtime(path),
                    "label": _artifact_label(stage, fname),
                })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries[:15]


def _artifact_label(stage: str, fname: str) -> str:
    name = fname.removesuffix(".json")
    if stage == "bibliography":
        return "Candidatos aprobados" if name == "candidates" else name
    if stage == "curation":
        if name.startswith("authorship_"):
            return f"Veredicto de autoría — {name.removeprefix('authorship_')}"
        if name.startswith("chunk_"):
            return f"Chunk descargado — {name.removeprefix('chunk_')}"
        if name.startswith("voice_"):
            return f"Score de voz — {name.removeprefix('voice_')}"
    if stage == "sage":
        if name.startswith("paraphrase_"):
            return f"Parafraseo — {name.removeprefix('paraphrase_')}"
    if stage == "mia_scores":
        return f"MIA scoring — {name}"
    if stage == "flow_checks":
        return "⚠️ Anomalía detectada"
    return name
