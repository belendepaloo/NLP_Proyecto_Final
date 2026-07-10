"""
main.py — Fase 4: interfaz web minima sobre el orquestador de la Fase 2/3. Server-
rendered (Jinja2) + un EventSource de unas pocas lineas para refrescar la pantalla del
run cuando cambia de estado -- sin SPA, ver README_feature_agentes.md.

Correr con: uvicorn webapp.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agents.tools.fs_tools import list_run_artifacts
from mia_common.settings import settings
from webapp.progress import build_pipeline_nodes, compute_pipeline_progress, detect_active_stage
from webapp.results import build_results
from webapp.run_manager import TARGET_MODEL_CHOICES, get_run, reconstruct_handle_from_disk, resume_run, start_run, submit_decision

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MIA pipeline")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Que proveedores tienen API key configurada en .env -- el form los muestra a todos
# (igual de utiles para planear el run), pero deshabilita el boton si falta la key, en
# vez de dejar que el usuario elija algo que va a fallar recien al arrancar el run.
PROVIDER_HAS_KEY = {
    "groq": lambda: bool(settings.groq_api_keys()),
    "openai": lambda: bool(settings.openai_api_key),
    "anthropic": lambda: bool(settings.anthropic_api_key),
    "google": lambda: bool(settings.google_api_key),
}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    providers = [
        {"id": provider, "model_name": model_name, "has_key": PROVIDER_HAS_KEY[provider]()}
        for provider, model_name in TARGET_MODEL_CHOICES.items()
    ]
    default_provider = next((p["id"] for p in providers if p["has_key"]), providers[0]["id"])
    return templates.TemplateResponse(
        request,
        "index.html",
        {"providers": providers, "default_provider": default_provider, "simia_enabled": settings.simia_enabled},
    )


@app.post("/runs")
def create_run(author: str = Form(...), n_texts: int = Form(5), target_provider: str = Form("groq"), demo_mode: bool = Form(False)) -> RedirectResponse:
    run_id = start_run(author.strip(), n_texts, target_provider=target_provider, demo_mode=demo_mode)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str) -> HTMLResponse:
    handle = get_run(run_id)
    artifacts = list_run_artifacts(run_id)
    if handle is None:
        if not artifacts:
            raise HTTPException(404, f"run '{run_id}' no encontrado (ni en memoria ni en disco)")
        # El handle en memoria se perdio (reinicio del server) pero quedan artifacts
        # reales en runs/<run_id>/ -- mostrar una vista degradada en vez de un 404 que
        # esconde un run que en realidad si produjo resultados (ver el docstring de
        # reconstruct_handle_from_disk).
        handle = reconstruct_handle_from_disk(run_id)
    results = build_results(run_id)
    action_requests = (handle.pending_interrupt or {}).get("action_requests", [])
    progress = compute_pipeline_progress(run_id)
    pipeline_nodes = build_pipeline_nodes(handle.status, progress)
    active_stage = detect_active_stage(handle.status, progress)
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run": handle,
            "artifacts": artifacts,
            "action_requests": action_requests,
            "results": results,
            "progress": progress,
            "pipeline_nodes": pipeline_nodes,
            "active_stage": active_stage,
        },
    )


@app.post("/runs/{run_id}/resume")
def resume_run_endpoint(run_id: str) -> RedirectResponse:
    """Retoma un run desde el checkpoint de LangGraph sin perder el trabajo ya hecho."""
    if not list_run_artifacts(run_id):
        raise HTTPException(404, f"No hay artifacts para '{run_id}' — no hay nada que retomar")
    resume_run(run_id)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/decide")
def decide(
    run_id: str,
    decision_type: str = Form(...),
    edited_args: str = Form(""),
    reject_message: str = Form(""),
) -> RedirectResponse:
    handle = get_run(run_id)
    if handle is None:
        raise HTTPException(404, f"run '{run_id}' no encontrado")
    if handle.pending_interrupt is None:
        raise HTTPException(400, "este run no tiene nada pendiente de revision humana")

    action_requests = handle.pending_interrupt.get("action_requests", [])
    decisions: list[dict[str, Any]] = []
    for req in action_requests:
        if decision_type == "edit":
            decisions.append({"type": "edit", "edited_action": {"name": req["name"], "args": json.loads(edited_args)}})
        elif decision_type == "reject":
            decisions.append({"type": "reject", "message": reject_message or "Rechazado por el usuario."})
        else:
            decisions.append({"type": "approve"})

    submit_decision(run_id, decisions)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}/stream")
async def stream(run_id: str) -> StreamingResponse:
    """SSE dual-channel:
    - Eventos default (data: reload/done/error) → el frontend recarga la página completa.
    - Eventos nombrados (event: agentlog) → el frontend los append al log sin recargar."""

    async def event_source():
        handle = get_run(run_id)
        if handle is None:
            return
        if handle.status in ("done", "error"):
            yield f"data: {handle.status}\n\n"
            return

        baseline_status = handle.status
        baseline_artifact_count = sum(len(v) for v in list_run_artifacts(run_id).values())
        baseline_event_count = handle.event_count
        ticks_since_emit = 0

        while True:
            handle = get_run(run_id)
            if handle is None:
                return

            # ── Nuevos eventos de agentes → SSE nombrado (no recarga) ──────
            current_event_count = handle.event_count
            if current_event_count > baseline_event_count:
                new_events = handle.events[baseline_event_count:current_event_count]
                for ev in new_events:
                    yield f"event: agentlog\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                baseline_event_count = current_event_count
                ticks_since_emit = 0

            # ── Status o artifact cambió → reload completo ─────────────────
            current_artifact_count = sum(len(v) for v in list_run_artifacts(run_id).values())
            status_changed = handle.status != baseline_status
            artifacts_changed = current_artifact_count != baseline_artifact_count
            if status_changed or artifacts_changed:
                yield f"data: reload\n\n"
                baseline_status = handle.status
                baseline_artifact_count = current_artifact_count
                ticks_since_emit = 0
                if handle.status in ("done", "error"):
                    return
            else:
                ticks_since_emit += 1
                if ticks_since_emit >= 30:  # 30 × 0.5s = 15s sin cambios → heartbeat
                    yield ": heartbeat\n\n"
                    ticks_since_emit = 0

            if handle.status in ("done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_source(), media_type="text/event-stream")
