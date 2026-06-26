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
from webapp.run_manager import TARGET_MODEL_CHOICES, get_run, start_run, submit_decision

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
def create_run(author: str = Form(...), n_texts: int = Form(5), target_provider: str = Form("groq")) -> RedirectResponse:
    run_id = start_run(author.strip(), n_texts, target_provider=target_provider)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str) -> HTMLResponse:
    handle = get_run(run_id)
    if handle is None:
        raise HTTPException(404, f"run '{run_id}' no encontrado (el server se reinicio?)")
    artifacts = list_run_artifacts(run_id)
    action_requests = (handle.pending_interrupt or {}).get("action_requests", [])
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run": handle,
            "artifacts": artifacts,
            "action_requests": action_requests,
        },
    )


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
    """SSE: un evento cada vez que cambia el status del run, para que la pantalla se
    refresque sola (ver el <script> minimo en run.html) en vez de hacer polling manual."""

    async def event_source():
        handle = get_run(run_id)
        if handle is None:
            return
        baseline_status = handle.status  # no emitir nada por el estado que ya tenia la pagina al cargar
        while True:
            handle = get_run(run_id)
            if handle is None:
                return
            if handle.status != baseline_status:
                yield f"data: {handle.status}\n\n"
                return
            if handle.status in ("done", "error"):
                return
            await asyncio.sleep(1.5)

    return StreamingResponse(event_source(), media_type="text/event-stream")
