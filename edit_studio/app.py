"""FastAPI application: JSON API, live events (SSE), media files and the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import __version__
from .config import WEB_DIR
from .jobs import TASKS
from .models.catalog import CATALOG, FAMILIES, TOOL_MODELS, family_from_index
from .models.manager import ModelError
from .services import Services

log = logging.getLogger(__name__)

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


def _host_only(header: str) -> str:
    h = header.strip().lower()
    if h.startswith("["):  # IPv6 literal, e.g. [::1]:8188
        return h.split("]")[0] + "]"
    return h.rsplit(":", 1)[0]


def open_in_file_manager(path: Path) -> None:
    if sys.platform == "win32":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if path.is_file() else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if path.is_file() else path)])


def create_app(services: Services, allow_remote: bool = False) -> FastAPI:
    app = FastAPI(title="Edit Studio", version=__version__, docs_url=None, redoc_url=None)
    app.state.services = services
    S = services
    _models_cache: dict[str, Any] = {"at": 0.0, "data": None}
    _cache_lock = threading.Lock()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Only this computer may talk to the app (blocks DNS-rebinding), and state-changing calls need
        # a custom header that other web sites cannot send without our (absent) CORS permission.
        if not allow_remote and _host_only(request.headers.get("host") or "") not in LOCAL_HOSTS:
            return JSONResponse({"error": "forbidden_host"}, status_code=400)
        if request.method in MUTATING and request.headers.get("x-edit-studio") != "1":
            return JSONResponse({"error": "missing_header"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ModelError)
    async def model_error(_req, exc: ModelError):
        return JSONResponse({"error": exc.key, "info": exc.info, "message": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ system
    @app.get("/api/system")
    def system():
        return {
            "version": __version__,
            "hardware": S.hw.summary(),
            "slots": S.jobs.slot_ids,
            "loaded": S.models.status(),
            "nf4": S.models.can_nf4(),
            "settings": S.settings.settings.public(),
            "data_dir": str(S.paths.home),
            "tasks": sorted(TASKS),
        }

    @app.get("/api/settings")
    def get_settings():
        return S.settings.settings.public()

    @app.put("/api/settings")
    def put_settings(changes: dict = Body(...)):
        before = S.settings.settings.enabled_gpus
        try:
            s = S.settings.update(changes)
        except ValueError as e:
            raise HTTPException(400, str(e))
        with _cache_lock:
            _models_cache["at"] = 0.0
        return {**s.public(), "restart_needed": s.enabled_gpus != before}

    # ------------------------------------------------------------------ models
    def _catalog_entry(spec) -> dict:
        installed = S.hub.is_installed(spec.repo)
        try:
            plan = S.models.plan(S.models.resolve(spec.id), S.jobs.slot_ids).public()
        except Exception as e:  # never break the listing because of one model
            log.info("plan failed for %s: %s", spec.id, e)
            plan = None
        return {
            "id": spec.id,
            "name": spec.name,
            "kind": spec.kind,
            "family": spec.family,
            "repo": spec.repo,
            "license": spec.license,
            "commercial": spec.commercial,
            "gated": spec.gated,
            "recommended": spec.recommended,
            "download_gb": spec.download_gb,
            "on_disk_gb": round(S.hub.size_on_disk(spec.repo) / 1e9, 1) if installed else 0,
            "description": spec.description,
            "tags": list(spec.tags),
            "tasks": list(spec.supported_tasks()),
            "defaults": spec.defaults,
            "prompt_note": spec.fam.prompt_note,
            "negative_prompt": spec.fam.negative_prompt,
            "installed": installed,
            "plan": plan,
            "source": "catalog",
        }

    def _models_payload() -> dict:
        catalog_repos = {m.repo.lower() for m in CATALOG}
        hf_other = []
        for entry in S.hub.cached_pipelines():
            found = family_from_index(entry["index"], entry["repo"])
            if not found or entry["repo"].lower() in catalog_repos:
                continue
            fam_id = found[0]
            fam = FAMILIES[fam_id]
            hf_other.append({
                "id": "hf:" + entry["repo"], "name": entry["repo"].split("/")[-1], "repo": entry["repo"],
                "kind": fam.kind, "family": fam_id, "tasks": list(fam.pipelines), "installed": entry["complete"],
                "on_disk_gb": round(entry["size"] / 1e9, 1), "source": "hf-cache", "prompt_note": fam.prompt_note,
                "negative_prompt": fam.negative_prompt,
            })
        local = []
        for lm in S.local.list():
            d = lm.public()
            fam = FAMILIES.get(lm.family or "")
            if lm.kind in ("checkpoint", "denoiser") and fam is not None:
                d["tasks"] = ["inpaint"] if lm.variant == "inpaint" else list(fam.pipelines)
                d["kind_media"] = fam.kind
                d["prompt_note"] = fam.prompt_note
                d["negative_prompt"] = fam.negative_prompt
                d["needs"] = (
                    [] if not lm.base_repo or S.hub.is_installed(lm.base_repo, fam.denoisers[:1]) else [lm.base_repo]
                )
            local.append(d)
        tools = [
            {"id": t.id, "kind": t.kind, "name": t.name, "size_mb": t.size_mb, "scale": t.scale,
             "installed": (S.paths.tools / t.filename).exists()}
            for t in TOOL_MODELS
        ]
        return {
            "catalog": [_catalog_entry(m) for m in CATALOG],
            "hf_cache": hf_other,
            "local": local,
            "tools": tools,
            "roots": S.local.roots_public(),
            "loaded": S.models.status(),
            "hf_cache_dir": str(S.hub.cache_fn() or ""),
        }

    @app.get("/api/models")
    def models(refresh: bool = False):
        with _cache_lock:
            fresh = time.time() - _models_cache["at"] < 30
            if refresh or not fresh or _models_cache["data"] is None:
                _models_cache["data"] = _models_payload()
                _models_cache["at"] = time.time()
            return _models_cache["data"]

    def _invalidate(_evt=None):
        with _cache_lock:
            _models_cache["at"] = 0.0

    @app.post("/api/models/scan")
    def scan():
        _invalidate()
        return S.jobs.submit("models.scan", {}).public()

    def _download_jobs(model_id: str) -> list[dict]:
        if model_id in {t.id for t in TOOL_MODELS}:
            return [S.jobs.submit("tool.download", {"id": model_id}).public()]
        src = S.models.resolve(model_id)
        jobs = []
        for m in S.models.missing_downloads(src):
            existing = _find_active_download(m["repo"])
            jobs.append(existing or S.jobs.submit("model.download", {**m, "label": src.name}).public())
        return jobs

    def _find_active_download(repo: str) -> dict | None:
        for j in S.jobs.list():
            if j.task == "model.download" and j.params.get("repo") == repo and not j.finished_state:
                return j.public()
        return None

    @app.post("/api/models/download")
    def download(body: dict = Body(...)):
        _invalidate()
        return {"jobs": _download_jobs(str(body.get("model", "")))}

    @app.delete("/api/models/{model_id:path}")
    def delete_model(model_id: str):
        tool = next((t for t in TOOL_MODELS if t.id == model_id), None)
        if tool:
            (S.paths.tools / tool.filename).unlink(missing_ok=True)
        else:
            spec = next((m for m in CATALOG if m.id == model_id), None)
            repo = spec.repo if spec else model_id[3:] if model_id.startswith("hf:") else None
            if not repo:
                raise HTTPException(400, "only downloaded models can be deleted")
            S.models.unload_model(model_id)
            S.hub.delete(repo)
        _invalidate()
        return {"ok": True}

    @app.post("/api/models/unload")
    def unload():
        return {"unloaded": S.models.unload_all(only_idle=True)}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs")
    def submit(body: dict = Body(...)):
        name = body.get("task")
        params = body.get("params") or {}
        spec = TASKS.get(name)
        if spec is None:
            raise HTTPException(400, f"unknown task {name}")
        if spec.params_model is not None:
            try:
                params = spec.params_model.model_validate(params).model_dump()
            except ValidationError as e:
                return JSONResponse({"error": "invalid_params", "details": json.loads(e.json())}, status_code=422)
        downloads: list[dict] = []
        if spec.model_param and params.get(spec.model_param):
            if body.get("auto_download", True):
                downloads = [d for d in _download_jobs(params[spec.model_param]) if d["status"] != "done"]
        depends = downloads[-1]["id"] if downloads else None
        job = S.jobs.submit(name, params, depends_on=depends)
        return {"job": job.public(), "downloads": downloads}

    @app.get("/api/jobs")
    def jobs():
        return [j.public() for j in S.jobs.list()]

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return {"ok": S.jobs.cancel(job_id)}

    @app.delete("/api/jobs/{job_id}")
    def remove_job(job_id: str):
        return {"ok": S.jobs.remove(job_id)}

    @app.post("/api/jobs/clear")
    def clear_jobs():
        return {"removed": S.jobs.clear_finished()}

    @app.get("/api/events")
    async def events(request: Request):
        q = S.bus.subscribe()

        async def stream():
            try:
                yield "retry: 2000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    if item["type"] == "models":
                        _invalidate()
                    yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            finally:
                S.bus.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ media
    @app.post("/api/uploads")
    def upload(file: UploadFile = File(...), hidden: bool = Form(False)):
        suffix = Path(file.filename or "upload").suffix.lower()
        fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=S.paths.cache)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    out.write(chunk)
            item = S.storage.add_upload(tmp, file.filename or f"upload{suffix}", hidden=hidden)
        except ValueError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        return item.public()

    @app.get("/api/gallery")
    def gallery(kind: str | None = None, favorites: bool = False, uploads: bool = False, offset: int = 0,
                limit: int = 60, q: str = "", task: str = ""):
        tasks = tuple(x for x in task.split(",") if x)
        items, total = S.storage.list(kind, favorites, uploads, offset, min(limit, 500), q, tasks)
        return {"items": [i.public() for i in items], "total": total}

    @app.get("/api/gallery/{item_id}")
    def gallery_item(item_id: str):
        item = S.storage.get(item_id)
        if item is None:
            raise HTTPException(404)
        return item.public()

    @app.patch("/api/gallery/{item_id}")
    def gallery_update(item_id: str, body: dict = Body(...)):
        try:
            return S.storage.set_favorite(item_id, bool(body.get("favorite"))).public()
        except KeyError:
            raise HTTPException(404)

    @app.delete("/api/gallery/{item_id}")
    def gallery_delete(item_id: str):
        try:
            S.storage.delete(item_id)
        except KeyError:
            raise HTTPException(404)
        S.bus.publish("gallery", {"removed": [item_id]})
        return {"ok": True}

    @app.post("/api/open")
    def open_path(body: dict = Body(...)):
        if allow_remote:
            raise HTTPException(403, "only on this computer")
        target = str(body.get("target", "outputs"))
        folders = {"outputs": S.paths.outputs, "inputs": S.paths.inputs, "home": S.paths.home, "tools": S.paths.tools}
        if target == "hf_cache":
            from .models.hub import cache_root

            folders[target] = cache_root(S.hub.cache_fn())
        if target in folders:
            path = folders[target]
            path.mkdir(parents=True, exist_ok=True)
        else:
            try:
                path = S.storage.path_of(target)
            except KeyError:
                raise HTTPException(404)
        open_in_file_manager(path)
        return {"ok": True}

    @app.get("/files/{rel:path}")
    def files(rel: str):
        base = S.paths.home.resolve()
        path = (base / rel).resolve()
        allowed = (S.paths.outputs.resolve(), S.paths.inputs.resolve())
        if not any(path.is_relative_to(a) for a in allowed) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------------ UI
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
