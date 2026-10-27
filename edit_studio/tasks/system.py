"""Background jobs for model management: downloads and scanning local folders."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..jobs import TaskContext, TaskError, task
from ..models.catalog import TOOLS_BY_ID
from ..models.hub import DownloadCancelled
from .common import download_tool


class DownloadModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repo: str = Field(pattern=r"^[\w.\-]+/[\w.\-]+$")
    variant: str | None = None
    skip_weights: list[str] = Field(default_factory=list)
    label: str = ""


@task("model.download", "net", DownloadModel)
def download_model(ctx: TaskContext, p: DownloadModel):
    from ..jobs import Cancelled

    def progress(done: int, total: int):
        ctx.progress(done / total if total else 0.0, "downloading", detail=f"{done / 1e9:.2f}/{total / 1e9:.2f} GB")

    ctx.progress(0.0, "preparing", force=True)
    try:
        ctx.services.hub.download(p.repo, p.variant, progress, ctx.job.cancel_event, tuple(p.skip_weights))
    except DownloadCancelled:
        raise Cancelled()
    except Exception as e:
        if ctx.job.cancel_event.is_set():
            raise Cancelled()
        name = type(e).__name__
        if name == "GatedRepoError" or "gated" in str(e).lower():
            raise TaskError("gated", str(e), repo=p.repo)
        if name == "RepositoryNotFoundError":
            raise TaskError("not_found", str(e), repo=p.repo)
        raise
    ctx.services.bus.publish("models", {"changed": p.repo})
    return []


class DownloadTool(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str


@task("tool.download", "net", DownloadTool)
def download_tool_job(ctx: TaskContext, p: DownloadTool):
    tool = TOOLS_BY_ID.get(p.id)
    if tool is None:
        raise TaskError("tool_missing", id=p.id)
    dest = ctx.services.paths.tools / tool.filename
    if not dest.exists():
        download_tool(tool, dest, ctx)
    ctx.services.bus.publish("models", {"changed": p.id})
    return []


class ScanModels(BaseModel):
    model_config = ConfigDict(extra="ignore")


@task("models.scan", "cpu", ScanModels)
def scan_models(ctx: TaskContext, p: ScanModels):
    s = ctx.services.settings.settings
    ctx.progress(0.0, "scanning", force=True)
    found = ctx.services.local.scan(
        s.model_folders, s.auto_detect_folders,
        progress=lambda i, n: (ctx.check(), ctx.progress(i / max(1, n), "scanning", detail=f"{i}/{n}")),
    )
    ctx.services.bus.publish("models", {"scanned": len(found)})
    return []
