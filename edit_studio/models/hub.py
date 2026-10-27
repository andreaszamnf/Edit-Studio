"""Hugging Face Hub helpers: file selection, downloads with progress, cache inspection."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Callable
from typing import Any
from pathlib import Path

from huggingface_hub import HfApi, constants, hf_hub_download, scan_cache_dir, snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm

log = logging.getLogger(__name__)

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".msgpack", ".h5", ".gguf", ".ot", ".pb")
SKIP_EXTS = (".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".pdf")
VARIANTS = ("fp16", "fp32", "bf16", "ema", "non_ema")


class DownloadCancelled(Exception):
    pass


def cache_root(cache_dir: str | None) -> Path:
    return Path(cache_dir or constants.HF_HUB_CACHE)


def repo_folder(repo: str, cache_dir: str | None) -> Path:
    return cache_root(cache_dir) / ("models--" + repo.replace("/", "--"))


def local_snapshot(repo: str, cache_dir: str | None) -> Path | None:
    """Snapshot folder of the cached `main` revision, if any (no network)."""
    folder = repo_folder(repo, cache_dir)
    ref = folder / "refs" / "main"
    snaps = folder / "snapshots"
    if ref.is_file():
        snap = snaps / ref.read_text().strip()
        if snap.is_dir():
            return snap
    if snaps.is_dir():  # downloaded at a pinned revision by another tool
        candidates = sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    return None


def _is_variant_file(name: str, variant: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return f".{variant}." in base or f".{variant}-" in base or base.endswith(f".index.{variant}.json")


def _has_any_variant(name: str) -> bool:
    return any(_is_variant_file(name, v) for v in VARIANTS)


def select_files(
    siblings: list[tuple[str, int]],
    components: list[str] | None,
    variant: str | None,
    skip_weights: tuple[str, ...] = (),
) -> list[tuple[str, int]]:
    """Picks the files a diffusers pipeline needs (one weight format, one variant).

    `skip_weights` lists components whose weights come from elsewhere (e.g. a local transformer
    file); only their config files are kept.
    """
    by_folder: dict[str, list[tuple[str, int]]] = {}
    for name, size in siblings:
        folder = name.rsplit("/", 1)[0] if "/" in name else ""
        if components is not None and folder.split("/")[0] not in components and name != "model_index.json":
            continue
        if folder == "" and name != "model_index.json":
            continue  # root single-file checkpoints, images, docs
        if name.lower().endswith(SKIP_EXTS):
            continue
        by_folder.setdefault(folder, []).append((name, size))

    out: list[tuple[str, int]] = []
    for folder, files in by_folder.items():
        weights = [f for f in files if f[0].lower().endswith(WEIGHT_EXTS)]
        indexes = [f for f in files if ".index." in f[0] and f[0].endswith(".json")]
        others = [f for f in files if f not in weights and f not in indexes]
        if folder.split("/")[0] in skip_weights:
            out.extend(others)
            continue
        st = [f for f in weights if f[0].endswith(".safetensors")]
        chosen = st or [f for f in weights if f[0].endswith(".bin")] or weights
        st_indexes = [f for f in indexes if ".safetensors.index" in f[0]] if st else indexes
        if variant and any(_is_variant_file(f[0], variant) for f in chosen):
            chosen = [f for f in chosen if _is_variant_file(f[0], variant)]
            st_indexes = [f for f in st_indexes if _is_variant_file(f[0], variant)]
        else:
            chosen = [f for f in chosen if not _has_any_variant(f[0])]
            st_indexes = [f for f in st_indexes if not _has_any_variant(f[0])]
        out.extend(others + st_indexes + chosen)
    return sorted(out)


def pipeline_components(model_index: dict) -> list[str]:
    return [
        k for k, v in model_index.items() if not k.startswith("_") and isinstance(v, list) and v and v[0] is not None
    ]


class _ProgressTqdm(hf_tqdm):
    """Forwards byte progress of `snapshot_download` to a callback; raising aborts the download."""

    callback: Callable[[int, int], None] | None = None
    cancel_event: threading.Event | None = None

    def __init__(self, *args, **kwargs):
        self._name = kwargs.get("name")
        kwargs["disable"] = True  # no console bars; a disabled tqdm doesn't count, so we do
        super().__init__(*args, **kwargs)
        self._count = int(kwargs.get("initial") or 0)

    def update(self, n=1):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise DownloadCancelled()
        self._count += int(n or 0)
        if self._name == "huggingface_hub.snapshot_download" and self.callback:
            self.callback(self._count, int(self.total or 0))


def _abort_transfers() -> None:
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception as e:  # older/newer huggingface_hub without it, or hf_xet not installed
        log.debug("No Xet session to abort: %s", e)


class HubClient:
    def __init__(self, token_fn: Callable[[], str | None], cache_fn: Callable[[], str | None], manifest_dir: Path):
        self.token_fn = token_fn
        self.cache_fn = cache_fn
        self.manifest_dir = manifest_dir
        self._aborts = 0  # cancelled downloads so far (each one stops all Xet transfers)

    # ---------------------------------------------------------------- remote
    def remote_files(self, repo: str, variant: str | None, skip_weights: tuple[str, ...] = ()) -> list[tuple[str, int]]:
        token = self.token_fn()
        info = HfApi().model_info(repo, files_metadata=True, token=token)
        siblings = [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
        components = None
        if any(name == "model_index.json" for name, _ in siblings):
            path = hf_hub_download(repo, "model_index.json", token=token, cache_dir=self.cache_fn())
            components = pipeline_components(json.loads(Path(path).read_text(encoding="utf-8")))
        return select_files(siblings, components, variant, skip_weights)

    def download(
        self,
        repo: str,
        variant: str | None,
        progress: Callable[[int, int], None],
        cancel_event: threading.Event,
        skip_weights: tuple[str, ...] = (),
    ) -> Path:
        files = self.remote_files(repo, variant, skip_weights)
        total = sum(s for _, s in files)
        snap = self.snapshot(repo)
        missing = sum(s for name, s in files if not (snap and (snap / name).exists()))
        root = cache_root(self.cache_fn())
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if missing > free - 2 * 1024**3:
            raise OSError(28, f"No space left on device: need {missing / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
        cls = type("_BoundProgress", (_ProgressTqdm,), {})
        cls.callback = staticmethod(lambda n, t: progress(n, max(t, total)))
        cls.cancel_event = cancel_event
        for _attempt in range(5):
            aborts = self._aborts
            try:
                path = self._snapshot_download(repo, [name for name, _ in files], cls, cancel_event)
                break
            except DownloadCancelled:
                raise
            except Exception:
                if self._aborts == aborts:
                    raise
                # another download was cancelled, which stops every Xet transfer: carry on where we were
                log.info("Resuming %s after another download was cancelled", repo)
        else:
            raise RuntimeError(f"download of {repo} was interrupted too often")
        self.write_manifest(repo, [name for name, _ in files], total, skip_weights)
        return Path(path)

    def _snapshot_download(self, repo: str, files: list[str], tqdm_cls, cancel_event: threading.Event) -> str:
        """snapshot_download in a helper thread, so cancelling works at once.

        Large files come through Xet, whose transfers run in Rust: an exception raised from its progress
        callback doesn't stop them. Cancelling therefore aborts the Xet session (as Ctrl+C does) and returns
        right away; the partial files stay in the cache and a later download resumes them."""
        result: dict[str, Any] = {}

        def run():
            try:
                result["path"] = snapshot_download(
                    repo,
                    allow_patterns=files,
                    token=self.token_fn(),
                    cache_dir=self.cache_fn(),
                    tqdm_class=tqdm_cls,
                    max_workers=8,
                )
            except BaseException as e:  # noqa: BLE001 - handed to the waiting thread
                result["error"] = e

        worker = threading.Thread(target=run, name=f"download-{repo}", daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.25)
            if cancel_event.is_set():
                self._aborts += 1
                _abort_transfers()
                worker.join(15)  # plain HTTP transfers stop at their next chunk
                raise DownloadCancelled()
        if "error" in result:
            raise result["error"]
        return result["path"]

    # ---------------------------------------------------------------- local state
    def _manifest_path(self, repo: str) -> Path:
        return self.manifest_dir / (repo.replace("/", "--") + ".json")

    def write_manifest(self, repo: str, files: list[str], total: int, skip_weights: tuple[str, ...] = ()) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = {"files": files, "bytes": total, "skip_weights": sorted(skip_weights)}
        self._manifest_path(repo).write_text(json.dumps(payload), encoding="utf-8")

    def snapshot(self, repo: str) -> Path | None:
        return local_snapshot(repo, self.cache_fn())

    def is_installed(self, repo: str, skip_weights: tuple[str, ...] = ()) -> bool:
        snap = self.snapshot(repo)
        if snap is None:
            return False
        try:
            manifest = json.loads(self._manifest_path(repo).read_text(encoding="utf-8"))
            files = manifest["files"]
            skipped = set(manifest.get("skip_weights", []))
        except (OSError, ValueError, KeyError):
            return snapshot_looks_complete(snap, skip_weights)
        if not skipped <= set(skip_weights):  # an earlier partial download (configs only)
            return snapshot_looks_complete(snap, skip_weights)
        return all((snap / f).exists() for f in files)

    def size_on_disk(self, repo: str) -> int:
        folder = repo_folder(repo, self.cache_fn()) / "blobs"
        if not folder.is_dir():
            return 0
        return sum(f.stat().st_size for f in folder.iterdir() if f.is_file())

    def delete(self, repo: str) -> None:
        folder = repo_folder(repo, self.cache_fn())
        if folder.is_dir():
            shutil.rmtree(folder)
        self._manifest_path(repo).unlink(missing_ok=True)

    def cached_pipelines(self) -> list[dict]:
        """Diffusers pipelines present in the HF cache (downloaded by any tool)."""
        out = []
        try:
            info = scan_cache_dir(self.cache_fn())
        except Exception as e:  # cache folder missing or unreadable
            log.info("HF cache scan skipped: %s", e)
            return out
        for repo in info.repos:
            if repo.repo_type != "model":
                continue
            snap = local_snapshot(repo.repo_id, self.cache_fn())
            if snap is None or not (snap / "model_index.json").is_file():
                continue
            try:
                idx = json.loads((snap / "model_index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "repo": repo.repo_id,
                    "class_name": idx.get("_class_name", ""),
                    "index": idx,
                    "path": str(snap),
                    "size": int(repo.size_on_disk),
                    "complete": snapshot_looks_complete(snap),
                }
            )
        return out


def snapshot_looks_complete(snap: Path, skip_weights: tuple[str, ...] = ()) -> bool:
    """Heuristic for snapshots without our manifest: every model component has its weights."""
    idx_file = snap / "model_index.json"
    if not idx_file.is_file():
        return any(p.suffix in (".safetensors", ".gguf") for p in snap.iterdir())
    try:
        idx = json.loads(idx_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for comp in pipeline_components(idx):
        library, cls_name = idx[comp][0], idx[comp][1] or ""
        folder = snap / comp
        if not folder.is_dir():
            return False
        if "Tokenizer" in cls_name or "Scheduler" in cls_name or "Processor" in cls_name or comp in skip_weights:
            continue
        if library not in ("diffusers", "transformers"):
            continue
        files = os.listdir(folder)
        shard_indexes = [f for f in files if ".index" in f and f.endswith(".json")]
        if shard_indexes:
            try:
                weight_map = json.loads((folder / shard_indexes[0]).read_text(encoding="utf-8"))["weight_map"]
            except (OSError, ValueError, KeyError):
                return False
            if not all((folder / shard).exists() for shard in set(weight_map.values())):
                return False
        elif not any(f.endswith((".safetensors", ".bin")) for f in files):
            return False
    return True
