"""Wires all long-lived app objects together."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Paths, SettingsStore
from .events import EventBus
from .hardware import Hardware, detect
from .jobs import JobManager
from .models.hub import HubClient
from .models.local import LocalLibrary
from .models.manager import ModelManager
from .storage import Storage


@dataclass
class Services:
    paths: Paths
    settings: SettingsStore
    hw: Hardware
    bus: EventBus
    hub: HubClient
    local: LocalLibrary
    storage: Storage
    models: ModelManager = field(init=False)
    jobs: JobManager = field(init=False)

    @classmethod
    def create(cls, paths: Paths | None = None, hw: Hardware | None = None) -> Services:
        paths = paths or Paths()
        paths.ensure()
        settings = SettingsStore(paths)
        hw = hw or detect()
        hub = HubClient(settings.hf_token, settings.hf_cache_dir, paths.cache / "hf-manifests")
        local = LocalLibrary(paths.local_index)
        storage = Storage(paths, lambda: settings.settings.image_format)
        svc = cls(paths, settings, hw, EventBus(), hub, local, storage)
        svc.models = ModelManager(hw, settings, hub, local, busy_slots=lambda: svc.jobs.busy_slots())
        from . import tasks  # noqa: F401  (registers task handlers)

        svc.jobs = JobManager(svc, hw.slot_ids(settings.settings.enabled_gpus))
        return svc

    def shutdown(self) -> None:
        self.jobs.stop()
