// Shared app state fed by the API and live server events.
import { api } from "./api.js";

const listeners = new Map();

export const store = {
  system: null,
  models: null,
  jobs: new Map(),
  connected: true,
};

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, data) {
  for (const fn of listeners.get(event) || []) {
    try { fn(data); } catch (e) { console.error(e); }
  }
}

export async function loadSystem() {
  store.system = await api.system();
  emit("system", store.system);
  return store.system;
}

let modelsPromise = null;
export function loadModels(refresh = false) {
  if (!modelsPromise || refresh) {
    modelsPromise = api.models(refresh).then((m) => { store.models = m; emit("models", m); return m; })
      .catch((e) => { modelsPromise = null; throw e; });
  }
  return modelsPromise;
}

export async function loadJobs() {
  const jobs = await api.jobs();
  store.jobs = new Map(jobs.map((j) => [j.id, j]));
  emit("jobs", store.jobs);
}

export function handleEvent(evt) {
  if (evt.type === "job") {
    const prev = store.jobs.get(evt.data.id);
    store.jobs.set(evt.data.id, evt.data);
    emit("job", evt.data);
    emit("jobs", store.jobs);
    if (prev && prev.status !== evt.data.status && ["done", "failed", "cancelled"].includes(evt.data.status)) {
      emit("job-finished", evt.data);
    }
  } else if (evt.type === "job_removed") {
    store.jobs.delete(evt.data.id);
    emit("jobs", store.jobs);
  } else if (evt.type === "gallery") {
    emit("gallery", evt.data);
  } else if (evt.type === "models") {
    loadModels(true).catch(() => {});
    loadSystem().catch(() => {});
  }
}

// All models able to do `task` (txt2img, img2img, inpaint, edit, txt2vid, img2vid, vid2vid).
export function modelsFor(task) {
  const m = store.models;
  if (!m) return { catalog: [], yours: [] };
  const catalog = m.catalog.filter((x) => x.tasks.includes(task));
  const yours = [
    ...m.hf_cache.filter((x) => x.tasks.includes(task)),
    ...m.local.filter((x) => (x.kind === "checkpoint" || x.kind === "denoiser") && x.supported && (x.tasks || []).includes(task)),
  ];
  return { catalog, yours };
}

export function findModel(id) {
  const m = store.models;
  if (!m) return null;
  return m.catalog.find((x) => x.id === id) || m.hf_cache.find((x) => x.id === id) || m.local.find((x) => x.id === id) || null;
}
