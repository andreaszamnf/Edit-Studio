// Thin client for the local Edit Studio server.

const HEADERS = { "X-Edit-Studio": "1" };

export class ApiError extends Error {
  constructor(status, body) {
    super((body && (body.message || body.detail || body.error)) || `HTTP ${status}`);
    this.status = status;
    this.body = body || {};
    this.key = this.body.error || "error";
  }
}

async function request(method, url, body) {
  const opts = { method, headers: { ...HEADERS } };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = { message: text }; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  get: (url) => request("GET", url),
  post: (url, body = {}) => request("POST", url, body),
  put: (url, body = {}) => request("PUT", url, body),
  patch: (url, body = {}) => request("PATCH", url, body),
  del: (url) => request("DELETE", url),

  system: () => api.get("/api/system"),
  models: (refresh = false) => api.get("/api/models" + (refresh ? "?refresh=1" : "")),
  jobs: () => api.get("/api/jobs"),
  submit: (task, params, autoDownload = true) => api.post("/api/jobs", { task, params, auto_download: autoDownload }),
  cancel: (id) => api.post(`/api/jobs/${id}/cancel`),
  removeJob: (id) => api.del(`/api/jobs/${id}`),
  clearJobs: () => api.post("/api/jobs/clear"),
  gallery: (q = {}) => api.get("/api/gallery?" + new URLSearchParams(q)),
  item: (id) => api.get(`/api/gallery/${encodeURIComponent(id)}`),
  setFavorite: (id, favorite) => api.patch(`/api/gallery/${encodeURIComponent(id)}`, { favorite }),
  deleteItem: (id) => api.del(`/api/gallery/${encodeURIComponent(id)}`),
  settings: () => api.get("/api/settings"),
  saveSettings: (changes) => api.put("/api/settings", changes),
  open: (target) => api.post("/api/open", { target }),
  download: (model) => api.post("/api/models/download", { model }),
  deleteModel: (id) => api.del(`/api/models/${encodeURIComponent(id)}`),
  scan: () => api.post("/api/models/scan"),
  unload: () => api.post("/api/models/unload"),

  upload(file, { hidden = false } = {}) {
    const fd = new FormData();
    fd.append("file", file, file.name || "upload.png");
    fd.append("hidden", hidden ? "true" : "false");
    return request("POST", "/api/uploads", fd);
  },
};

// Server-sent events with automatic reconnect.
export function connectEvents(onEvent, onState) {
  let es;
  let retry = 1000;
  const open = () => {
    es = new EventSource("/api/events");
    es.onopen = () => { retry = 1000; onState && onState(true); };
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
    };
    es.onerror = () => {
      onState && onState(false);
      es.close();
      setTimeout(open, retry);
      retry = Math.min(retry * 2, 10000);
    };
  };
  open();
  return () => es && es.close();
}
