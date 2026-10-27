// Jobs panel, inline result grids and the GPU status chips.
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, clear, progressBar, fmtDuration, errorText, toast } from "../ui.js";
import { store, on } from "../state.js";
import { openViewer, isGif } from "./viewer.js";

const FINISHED = ["done", "failed", "cancelled"];

export function jobTitle(j) {
  const base = t(`task.${j.task}`);
  const p = j.params || {};
  if (j.task === "model.download") return `${base}: ${p.label || p.repo}`;
  if (j.task === "tool.download") return `${base}: ${p.id}`;
  if (p.prompt) return `${base}: ${p.prompt.slice(0, 60)}`;
  return base;
}

export function stageText(j) {
  const stage = (j.stage || "").split(":")[0];
  const label = t(`stage.${stage}`) !== `stage.${stage}` ? t(`stage.${stage}`) : (j.stage || "");
  const bits = [label];
  if (j.status === "queued" && j.depends_on) bits[0] = t("jobs.waiting_for");
  if (j.steps && j.status === "running" && (stage === "generating" || stage === "upscaling" || stage === "processing")) bits.push(t("jobs.step", { a: j.step, b: j.steps }));
  if (j.detail) bits.push(j.detail);
  if (j.eta && j.status === "running") bits.push(t("jobs.remaining", { t: fmtDuration(j.eta) }));
  return bits.filter(Boolean).join(" · ");
}

function rafThrottle(fn) {
  let queued = false;
  return () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; fn(); }); };
}

export function jobCard(j) {
  const running = j.status === "running";
  const card = h("div", { class: `job ${j.status}` },
    h("div", { class: "job-top" },
      h("span", { class: "job-title", title: jobTitle(j) }, jobTitle(j)),
      FINISHED.includes(j.status)
        ? h("button", { class: "ghost icon small", title: t("common.remove"), onclick: () => api.removeJob(j.id) }, "✕")
        : h("button", { class: "ghost small danger", onclick: () => api.cancel(j.id) }, t("jobs.cancel")),
    ),
    h("div", { class: "job-sub" },
      h("span", { class: `tag ${j.status === "done" ? "ok" : j.status === "failed" ? "err" : j.status === "running" ? "accent" : ""}` }, t(`status.${j.status}`)),
      running || j.status === "queued" ? h("span", null, stageText(j)) : null,
      j.slots && j.slots.length && store.system && store.system.hardware.backend === "cuda"
        ? h("span", null, j.slots.length > 1 ? t("jobs.on_gpus", { n: j.slots.join("+") }) : t("jobs.on_gpu", { n: j.slots[0] })) : null,
    ),
    running ? progressBar(j.progress, j.progress <= 0.001) : null,
    j.status === "failed" ? h("div", { class: "err" }, errorText({ key: j.error_key || "error", message: j.error, info: j.error_info }), j.error_key === "error" && j.error ? "\n" + j.error : "") : null,
  );
  if (j.outputs && j.outputs.length && j.status === "done") {
    const thumbs = h("div", { class: "job-thumbs" });
    for (const id of j.outputs.slice(0, 8)) {
      thumbs.append(h("img", { alt: "", dataset: { id }, onclick: () => openViewer(id) }));
    }
    card.append(thumbs);
    // resolve thumbnails lazily
    for (const img of thumbs.querySelectorAll("img")) {
      api.item(img.dataset.id).then((it) => { img.src = it.thumb_url; }).catch(() => img.remove());
    }
  }
  return card;
}

export function mountJobsPanel() {
  const panel = document.getElementById("jobs-panel");
  const body = document.getElementById("jobs-body");
  const badge = document.getElementById("jobs-badge");
  const toggle = document.getElementById("jobs-toggle");
  panel.querySelector(".jobs-toggle-label").textContent = t("jobs.title");
  toggle.onclick = () => panel.classList.toggle("collapsed");

  const cards = new Map(); // job id -> {status, el}
  const render = rafThrottle(() => {
    const jobs = [...store.jobs.values()].sort((a, b) => b.created - a.created);
    const active = jobs.filter((j) => !FINISHED.includes(j.status)).length;
    badge.textContent = active;
    badge.classList.toggle("hidden", active === 0);
    clear(body);
    body.append(h("div", { class: "row between" }, h("strong", null, t("jobs.title")),
      jobs.some((j) => FINISHED.includes(j.status)) ? h("button", { class: "ghost small", onclick: () => api.clearJobs() }, t("jobs.clear")) : null));
    if (!jobs.length) body.append(h("div", { class: "empty" }, t("jobs.empty")));
    for (const j of jobs) {
      const prev = cards.get(j.id);
      // keep finished cards (their thumbnails are already loaded)
      if (prev && FINISHED.includes(j.status) && prev.status === j.status) { body.append(prev.el); continue; }
      const el = jobCard(j);
      cards.set(j.id, { status: j.status, el });
      body.append(el);
    }
  });
  on("jobs", render);
  render();
  return { open: () => panel.classList.remove("collapsed") };
}

/** Grid of recent results for some tasks, with live "in progress" tiles on top. */
export function resultsArea({ tasks, big = false, limit = 24 }) {
  const grid = h("div", { class: `results ${big ? "big" : ""}` });
  let items = [];

  const pendingTile = (j) => h("div", { class: "tile pending" },
    h("strong", null, t(`status.${j.status}`)),
    h("div", { class: "small" }, stageText(j)),
    progressBar(j.progress, j.status !== "running" || j.progress <= 0.001),
    h("button", { class: "small ghost danger", onclick: () => api.cancel(j.id) }, t("jobs.cancel")),
  );
  const tile = (it) => {
    const media = it.kind === "video" && !isGif(it)
      ? h("video", { src: it.url, muted: true, loop: true, preload: "metadata", playsInline: true,
        onmouseenter: (e) => e.target.play().catch(() => {}), onmouseleave: (e) => e.target.pause() })
      : h("img", { src: it.thumb_url, loading: "lazy" });
    return h("div", { class: "tile", onclick: () => openViewer(it, { list: items, onDeleted: load }) },
      media,
      it.kind === "video" ? h("div", { class: "tile-badge" }, h("span", { class: "tag" }, "▶ " + (it.duration ? it.duration.toFixed(1) + "s" : ""))) : null,
      it.favorite ? h("div", { class: "tile-fav" }, "★") : null,
    );
  };
  const render = rafThrottle(() => {
    const pending = [...store.jobs.values()].filter((j) => tasks.includes(j.task) && !FINISHED.includes(j.status))
      .sort((a, b) => b.created - a.created);
    clear(grid);
    for (const j of pending) grid.append(pendingTile(j));
    for (const it of items) grid.append(tile(it));
    if (!pending.length && !items.length) grid.append(h("div", { class: "empty", style: { gridColumn: "1 / -1" } }, h("div", { class: "big-ico" }, "✨"), t("common.no_results")));
  });
  const load = async () => {
    try { items = (await api.gallery({ task: tasks.join(","), limit })).items; } catch (e) { toast(errorText(e), "err"); }
    render();
  };
  const offs = [on("jobs", render), on("gallery", load)];
  load();
  grid.dispose = () => offs.forEach((f) => f());
  return grid;
}

export function mountGpuStatus() {
  const box = document.getElementById("gpu-status");
  const render = rafThrottle(() => {
    clear(box);
    const sys = store.system;
    if (!sys) return;
    const hw = sys.hardware;
    const loaded = sys.loaded || [];
    const running = [...store.jobs.values()].filter((j) => j.status === "running");
    const gpus = hw.backend === "cuda" ? hw.gpus : [{ index: 0, name: hw.backend === "mps" ? "Apple Silicon" : "CPU", total_gb: hw.ram_total_gb }];
    for (const g of gpus) {
      const job = running.find((j) => (j.slots || []).includes(g.index));
      const model = loaded.find((l) => l.slots.includes(g.index));
      box.append(h("div", { class: `gpu-chip ${job ? "busy" : ""}`, title: g.name },
        h("div", { class: "gpu-name" }, h("span", { class: "dot" }), hw.backend === "cuda" ? `GPU ${g.index}` : g.name,
          h("span", { class: "spacer" }), h("span", { class: "gpu-mem faint" }, g.total_gb ? `${Math.round(g.total_gb)} GB` : "")),
        h("div", { class: "gpu-model" }, job ? jobTitle(job) : model ? model.model : "—"),
      ));
    }
  });
  on("system", render);
  on("jobs", render);
  render();
}
