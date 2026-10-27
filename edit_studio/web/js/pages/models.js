// Models: hardware overview, downloadable catalog, your own files, tools, folders.
import { t, desc } from "../i18n.js";
import { api } from "../api.js";
import { h, clear, toast, errorText, confirmBox, fmtGB, add } from "../ui.js";
import { store, on, loadModels, loadSystem } from "../state.js";
import { planLabel, planTags } from "../components/modelPicker.js";

export function modelsPage(root) {
  const body = h("div", { class: "page-body" });
  const offs = [];

  const hardware = () => {
    const sys = store.system;
    if (!sys) return h("div");
    const hw = sys.hardware;
    const items = [];
    if (hw.backend === "cuda") {
      for (const g of hw.gpus) {
        items.push(h("div", { class: "hw-item" }, h("div", { class: "faint small" }, `GPU ${g.index}`), h("div", { class: "big" }, g.name),
          h("div", { class: "muted small" }, t("models.vram", { total: g.total_gb, free: g.usable_gb }))));
      }
    } else {
      items.push(h("div", { class: "notice warn" }, hw.backend === "mps" ? "Apple Silicon (MPS)" : t("models.no_gpu")));
    }
    items.push(h("div", { class: "hw-item" }, h("div", { class: "faint small" }, t("models.ram")), h("div", { class: "big" }, `${Math.round(hw.ram_total_gb)} GB`),
      h("div", { class: "muted small" }, `PyTorch ${hw.torch}${hw.cuda ? " · CUDA " + hw.cuda : ""}`)));
    items.push(h("div", { class: "hw-item" }, h("div", { class: "faint small" }, "bitsandbytes"),
      h("div", { class: sys.nf4 ? "tag ok" : "tag warn" }, sys.nf4 ? t("models.nf4_ok") : t("models.nf4_missing"))));
    return h("div", { class: "hw-grid" }, items);
  };

  const download = async (id) => {
    try { await api.download(id); toast(t("common.sent"), "ok"); } catch (e) { toast(errorText(e), "err"); }
  };
  const remove = async (id) => {
    if (!(await confirmBox(t("models.delete_confirm")))) return;
    try { await api.deleteModel(id); await loadModels(true); } catch (e) { toast(errorText(e), "err"); }
  };

  const catalogCard = (m) => h("div", { class: "card model-card" },
    h("div", { class: "top" }, h("h3", null, m.name), m.recommended ? h("span", { class: "tag accent" }, "★") : null),
    h("div", { class: "muted small" }, desc(m.description)),
    h("div", { class: "tags" },
      h("span", { class: "tag" }, m.license),
      m.gated ? h("span", { class: "tag info" }, t("model.gated")) : null,
      ...planTags(m.plan),
      ...(m.tags || []).map((x) => h("span", { class: "tag" }, t(`tag.${x}`)))),
    h("div", { class: "foot" },
      m.installed ? h("span", { class: "tag ok" }, "✓ " + t("models.on_disk", { gb: m.on_disk_gb })) : h("span", { class: "muted small" }, `~${fmtGB(m.download_gb)}`),
      h("span", { class: "spacer" }),
      m.installed ? h("button", { class: "small danger", onclick: () => remove(m.id) }, t("common.delete"))
        : h("button", { class: "small", onclick: () => download(m.id) }, "⬇ " + t("models.download"))),
  );

  const localTable = (list) => {
    if (!list.length) return h("div", { class: "notice" }, t("models.no_local"));
    const rows = list.map((m) => h("tr", null,
      h("td", null, h("div", null, h("strong", null, m.name)), h("div", { class: "faint small mono" }, m.path)),
      h("td", null, t(`kind.${m.kind}`)),
      h("td", null, m.family || "—", m.variant ? h("span", { class: "faint" }, ` · ${m.variant}`) : null),
      h("td", null, m.dtype && m.dtype !== "?" ? m.dtype : ""),
      h("td", null, fmtGB(m.size / 1e9)),
      h("td", null, m.supported
        ? (m.needs && m.needs.length ? h("span", { class: "tag warn", title: m.needs[0] }, t("model.needs_base"))
          : h("div", { class: "tags" }, h("span", { class: "tag ok" }, t("models.ok")), ...planTags(m.plan)))
        : h("span", { class: "tag", title: m.note }, t(`note.${(m.note || "unknown").split(":")[0]}`) !== `note.${(m.note || "unknown").split(":")[0]}` ? t(`note.${(m.note || "unknown").split(":")[0]}`) : m.note)),
      h("td", null, m.supported && m.needs && m.needs.length ? h("button", { class: "small", onclick: () => download(m.id) }, "⬇") : null),
    ));
    return h("table", { class: "list" },
      h("thead", null, h("tr", null, ...["models.file", "models.kind", "models.family", "", "models.size", "models.status", ""].map((k) => h("th", null, k ? t(k) : "")))),
      h("tbody", null, rows));
  };

  const foldersBox = () => {
    const s = store.system ? store.system.settings : { model_folders: [], auto_detect_folders: true };
    const input = h("input", { type: "text", placeholder: t("models.folder_ph") });
    const save = async (folders, auto = s.auto_detect_folders) => {
      try {
        await api.saveSettings({ model_folders: folders, auto_detect_folders: auto });
        await loadSystem();
        await api.scan();
        toast(t("stage.scanning"), "ok");
      } catch (e) { toast(errorText(e), "err"); }
    };
    const roots = (store.models && store.models.roots) || [];
    return h("div", { class: "card col" },
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: s.auto_detect_folders, onchange: (e) => save(s.model_folders, e.target.checked) }), t("models.auto_detect")),
      roots.length ? h("div", { class: "col small" }, ...roots.map((r) => h("div", { class: "row" }, h("span", { class: "tag ok" }, r.source), h("span", { class: "mono" }, r.path)))) : null,
      ...(s.model_folders || []).map((f) => h("div", { class: "row" }, h("span", { class: "mono grow" }, f),
        h("button", { class: "ghost small", onclick: () => save(s.model_folders.filter((x) => x !== f)) }, "✕"))),
      h("div", { class: "row" }, input, h("button", { onclick: () => { if (input.value.trim()) save([...(s.model_folders || []), input.value.trim()]); } }, t("models.add_folder"))),
    );
  };

  const render = () => {
    clear(body);
    const m = store.models;
    add(body, h("h2", { class: "section-title" }, t("models.hardware")), hardware());
    if (!m) { add(body, h("div", { class: "empty" }, t("common.loading"))); return; }
    const loaded = m.loaded || [];
    add(body, 
      h("h2", { class: "section-title" }, t("models.loaded"), h("span", { class: "spacer" }),
        loaded.length ? h("button", { class: "small", onclick: async () => { await api.unload(); await loadSystem(); await loadModels(true); } }, t("models.unload")) : null),
      loaded.length ? h("div", { class: "col" }, ...loaded.map((l) => h("div", { class: "row card", style: { padding: "8px 12px" } },
        h("strong", null, l.model), h("span", { class: "tag accent" }, planLabel(l.plan)), h("span", { class: "muted small" }, l.slots.map((s) => `GPU ${s}`).join(" + ")))))
        : h("div", { class: "muted" }, t("models.none_loaded")),
      h("h2", { class: "section-title" }, t("models.catalog")),
      h("div", { class: "hint", style: { marginBottom: "10px" } }, t("models.catalog_hint")),
      h("h3", { class: "muted" }, t("models.images")),
      h("div", { class: "cards" }, m.catalog.filter((x) => x.kind === "image").map(catalogCard)),
      h("h3", { class: "muted", style: { marginTop: "16px" } }, t("models.videos")),
      h("div", { class: "cards" }, m.catalog.filter((x) => x.kind === "video").map(catalogCard)),
      h("h2", { class: "section-title" }, t("models.local"), h("span", { class: "spacer" }),
        h("button", { class: "small", onclick: async () => { await api.scan(); toast(t("stage.scanning"), "ok"); } }, "🔎 " + t("models.rescan"))),
      h("div", { class: "hint", style: { marginBottom: "10px" } }, t("models.local_hint")),
      m.hf_cache.length ? h("div", { class: "card col", style: { marginBottom: "10px" } }, h("strong", null, t("model.hf_cache")),
        ...m.hf_cache.map((x) => h("div", { class: "row" }, h("span", { class: "grow" }, x.repo), h("span", { class: "tag" }, x.family), h("span", { class: "muted small" }, fmtGB(x.on_disk_gb))))) : null,
      localTable(m.local.filter((x) => x.kind !== "upscaler")),
      h("h2", { class: "section-title" }, t("models.folders")),
      foldersBox(),
      h("h2", { class: "section-title" }, t("models.tools")),
      h("div", { class: "cards" }, ...m.tools.map((tm) => h("div", { class: "card row" },
        h("div", { class: "grow" }, h("strong", null, tm.name), h("div", { class: "muted small" }, `${tm.size_mb} MB`)),
        tm.installed ? h("span", { class: "tag ok" }, "✓") : h("button", { class: "small", onclick: () => download(tm.id) }, "⬇"),
        tm.installed ? h("button", { class: "ghost small danger", onclick: () => remove(tm.id) }, "✕") : null))),
      m.local.some((x) => x.kind === "upscaler") ? localTable(m.local.filter((x) => x.kind === "upscaler")) : null,
    );
  };

  add(root, h("div", { class: "page-head" }, h("h1", null, t("models.title"))), body);
  offs.push(on("models", render), on("system", render));
  render();
  loadModels(true).catch((e) => toast(errorText(e), "err"));
  return () => offs.forEach((f) => f());
}
