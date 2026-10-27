// Settings.
import { t, getLang, setLang } from "../i18n.js";
import { api } from "../api.js";
import { h, field, seg, toast, errorText, slider, add } from "../ui.js";
import { store, loadSystem, loadModels } from "../state.js";
import { start } from "../router.js";

export function settingsPage(root) {
  const s = { ...(store.system ? store.system.settings : {}) };
  const save = async (changes) => {
    try {
      const res = await api.saveSettings(changes);
      Object.assign(s, res);
      toast(res.restart_needed ? t("settings.restart") : t("settings.saved"), "ok");
      await loadSystem();
      loadModels(true).catch(() => {});
    } catch (e) { toast(errorText(e), "err"); }
  };

  const token = h("input", { type: "password", placeholder: s.hf_token_set ? "••••••••  (" + t("settings.hf_token_set") + ")" : t("settings.hf_token_ph") });
  const cache = h("input", { type: "text", placeholder: t("settings.hf_cache_ph"), value: s.hf_cache_dir || "" });
  const select = (key, options) => {
    const el = h("select", null, ...options.map((o) => h("option", { value: o }, t(`settings.${key === "memory_mode" ? "mem" : "comp"}.${o}`))));
    el.value = s[key];
    el.addEventListener("change", () => save({ [key]: el.value }));
    return el;
  };
  const hw = store.system ? store.system.hardware : { gpus: [], backend: "cpu" };
  const gpuBoxes = hw.backend === "cuda" && hw.gpus.length > 1 ? h("div", { class: "col" }, ...hw.gpus.map((g) => {
    const enabled = !s.enabled_gpus || s.enabled_gpus.includes(g.index);
    return h("label", { class: "check" }, h("input", { type: "checkbox", checked: enabled, onchange: (e) => {
      const cur = new Set(s.enabled_gpus || hw.gpus.map((x) => x.index));
      if (e.target.checked) cur.add(g.index); else cur.delete(g.index);
      const list = [...cur].sort();
      save({ enabled_gpus: list.length === hw.gpus.length ? null : list });
    } }), `GPU ${g.index} · ${g.name} (${g.total_gb} GB)`);
  })) : null;

  const card = (title, ...children) => h("div", { class: "card col", style: { gap: "12px" } }, h("h3", null, title), ...children);
  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("settings.title"))),
    h("div", { class: "page-body" }, h("div", { class: "cards", style: { gridTemplateColumns: "repeat(auto-fill, minmax(380px, 1fr))" } },
      card(t("settings.language"), seg([{ value: "el", label: "Ελληνικά" }, { value: "en", label: "English" }], getLang(), (v) => {
        setLang(v); save({ language: v }); start();
      })),
      card(t("settings.gpus"),
        gpuBoxes ? field(t("settings.gpus"), gpuBoxes, t("settings.gpus_hint")) : null,
        field(t("settings.memory"), select("memory_mode", ["auto", "single", "split", "offload", "lowvram"])),
        field(t("settings.compression"), select("compression", ["auto", "none", "fp8", "nf4"])),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: s.keep_models_loaded, onchange: (e) => save({ keep_models_loaded: e.target.checked }) }), t("settings.keep_loaded")),
      ),
      card(t("settings.hf"),
        field(t("settings.hf_token"), h("div", { class: "row" }, token, h("button", { onclick: () => { save({ hf_token: token.value }); token.value = ""; } }, t("common.save")))),
        h("a", { href: "https://huggingface.co/settings/tokens", target: "_blank", rel: "noreferrer", class: "small" }, "huggingface.co/settings/tokens"),
        field(t("settings.hf_cache"), h("div", { class: "row" }, cache, h("button", { onclick: () => save({ hf_cache_dir: cache.value }) }, t("common.save"))), t("settings.hf_cache_hint")),
        h("button", { class: "small", onclick: () => api.open("hf_cache") }, "📂 " + t("common.open_folder")),
      ),
      card(t("settings.output"),
        field(t("settings.image_format"), seg(["png", "webp", "jpg"].map((f) => ({ value: f, label: f.toUpperCase() })), s.image_format, (v) => save({ image_format: v }))),
        slider({ label: t("settings.video_quality"), min: 12, max: 30, value: s.video_quality ?? 18, onInput: debounceSave((v) => save({ video_quality: v })) }),
      ),
      card(t("settings.folders"),
        h("div", { class: "kv" }, h("dt", null, t("settings.data_dir")), h("dd", { class: "mono" }, store.system ? store.system.data_dir : "")),
        h("div", { class: "row wrap" },
          h("button", { class: "small", onclick: () => api.open("outputs") }, "📂 " + t("gallery.title")),
          h("button", { class: "small", onclick: () => api.open("home") }, "📂 " + t("settings.data_dir"))),
      ),
      card(t("settings.about"), h("div", { class: "muted small" }, `Edit Studio ${store.system ? store.system.version : ""}`),
        h("div", { class: "muted small" }, t("settings.logs_hint")),
        h("div", { class: "row wrap" }, h("button", { class: "small", onclick: () => api.open("logs") }, "📂 " + t("settings.logs")))),
    )),
  );
}

function debounceSave(fn) {
  let id;
  return (v) => { clearTimeout(id); id = setTimeout(() => fn(v), 500); };
}
