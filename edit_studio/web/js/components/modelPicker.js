// Model dropdown for one task (txt2img, edit, txt2vid…), grouped into catalog / your files.
import { t, desc } from "../i18n.js";
import { h, clear } from "../ui.js";
import { store, on, modelsFor, findModel } from "../state.js";

/** The memory plan in short pieces: where it runs, compression, text encoder on the CPU, one-card fallback. */
export function planParts(plan) {
  if (!plan) return [];
  const many = plan.mode === "split" || plan.mode === "shard";
  const parts = [many ? t(`plan.${plan.mode}`, { n: plan.gpus_needed }) : t(`plan.${plan.mode}`)];
  const q = plan.main_quant !== undefined ? plan.main_quant : Object.values(plan.quant || {}).sort().pop();
  if (q === "nf4") parts.push(t("plan.nf4"));
  else if (q === "fp8") parts.push(t("plan.fp8"));
  if ((plan.cpu_parts || []).length) parts.push(t("plan.cpu_text"));
  if (plan.gpus && plan.gpus[0] === 1 && plan.gpus[1] > 1) parts.push(t("plan.or_one"));
  return parts;
}

export function planLabel(plan) {
  return planParts(plan).join(" · ");
}

/** One tag per piece (they wrap on narrow screens), all explaining the plan on hover. */
export function planTags(plan) {
  const title = planTitle(plan);
  return planParts(plan).map((text, i) => h("span", { class: i ? "tag" : "tag accent", title }, text));
}

/** Longer explanation for the plan tag's tooltip. */
export function planTitle(plan) {
  if (!plan) return "";
  const bits = [t(`plan.why.${plan.mode}`)];
  if (plan.gpus && plan.gpus[0] > 1) bits.push(t("plan.why.waits"));
  if (plan.alone) bits.push(t("plan.why.alone", { plan: planLabel({ ...plan.alone, gpus: null }) }));
  if ((plan.cpu_parts || []).length) bits.push(t("plan.why.cpu_text"));
  if (plan.text_quant === "nf4" && plan.main_quant !== "nf4") bits.push(t("plan.why.text_nf4"));
  return bits.join("\n");
}

export function modelTags(m) {
  const tags = [];
  if (!m) return tags;
  if (m.source === "catalog") {
    tags.push(m.installed ? h("span", { class: "tag ok" }, "✓ " + t("model.installed"))
      : h("span", { class: "tag warn" }, "⬇ " + t("model.not_installed", { gb: Math.round(m.download_gb) })));
    if (!m.commercial) tags.push(h("span", { class: "tag" }, t("model.noncommercial")));
    if (m.gated) tags.push(h("span", { class: "tag info" }, t("model.gated")));
  } else if (m.source === "hf-cache") {
    tags.push(h("span", { class: "tag ok" }, t("model.hf_cache")));
  } else {
    tags.push(h("span", { class: "tag ok" }, m.source || "local"));
    if (m.dtype && m.dtype !== "?") tags.push(h("span", { class: "tag" }, m.dtype));
    if (m.needs && m.needs.length) tags.push(h("span", { class: "tag warn" }, t("model.needs_base")));
  }
  tags.push(...planTags(m.plan));
  return tags;
}

/**
 * modelPicker(task, value, onChange) -> element with .value / .model
 * `kinds` may restrict to "image"/"video" models.
 */
export function modelPicker(task, value, onChange) {
  const wrap = h("div", { class: "model-picker" });
  let open = false;
  let current = value;

  const close = (e) => { if (!wrap.contains(e.target)) toggle(false); };
  const toggle = (v) => {
    open = v;
    render();
    if (open) setTimeout(() => document.addEventListener("mousedown", close), 0);
    else document.removeEventListener("mousedown", close);
  };

  const choose = (id) => {
    current = id;
    toggle(false);
    onChange(id, findModel(id));
  };

  const option = (m) => h("button", { type: "button", class: `model-opt ${m.id === current ? "on" : ""}`, onclick: () => choose(m.id) },
    h("div", { class: "row wrap" }, h("strong", null, m.name), ...modelTags(m)),
    m.description ? h("div", { class: "desc" }, desc(m.description)) : m.path ? h("div", { class: "desc mono" }, m.path) : null,
  );

  const render = () => {
    const { catalog, yours } = modelsFor(task);
    const all = [...catalog, ...yours];
    if (!all.find((m) => m.id === current) && all.length) {
      // keep the user's choice if models are still loading; otherwise pick the first recommended
      if (store.models) {
        // prefer something already on disk, then the lightest recommended download
        const ready = yours.find((m) => !(m.needs && m.needs.length));
        current = (catalog.find((m) => m.installed && m.recommended) || catalog.find((m) => m.installed) || ready
          || yours[0] || catalog.find((m) => m.recommended) || all[0]).id;
        setTimeout(() => onChange(current, findModel(current)), 0);
      }
    }
    const m = findModel(current);
    clear(wrap);
    wrap.append(h("button", { type: "button", class: "model-current", onclick: () => toggle(!open) },
      h("div", { class: "grow" },
        h("div", { class: "name" }, m ? m.name : (store.models ? t("model.none") : t("common.loading"))),
        m ? h("div", { class: "meta" }, ...modelTags(m)) : null,
      ),
      h("span", { class: "muted" }, open ? "▴" : "▾"),
    ));
    if (open) {
      const menu = h("div", { class: "model-menu" });
      if (catalog.length) menu.append(h("div", { class: "model-group" }, t("model.recommended")), ...catalog.map(option));
      if (yours.length) menu.append(h("div", { class: "model-group" }, t("model.yours")), ...yours.map(option));
      if (!all.length) menu.append(h("div", { class: "empty" }, t("model.none")));
      wrap.append(menu);
    }
  };

  const off = on("models", render);
  render();
  Object.defineProperty(wrap, "value", { get: () => current });
  Object.defineProperty(wrap, "model", { get: () => findModel(current) });
  wrap.dispose = off;
  return wrap;
}
