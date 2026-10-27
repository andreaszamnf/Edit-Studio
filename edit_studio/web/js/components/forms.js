// Reusable form controls shared by the pages.
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, clear, field, seg, slider, toast, errorText } from "../ui.js";
import { store, findModel } from "../state.js";

export function promptBox({ value = "", placeholder = "", onInput, onSubmit, big = true }) {
  const ta = h("textarea", { class: big ? "prompt" : "", placeholder, value, spellcheck: false });
  ta.addEventListener("input", () => onInput(ta.value));
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); onSubmit && onSubmit(); }
  });
  return ta;
}

export function seedField(state) {
  const input = h("input", { type: "number", value: state.seed ?? -1, step: 1 });
  input.addEventListener("input", () => { state.seed = input.value === "" ? -1 : parseInt(input.value, 10); });
  return field(t("field.seed"), h("div", { class: "row" }, input,
    h("button", { type: "button", class: "small", title: t("field.seed_random"), onclick: () => { state.seed = -1; input.value = -1; } }, "🎲")));
}

export function gpuField(state) {
  const sys = store.system;
  if (!sys || sys.hardware.backend !== "cuda" || sys.hardware.gpus.length < 2) return null;
  const sel = h("select", null,
    h("option", { value: "" }, t("common.gpu_auto")),
    ...sys.hardware.gpus.map((g) => h("option", { value: String(g.index) }, `${t("common.gpu_n", { n: g.index })} · ${g.name}`)));
  sel.value = state.gpu === null || state.gpu === undefined ? "" : String(state.gpu);
  sel.addEventListener("change", () => { state.gpu = sel.value === "" ? null : parseInt(sel.value, 10); });
  return field(t("common.gpu"), sel);
}

export function stepsGuidance(state, model, { guidanceMax = 15 } = {}) {
  const d = (model && model.defaults) || {};
  const wrap = h("div", { class: "col" });
  const steps = h("input", { type: "number", min: 1, max: 200, placeholder: d.steps ? String(d.steps) : t("common.auto"), value: state.steps ?? "" });
  steps.addEventListener("input", () => { state.steps = steps.value ? parseInt(steps.value, 10) : null; });
  const guidance = h("input", { type: "number", min: 0, max: guidanceMax, step: 0.1, placeholder: d.guidance !== undefined ? String(d.guidance) : t("common.auto"), value: state.guidance ?? "" });
  guidance.addEventListener("input", () => { state.guidance = guidance.value === "" ? null : parseFloat(guidance.value); });
  wrap.append(h("div", { class: "row" }, h("div", { class: "grow" }, field(t("field.steps"), steps)), h("div", { class: "grow" }, field(t("field.guidance"), guidance))));
  return wrap;
}

const ASPECTS = [
  ["1:1", 1, 1], ["4:3", 4, 3], ["3:4", 3, 4], ["3:2", 3, 2], ["2:3", 2, 3], ["16:9", 16, 9], ["9:16", 9, 16], ["21:9", 21, 9],
];

export function sizeFor(aspect, megapixels, multiple = 16) {
  const a = ASPECTS.find((x) => x[0] === aspect) || ASPECTS[0];
  const ratio = a[1] / a[2];
  const px = megapixels * 1024 * 1024;
  let w = Math.sqrt(px * ratio), hh = w / ratio;
  w = Math.max(multiple, Math.round(w / multiple) * multiple);
  hh = Math.max(multiple, Math.round(hh / multiple) * multiple);
  return [w, hh];
}

export function aspectPicker(state, onChange) {
  const info = h("span", { class: "hint" });
  const update = () => {
    const [w, hh] = sizeFor(state.aspect, state.mp);
    info.textContent = `${w} × ${hh}`;
    onChange && onChange(w, hh);
  };
  const el = h("div", { class: "col" },
    field(t("field.aspect"), seg(ASPECTS.map((a) => ({ value: a[0], label: a[0] })), state.aspect, (v) => { state.aspect = v; update(); })),
    field(t("field.size"), seg([{ value: 1, label: t("field.size_std") }, { value: 2, label: t("field.size_large") }], state.mp, (v) => { state.mp = v; update(); })),
    info,
  );
  update();
  return el;
}

/** LoRA list for the chosen model's family (files found in your ComfyUI/Forge folders). */
export function loraField(state, getModel) {
  const wrap = h("div", { class: "col" });
  const compatible = () => {
    const m = getModel();
    const fam = m && m.family;
    const all = ((store.models && store.models.local) || []).filter((x) => x.kind === "lora" && x.supported);
    return fam ? all.filter((x) => !x.family || fam.startsWith(x.family)) : [];
  };
  const render = () => {
    clear(wrap);
    const list = compatible();
    state.loras = (state.loras || []).filter((l) => list.find((x) => x.id === l.id));
    for (const l of state.loras) {
      const lm = list.find((x) => x.id === l.id);
      wrap.append(h("div", { class: "card", style: { padding: "8px 10px" } },
        h("div", { class: "row" }, h("span", { class: "grow small" }, lm ? lm.name : l.id),
          h("button", { type: "button", class: "ghost small", onclick: () => { state.loras = state.loras.filter((x) => x !== l); render(); } }, "✕")),
        slider({ label: "", min: -1, max: 2, step: 0.05, value: l.weight, onInput: (v) => { l.weight = v; state.loras = [...state.loras]; } }),
      ));
    }
    if (!list.length) { wrap.append(h("span", { class: "hint" }, t("field.no_loras"))); return; }
    const sel = h("select", null, h("option", { value: "" }, t("field.add_lora")),
      ...list.filter((x) => !state.loras.find((l) => l.id === x.id)).map((x) => h("option", { value: x.id }, `${x.name} (${x.source})`)));
    sel.addEventListener("change", () => { if (sel.value) { state.loras = [...state.loras, { id: sel.value, weight: 0.8 }]; render(); } });
    wrap.append(sel);
  };
  render();
  wrap.refresh = render;
  return field(t("field.loras"), wrap);
}

export function modelNote(model) {
  if (!model) return null;
  const notes = [];
  if (model.prompt_note === "english_prompts") notes.push(h("div", { class: "hint warn" }, t("field.english_prompts")));
  if (model.source === "catalog" && !model.installed) {
    notes.push(h("div", { class: "notice info small" }, t("model.auto_download", { gb: Math.round(model.download_gb) })));
  }
  if (model.needs && model.needs.length) notes.push(h("div", { class: "notice info small" }, t("models.needs_base", { repo: model.needs[0] })));
  return notes.length ? h("div", { class: "col" }, notes) : null;
}

export async function submitJob(task, params) {
  try {
    const res = await api.submit(task, params);
    toast(t("common.sent"), "ok", 2000);
    return res;
  } catch (e) {
    toast(errorText(e), "err", 7000);
    return null;
  }
}

export function modelById(id) { return findModel(id); }
