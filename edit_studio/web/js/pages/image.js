// Create image (text-to-image).
import { t } from "../i18n.js";
import { h, field, remember, seg, section, add } from "../ui.js";
import { modelPicker } from "../components/modelPicker.js";
import { promptBox, seedField, gpuField, stepsGuidance, aspectPicker, loraField, modelNote, submitJob } from "../components/forms.js";
import { resultsArea } from "../components/jobs.js";
import { takeIntent } from "../router.js";

export function imagePage(root) {
  const state = remember("image", {
    model: "z-image-turbo", prompt: "", negative_prompt: "", aspect: "1:1", mp: 1, count: 1,
    steps: null, guidance: null, seed: -1, loras: [], gpu: null,
  });
  const intent = takeIntent("image");
  if (intent && intent.params) {
    const p = intent.params;
    Object.assign(state, { model: p.model || state.model, prompt: p.prompt || "", negative_prompt: p.negative_prompt || "",
      steps: p.steps ?? null, guidance: p.guidance ?? null, seed: p.seed ?? -1 });
  }
  let size = [1024, 1024];

  const notes = h("div");
  const advanced = h("div", { class: "col" });
  let picker;
  const run = async () => {
    if (!state.prompt.trim()) return promptEl.focus();
    await submitJob("image.generate", {
      model: state.model, prompt: state.prompt, negative_prompt: state.negative_prompt,
      width: size[0], height: size[1], count: state.count, steps: state.steps, guidance: state.guidance,
      seed: state.seed, loras: state.loras, gpu: state.gpu,
    });
  };
  const promptEl = promptBox({ value: state.prompt, placeholder: t("field.prompt_ph"), onInput: (v) => { state.prompt = v; }, onSubmit: run });
  const negEl = promptBox({ value: state.negative_prompt, placeholder: t("field.negative_ph"), onInput: (v) => { state.negative_prompt = v; }, big: false });
  const negField = field(t("field.negative"), negEl);

  const refreshModel = () => {
    const m = picker.model;
    notes.replaceChildren(modelNote(m) || "");
    negField.classList.toggle("hidden", !!m && m.negative_prompt === false);
    advanced.replaceChildren(stepsGuidance(state, m), seedField(state), loraField(state, () => picker.model), gpuField(state) || "");
  };
  picker = modelPicker("txt2img", state.model, (id) => { state.model = id; refreshModel(); });

  const results = resultsArea({ tasks: ["image.generate"] });
  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("image.title"))),
    h("div", { class: "workspace" },
      h("div", { class: "controls" },
        field(t("field.model"), picker),
        notes,
        field(t("field.prompt"), promptEl),
        negField,
        aspectPicker(state, (w, hh) => { size = [w, hh]; }),
        field(t("field.count"), seg([1, 2, 3, 4].map((n) => ({ value: n, label: String(n) })), state.count, (v) => { state.count = v; })),
        section(t("common.advanced"), advanced),
        h("div", { class: "sticky-actions" }, h("button", { class: "primary wide", onclick: run }, "✨ " + t("image.button"), h("kbd", null, "Ctrl+Enter"))),
      ),
      h("div", { class: "stage" }, results),
    ),
  );
  refreshModel();
  promptEl.focus();
  return () => { picker.dispose(); results.dispose(); };
}
