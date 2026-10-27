// Create video: text-to-video or animate a photo.
import { t } from "../i18n.js";
import { h, field, remember, seg, section, slider, toast, clear, add } from "../ui.js";
import { modelPicker } from "../components/modelPicker.js";
import { mediaPicker } from "../components/mediaPicker.js";
import { promptBox, seedField, gpuField, stepsGuidance, modelNote, submitJob } from "../components/forms.js";
import { resultsArea } from "../components/jobs.js";
import { takeIntent } from "../router.js";

let startImage = null;

export function videoPage(root) {
  const state = remember("video", {
    mode: "text", t2vModel: "wan2.2-ti2v-5b", i2vModel: "wan2.2-ti2v-5b", prompt: "", negative_prompt: "",
    orientation: "landscape", seconds: 5, quality: "normal", steps: null, guidance: null, seed: -1, gpu: null,
  });
  const intent = takeIntent("video");
  if (intent && intent.image) { startImage = intent.image; state.mode = "image"; }
  if (intent && intent.params) {
    const p = intent.params;
    Object.assign(state, { prompt: p.prompt || "", negative_prompt: p.negative_prompt || "", seed: p.seed ?? -1 });
    if (p.model) state[p.image ? "i2vModel" : "t2vModel"] = p.model;
  }

  const controls = h("div", { class: "controls" });
  const results = resultsArea({ tasks: ["video.generate"], big: true, limit: 12 });
  const disposers = [results.dispose];

  const render = () => {
    clear(controls);
    const isImage = state.mode === "image";
    const notes = h("div");
    const adv = h("div", { class: "col" });
    let picker;
    const refresh = () => {
      const m = picker.model;
      notes.replaceChildren(modelNote(m) || "");
      adv.replaceChildren(stepsGuidance(state, m, { guidanceMax: 20 }), seedField(state), gpuField(state) || "");
    };
    picker = modelPicker(isImage ? "img2vid" : "txt2vid", isImage ? state.i2vModel : state.t2vModel, (id) => {
      state[isImage ? "i2vModel" : "t2vModel"] = id;
      refresh();
    });
    disposers.push(picker.dispose);

    const run = async () => {
      if (isImage && !startImage) return toast(t("edit.no_source"), "err");
      if (!state.prompt.trim()) return pr.focus();
      const m = picker.model;
      const d = (m && m.defaults) || {};
      const fps = d.fps || 16;
      let w = d.width || 832, hh = d.height || 480;
      if (!isImage && state.orientation === "portrait") [w, hh] = [hh, w];
      if (isImage) { w = 0; hh = 0; } // follow the photo's orientation
      const frames = Math.round(state.seconds * fps) + 1;
      const base = d.steps || 30;
      const steps = state.steps || (state.quality === "fast" ? Math.max(10, Math.round(base * 0.6)) : state.quality === "best" ? Math.round(base * 1.25) : base);
      await submitJob("video.generate", {
        model: picker.value, prompt: state.prompt, negative_prompt: state.negative_prompt, image: isImage ? startImage.id : null,
        width: w, height: hh, frames, fps, steps, guidance: state.guidance, seed: state.seed, gpu: state.gpu,
      });
    };
    const pr = promptBox({ value: state.prompt, placeholder: t(isImage ? "video.i2v_ph" : "video.prompt_ph"), onInput: (v) => { state.prompt = v; }, onSubmit: run });
    const neg = promptBox({ value: state.negative_prompt, placeholder: t("field.negative_ph"), onInput: (v) => { state.negative_prompt = v; }, big: false });

    add(controls, 
      seg([{ value: "text", label: "✍️ " + t("video.mode_text") }, { value: "image", label: "🖼️ " + t("video.mode_image") }], state.mode,
        (v) => { state.mode = v; render(); }, { full: true }),
      isImage ? field(t("video.start_image"), mediaPicker({ kind: "image", value: startImage, onChange: (it) => { startImage = it; } })) : null,
      field(t("field.model"), picker),
      notes,
      field(t("field.prompt"), pr),
      !isImage ? field(t("field.orientation"), seg([
        { value: "landscape", label: "▭ " + t("field.landscape") }, { value: "portrait", label: "▯ " + t("field.portrait") },
      ], state.orientation, (v) => { state.orientation = v; })) : null,
      slider({ label: t("field.duration"), min: 2, max: 10, step: 1, value: state.seconds, format: (v) => `${v} ${t("common.seconds")}`, onInput: (v) => { state.seconds = v; } }),
      field(t("field.quality"), seg([
        { value: "fast", label: t("field.speed_fast") }, { value: "normal", label: t("field.speed_normal") }, { value: "best", label: t("field.speed_best") },
      ], state.quality, (v) => { state.quality = v; })),
      section(t("common.advanced"), [field(t("field.negative"), neg), adv]),
      h("div", { class: "hint" }, t("video.time_note")),
      h("div", { class: "sticky-actions" }, h("button", { class: "primary wide", onclick: run }, "🎬 " + t("video.button"))),
    );
    refresh();
  };

  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("video.title"))),
    h("div", { class: "workspace" }, controls, h("div", { class: "stage" }, results)),
  );
  render();
  return () => disposers.forEach((d) => d && d());
}
