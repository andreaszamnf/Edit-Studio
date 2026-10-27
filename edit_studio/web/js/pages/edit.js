// Edit image: instructions, inpaint, eraser, extend, variations, background, upscale.
import { t } from "../i18n.js";
import { h, field, remember, seg, section, slider, toast, clear, add } from "../ui.js";
import { store } from "../state.js";
import { modelPicker } from "../components/modelPicker.js";
import { mediaPicker, multiImagePicker } from "../components/mediaPicker.js";
import { maskCanvas } from "../components/maskCanvas.js";
import { promptBox, seedField, gpuField, stepsGuidance, loraField, modelNote, submitJob } from "../components/forms.js";
import { resultsArea } from "../components/jobs.js";
import { takeIntent } from "../router.js";

const TOOLS = ["instruct", "inpaint", "erase", "outpaint", "variations", "background", "upscale"];
const EDIT_TASKS = ["image.edit", "image.inpaint", "image.erase", "image.outpaint", "image.img2img", "image.remove_bg", "image.upscale"];
const MULTI = ["qwen_image_edit", "flux2_klein", "qwen_image_21"];
let source = null; // kept while the app is open

export function editPage(root) {
  const state = remember("edit", {
    tool: "instruct", editModel: "", inpaintModel: "", varModel: "",
    prompt: "", inpaintPrompt: "", outPrompt: "", varPrompt: "", strength: null, varStrength: 0.6, count: 1,
    steps: null, guidance: null, seed: -1, gpu: null, loras: [], bgModel: "birefnet", bgMode: "transparent",
    bgColor: "#ffffff", upModel: "realesrgan-x4", upScale: 2, left: 0, right: 0, top: 0, bottom: 0, grow: 6, feather: 10,
  });
  const intent = takeIntent("edit");
  if (intent && intent.image) source = intent.image;
  if (!TOOLS.includes(state.tool)) state.tool = "instruct";

  let refs = [];
  let reference = null;
  const disposers = [];
  const canvas = maskCanvas();
  const tabs = h("div", { class: "tabs" });
  const controls = h("div", { class: "controls" });
  const stageTop = h("div");
  const results = resultsArea({ tasks: EDIT_TASKS });
  disposers.push(results.dispose);

  const srcPicker = mediaPicker({ kind: "image", value: source, compact: true, onChange: (it) => { source = it; renderStage(); } });

  const needsMask = () => state.tool === "inpaint" || state.tool === "erase";

  const renderStage = () => {
    clear(stageTop);
    if (needsMask()) {
      canvas.setImage(source ? source.url : null);
      stageTop.append(h("div", { class: "hint" }, t(state.tool === "erase" ? "edit.erase_hint" : "edit.paint_hint")),
        source ? canvas : h("div", { class: "empty" }, t("edit.no_source")));
    } else if (source) {
      stageTop.append(h("div", { class: "preview-main" }, h("img", { src: source.url })));
    }
  };

  const advanced = (model) => section(t("common.advanced"), [stepsGuidance(state, model), seedField(state), gpuField(state) || ""]);

  const requireSource = () => { if (!source) { toast(t("edit.no_source"), "err"); return false; } return true; };
  const maskId = async () => {
    if (canvas.isEmpty()) { toast(t("edit.no_mask"), "err"); return null; }
    return (await canvas.upload()).id;
  };
  const common = () => ({ seed: state.seed, steps: state.steps, guidance: state.guidance, gpu: state.gpu, count: state.count });

  const countSeg = () => field(t("field.count"), seg([1, 2, 3, 4].map((n) => ({ value: n, label: String(n) })), state.count, (v) => { state.count = v; }));
  const button = (label, fn) => h("div", { class: "sticky-actions" }, h("button", { class: "primary wide", onclick: fn }, label));

  const tools = {
    instruct() {
      let picker;
      const refBox = h("div");
      const notes = h("div");
      const refresh = () => {
        const m = picker.model;
        notes.replaceChildren(modelNote(m) || "");
        clear(refBox);
        if (m && MULTI.includes(m.family)) {
          add(refBox, field(t("picker.extra_refs"), multiImagePicker({ value: refs, onChange: (v) => { refs = v; } })));
        }
      };
      picker = modelPicker("edit", state.editModel, (id) => { state.editModel = id; refresh(); });
      disposers.push(picker.dispose);
      const run = async () => {
        if (!requireSource() || !state.prompt.trim()) return;
        await submitJob("image.edit", { model: state.editModel, images: [source.id, ...refs.map((r) => r.id)], prompt: state.prompt, ...common() });
      };
      const pr = promptBox({ value: state.prompt, placeholder: t("edit.instruct_ph"), onInput: (v) => { state.prompt = v; }, onSubmit: run });
      setTimeout(refresh, 0);
      return [field(t("field.model"), picker), notes, field(t("field.prompt"), pr), refBox, countSeg(), advanced(picker.model), button("🪄 " + t("common.apply"), run)];
    },
    inpaint() {
      let picker;
      const extra = h("div", { class: "col" });
      const notes = h("div");
      const refresh = () => {
        const m = picker.model;
        notes.replaceChildren(modelNote(m) || "");
        clear(extra);
        if (m && m.family === "flux2_klein") {
          add(extra, field(t("edit.reference"), mediaPicker({ kind: "image", value: reference, compact: true, onChange: (it) => { reference = it; } })));
        }
        add(extra, loraField(state, () => picker.model));
      };
      picker = modelPicker("inpaint", state.inpaintModel, (id) => { state.inpaintModel = id; refresh(); });
      disposers.push(picker.dispose);
      const run = async () => {
        if (!requireSource()) return;
        const mask = await maskId();
        if (!mask) return;
        await submitJob("image.inpaint", {
          model: state.inpaintModel, image: source.id, mask, prompt: state.inpaintPrompt, strength: state.strength,
          grow: state.grow, feather: state.feather, reference: reference ? reference.id : null, loras: state.loras, ...common(),
        });
      };
      const pr = promptBox({ value: state.inpaintPrompt, placeholder: t("edit.inpaint_ph"), onInput: (v) => { state.inpaintPrompt = v; }, onSubmit: run, big: false });
      setTimeout(refresh, 0);
      return [
        field(t("field.model"), picker), notes, field(t("field.prompt"), pr),
        slider({ label: t("field.strength"), min: 0.3, max: 1, step: 0.01, value: state.strength ?? 0.95, onInput: (v) => { state.strength = v; } }),
        extra, countSeg(),
        section(t("common.advanced"), [
          slider({ label: "Grow", min: 0, max: 48, value: state.grow, onInput: (v) => { state.grow = v; } }),
          slider({ label: "Feather", min: 0, max: 48, value: state.feather, onInput: (v) => { state.feather = v; } }),
          stepsGuidance(state, picker.model), seedField(state), gpuField(state) || "",
        ]),
        button("🖌️ " + t("common.apply"), run),
      ];
    },
    erase() {
      const run = async () => {
        if (!requireSource()) return;
        const mask = await maskId();
        if (!mask) return;
        await submitJob("image.erase", { image: source.id, mask, gpu: state.gpu });
      };
      return [h("div", { class: "notice info" }, t("edit.erase_hint")), gpuField(state) || "", button("🧽 " + t("common.apply"), run)];
    },
    outpaint() {
      const picker = modelPicker("inpaint", state.inpaintModel, (id) => { state.inpaintModel = id; });
      disposers.push(picker.dispose);
      const num = (key) => {
        const i = h("input", { type: "number", min: 0, max: 4096, step: 16, value: state[key] });
        i.addEventListener("input", () => { state[key] = parseInt(i.value || "0", 10); });
        return i;
      };
      const inputs = { left: num("left"), right: num("right"), top: num("top"), bottom: num("bottom") };
      const toAspect = (r) => {
        if (!source) return;
        const W = source.width, H = source.height;
        let nw = W, nh = H;
        if (W / H < r) nw = Math.round(H * r); else nh = Math.round(W / r);
        const dx = nw - W, dy = nh - H;
        Object.assign(state, { left: Math.floor(dx / 2), right: Math.ceil(dx / 2), top: Math.floor(dy / 2), bottom: Math.ceil(dy / 2) });
        for (const k of Object.keys(inputs)) inputs[k].value = state[k];
      };
      const run = async () => {
        if (!requireSource()) return;
        await submitJob("image.outpaint", { model: state.inpaintModel, image: source.id, prompt: state.outPrompt,
          left: state.left, right: state.right, top: state.top, bottom: state.bottom, ...common() });
      };
      const pr = promptBox({ value: state.outPrompt, placeholder: t("edit.outpaint_ph"), onInput: (v) => { state.outPrompt = v; }, onSubmit: run, big: false });
      return [
        field(t("field.model"), picker), field(t("field.prompt"), pr),
        field(t("edit.expand"), h("div", { class: "col" },
          h("div", { class: "row" }, field(t("edit.left"), inputs.left), field(t("edit.right"), inputs.right)),
          h("div", { class: "row" }, field(t("edit.top"), inputs.top), field(t("edit.bottom"), inputs.bottom)))),
        field(t("edit.to_aspect"), seg(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"].map((a) => ({ value: a, label: a })), "", (v) => {
          const [a, b] = v.split(":").map(Number); toAspect(a / b);
        })),
        countSeg(), advanced(picker.model), button("↔ " + t("common.apply"), run),
      ];
    },
    variations() {
      const picker = modelPicker("img2img", state.varModel, (id) => { state.varModel = id; });
      disposers.push(picker.dispose);
      const run = async () => {
        if (!requireSource()) return;
        await submitJob("image.img2img", { model: state.varModel, image: source.id, prompt: state.varPrompt, strength: state.varStrength, loras: state.loras, ...common() });
      };
      const pr = promptBox({ value: state.varPrompt, placeholder: t("edit.variations_ph"), onInput: (v) => { state.varPrompt = v; }, onSubmit: run, big: false });
      return [
        field(t("field.model"), picker), field(t("field.prompt"), pr),
        slider({ label: t("field.strength"), min: 0.1, max: 1, step: 0.01, value: state.varStrength, onInput: (v) => { state.varStrength = v; } }),
        loraField(state, () => picker.model), countSeg(), advanced(picker.model), button("🎲 " + t("common.generate"), run),
      ];
    },
    background() {
      const color = h("input", { type: "color", value: state.bgColor, style: { width: "48px", height: "34px", padding: "2px" } });
      color.addEventListener("input", () => { state.bgColor = color.value; });
      const run = async () => {
        if (!requireSource()) return;
        await submitJob("image.remove_bg", { image: source.id, model: state.bgModel, mode: state.bgMode, color: state.bgColor });
      };
      return [
        field(t("edit.bg_mode"), seg([
          { value: "transparent", label: t("edit.bg_transparent") }, { value: "color", label: t("edit.bg_color") }, { value: "blur", label: t("edit.bg_blur") },
        ], state.bgMode, (v) => { state.bgMode = v; })),
        field(t("edit.bg_color"), color),
        field(t("edit.bg_model"), seg([
          { value: "birefnet", label: "BiRefNet" }, { value: "birefnet-lite", label: "Lite" }, { value: "isnet", label: "ISNet" },
        ], state.bgModel, (v) => { state.bgModel = v; })),
        button("✂️ " + t("common.apply"), run),
      ];
    },
    upscale() {
      const sel = h("select");
      const tools = (store.models && store.models.tools) || [];
      for (const tm of tools.filter((x) => x.kind === "upscaler")) sel.append(h("option", { value: tm.id }, `${tm.name}${tm.installed ? "" : " ⬇"}`));
      for (const lm of ((store.models && store.models.local) || []).filter((x) => x.kind === "upscaler")) sel.append(h("option", { value: lm.id }, `${lm.name} (${lm.source})`));
      sel.value = state.upModel;
      sel.addEventListener("change", () => { state.upModel = sel.value; });
      const run = async () => {
        if (!requireSource()) return;
        await submitJob("image.upscale", { image: source.id, model: state.upModel, scale: state.upScale, gpu: state.gpu });
      };
      return [
        field(t("edit.upscale_model"), sel),
        field(t("edit.scale"), seg([2, 3, 4].map((n) => ({ value: n, label: `×${n}` })), state.upScale, (v) => { state.upScale = v; })),
        source ? h("div", { class: "hint" }, `${source.width}×${source.height} → ${source.width * state.upScale}×${source.height * state.upScale}`) : null,
        gpuField(state) || "", button("🔍 " + t("common.apply"), run),
      ];
    },
  };

  const renderControls = () => {
    clear(controls);
    add(controls, field(t("edit.source"), srcPicker), ...tools[state.tool]());
  };
  const renderTabs = () => {
    clear(tabs);
    for (const tool of TOOLS) {
      tabs.append(h("button", { class: state.tool === tool ? "on" : "", onclick: () => { state.tool = tool; renderTabs(); renderControls(); renderStage(); } }, t(`edit.tab.${tool}`)));
    }
  };

  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("edit.title"))),
    tabs,
    h("div", { class: "workspace" }, controls, h("div", { class: "stage" }, stageTop, h("h2", { class: "section-title" }, t("common.results")), results)),
  );
  renderTabs();
  renderControls();
  renderStage();
  return () => disposers.forEach((d) => d && d());
}
