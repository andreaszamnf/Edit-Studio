// Edit video: cut/speed, AI restyle, AI continue, AI upscale, background, smooth motion, GIF, frames, join.
import { t } from "../i18n.js";
import { h, field, remember, seg, section, slider, toast, clear, add } from "../ui.js";
import { store } from "../state.js";
import { modelPicker } from "../components/modelPicker.js";
import { mediaPicker, galleryChooser } from "../components/mediaPicker.js";
import { promptBox, seedField, gpuField, stepsGuidance, modelNote, submitJob } from "../components/forms.js";
import { resultsArea } from "../components/jobs.js";
import { takeIntent } from "../router.js";

const TOOLS = ["cut", "restyle", "extend", "upscale", "background", "smooth", "gif", "frame", "join"];
const TASKS = ["video.transform", "video.restyle", "video.extend", "video.upscale", "video.remove_bg", "video.interpolate",
  "video.to_gif", "video.extract_frame", "video.concat"];
let source = null;
let joinList = [];

export function veditPage(root) {
  const state = remember("vedit", {
    tool: "cut", start: 0, end: 0, speed: 1, reverse: false, mute: false, flip: false, rotate: 0, aspect: "", outHeight: 0,
    restyleModel: "", restylePrompt: "", strength: 0.7, maxSeconds: 5,
    extendModel: "", extendPrompt: "", extendSeconds: 5,
    upModel: "realesr-general-x4", upScale: 2, bgModel: "birefnet-lite", bgMode: "color", bgColor: "#00ff00",
    factor: 2, slowmo: false, gifFps: 12, gifWidth: 480, steps: null, guidance: null, seed: -1, gpu: null,
  });
  const intent = takeIntent("vedit");
  if (intent && intent.video) source = intent.video;
  if (!TOOLS.includes(state.tool)) state.tool = "cut";

  const tabs = h("div", { class: "tabs" });
  const controls = h("div", { class: "controls" });
  const results = resultsArea({ tasks: TASKS, big: true, limit: 18 });
  const disposers = [results.dispose];
  const srcPicker = mediaPicker({ kind: "video", value: source, onChange: (it) => { source = it; renderControls(); } });
  const player = () => srcPicker.querySelector("video");
  const now = () => { const v = player(); return v ? Math.round(v.currentTime * 100) / 100 : 0; };

  const need = () => { if (!source) { toast(t("vedit.no_source"), "err"); return false; } return true; };
  const button = (label, fn) => h("div", { class: "sticky-actions" }, h("button", { class: "primary wide", onclick: fn }, label));
  const timeInput = (key) => {
    const i = h("input", { type: "number", min: 0, step: 0.1, value: state[key] });
    i.addEventListener("input", () => { state[key] = parseFloat(i.value || "0"); });
    return h("div", { class: "row" }, i, h("button", { type: "button", class: "small", onclick: () => { state[key] = now(); i.value = state[key]; } }, t("vedit.set_now")));
  };
  const aiAdvanced = (model) => section(t("common.advanced"), [stepsGuidance(state, model), seedField(state), gpuField(state) || ""]);

  const tools = {
    cut() {
      return [
        h("div", { class: "row" }, h("div", { class: "grow" }, field(t("vedit.start"), timeInput("start"))), h("div", { class: "grow" }, field(t("vedit.end"), timeInput("end")))),
        field(t("vedit.speed"), seg([0.25, 0.5, 0.75, 1, 1.5, 2, 4].map((s) => ({ value: s, label: `${s}×` })), state.speed, (v) => { state.speed = v; })),
        field(t("vedit.crop"), seg([{ value: "", label: t("vedit.keep") }, ...["16:9", "9:16", "1:1", "4:5"].map((a) => ({ value: a, label: a }))], state.aspect, (v) => { state.aspect = v; })),
        field(t("vedit.out_height"), seg([{ value: 0, label: t("vedit.keep") }, { value: 480, label: "480p" }, { value: 720, label: "720p" }, { value: 1080, label: "1080p" }], state.outHeight, (v) => { state.outHeight = v; })),
        field(t("vedit.rotate"), seg([0, 90, 180, 270].map((r) => ({ value: r, label: `${r}°` })), state.rotate, (v) => { state.rotate = v; })),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.reverse, onchange: (e) => { state.reverse = e.target.checked; } }), t("vedit.reverse")),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.flip, onchange: (e) => { state.flip = e.target.checked; } }), t("vedit.flip")),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.mute, onchange: (e) => { state.mute = e.target.checked; } }), t("vedit.mute")),
        button("✂️ " + t("common.apply"), async () => {
          if (!need()) return;
          await submitJob("video.transform", { video: source.id, start: state.start, end: state.end, speed: state.speed, reverse: state.reverse,
            aspect: state.aspect, height: state.outHeight, rotate: state.rotate, flip: state.flip, mute: state.mute });
        }),
      ];
    },
    restyle() {
      const picker = modelPicker("vid2vid", state.restyleModel, (id) => { state.restyleModel = id; notes.replaceChildren(modelNote(picker.model) || ""); });
      const notes = h("div");
      disposers.push(picker.dispose);
      const run = async () => {
        if (!need() || !state.restylePrompt.trim()) return;
        await submitJob("video.restyle", { model: state.restyleModel, video: source.id, prompt: state.restylePrompt, strength: state.strength,
          max_seconds: state.maxSeconds, steps: state.steps, guidance: state.guidance, seed: state.seed, gpu: state.gpu });
      };
      return [
        field(t("field.model"), picker), notes,
        field(t("field.prompt"), promptBox({ value: state.restylePrompt, placeholder: t("vedit.restyle_ph"), onInput: (v) => { state.restylePrompt = v; }, onSubmit: run, big: false })),
        slider({ label: t("field.strength"), min: 0.2, max: 1, step: 0.01, value: state.strength, onInput: (v) => { state.strength = v; } }),
        slider({ label: t("vedit.max_seconds"), min: 1, max: 10, value: state.maxSeconds, format: (v) => `${v} ${t("common.seconds")}`, onInput: (v) => { state.maxSeconds = v; } }),
        h("div", { class: "hint" }, t("vedit.restyle_note")), aiAdvanced(picker.model), button("🎨 " + t("common.apply"), run),
      ];
    },
    extend() {
      const picker = modelPicker("img2vid", state.extendModel, (id) => { state.extendModel = id; });
      disposers.push(picker.dispose);
      const run = async () => {
        if (!need() || !state.extendPrompt.trim()) return;
        const fps = (picker.model && picker.model.defaults && picker.model.defaults.fps) || 16;
        await submitJob("video.extend", { model: state.extendModel, video: source.id, prompt: state.extendPrompt,
          frames: Math.round(state.extendSeconds * fps) + 1, steps: state.steps, guidance: state.guidance, seed: state.seed, gpu: state.gpu });
      };
      return [
        field(t("field.model"), picker),
        field(t("field.prompt"), promptBox({ value: state.extendPrompt, placeholder: t("vedit.extend_ph"), onInput: (v) => { state.extendPrompt = v; }, onSubmit: run, big: false })),
        slider({ label: t("field.duration"), min: 2, max: 10, value: state.extendSeconds, format: (v) => `+${v} ${t("common.seconds")}`, onInput: (v) => { state.extendSeconds = v; } }),
        aiAdvanced(picker.model), button("➕ " + t("common.apply"), run),
      ];
    },
    upscale() {
      const sel = h("select");
      for (const tm of ((store.models && store.models.tools) || []).filter((x) => x.kind === "upscaler")) sel.append(h("option", { value: tm.id }, tm.name));
      for (const lm of ((store.models && store.models.local) || []).filter((x) => x.kind === "upscaler")) sel.append(h("option", { value: lm.id }, `${lm.name} (${lm.source})`));
      sel.value = state.upModel;
      sel.addEventListener("change", () => { state.upModel = sel.value; });
      return [
        field(t("edit.upscale_model"), sel),
        field(t("edit.scale"), seg([2, 3, 4].map((n) => ({ value: n, label: `×${n}` })), state.upScale, (v) => { state.upScale = v; })),
        gpuField(state) || "",
        button("🔍 " + t("common.apply"), async () => { if (need()) await submitJob("video.upscale", { video: source.id, model: state.upModel, scale: state.upScale, gpu: state.gpu }); }),
      ];
    },
    background() {
      const color = h("input", { type: "color", value: state.bgColor, style: { width: "48px", height: "34px", padding: "2px" } });
      color.addEventListener("input", () => { state.bgColor = color.value; });
      return [
        field(t("edit.bg_mode"), seg([
          { value: "color", label: t("edit.bg_color") }, { value: "blur", label: t("edit.bg_blur") }, { value: "transparent", label: t("edit.bg_transparent") + " (WebM)" },
        ], state.bgMode, (v) => { state.bgMode = v; })),
        field(t("edit.bg_color"), color),
        field(t("edit.bg_model"), seg([{ value: "birefnet-lite", label: "Lite" }, { value: "birefnet", label: "BiRefNet" }, { value: "isnet", label: "ISNet" }], state.bgModel, (v) => { state.bgModel = v; })),
        button("✂️ " + t("common.apply"), async () => { if (need()) await submitJob("video.remove_bg", { video: source.id, model: state.bgModel, mode: state.bgMode, color: state.bgColor }); }),
      ];
    },
    smooth() {
      return [
        field(t("vedit.factor"), seg([2, 3, 4].map((n) => ({ value: n, label: `×${n}` })), state.factor, (v) => { state.factor = v; })),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.slowmo, onchange: (e) => { state.slowmo = e.target.checked; } }), t("vedit.slowmo")),
        source ? h("div", { class: "hint" }, `${Math.round(source.fps)} fps → ${Math.round(source.fps * state.factor)} fps`) : null,
        button("〰 " + t("common.apply"), async () => { if (need()) await submitJob("video.interpolate", { video: source.id, factor: state.factor, slowmo: state.slowmo }); }),
      ];
    },
    gif() {
      return [
        h("div", { class: "row" }, h("div", { class: "grow" }, field(t("vedit.start"), timeInput("start"))), h("div", { class: "grow" }, field(t("vedit.end"), timeInput("end")))),
        slider({ label: t("field.fps"), min: 5, max: 30, value: state.gifFps, onInput: (v) => { state.gifFps = v; } }),
        slider({ label: t("field.width"), min: 240, max: 960, step: 40, value: state.gifWidth, format: (v) => `${v}px`, onInput: (v) => { state.gifWidth = v; } }),
        button("🎞 " + t("common.apply"), async () => {
          if (need()) await submitJob("video.to_gif", { video: source.id, start: state.start, end: state.end, fps: state.gifFps, width: state.gifWidth });
        }),
      ];
    },
    frame() {
      return [
        h("div", { class: "col" },
          h("button", { onclick: async () => { if (need()) await submitJob("video.extract_frame", { video: source.id, time: now() }); } }, "📸 " + t("vedit.current_frame")),
          h("button", { onclick: async () => { if (need()) await submitJob("video.extract_frame", { video: source.id, time: -1 }); } }, "⏭ " + t("vedit.last_frame")),
        ),
      ];
    },
    join() {
      const list = h("div", { class: "col" });
      const renderList = () => {
        clear(list);
        joinList.forEach((it, i) => list.append(h("div", { class: "row card", style: { padding: "6px 8px" } },
          h("img", { src: it.thumb_url, style: { width: "64px", height: "40px", objectFit: "cover", borderRadius: "5px" } }),
          h("span", { class: "grow small" }, `${i + 1}. ${it.duration ? it.duration.toFixed(1) + "s" : ""} ${it.width}×${it.height}`),
          h("button", { class: "ghost small", disabled: i === 0, onclick: () => { [joinList[i - 1], joinList[i]] = [joinList[i], joinList[i - 1]]; renderList(); } }, "↑"),
          h("button", { class: "ghost small", onclick: () => { joinList.splice(i, 1); renderList(); } }, "✕"))));
        list.append(h("button", { class: "small", onclick: async () => { joinList.push(...(await galleryChooser("video", { multiple: true }))); renderList(); } }, t("vedit.add_video")));
      };
      if (source && !joinList.find((x) => x.id === source.id)) joinList.unshift(source);
      renderList();
      return [
        field(t("vedit.join_list"), list),
        button("🔗 " + t("common.apply"), async () => {
          if (joinList.length < 2) return;
          await submitJob("video.concat", { videos: joinList.map((x) => x.id) });
        }),
      ];
    },
  };

  const renderControls = () => {
    clear(controls);
    add(controls, field(t("vedit.source"), srcPicker), ...tools[state.tool]());
  };
  const renderTabs = () => {
    clear(tabs);
    for (const tool of TOOLS) tabs.append(h("button", { class: state.tool === tool ? "on" : "", onclick: () => { state.tool = tool; renderTabs(); renderControls(); } }, t(`vedit.tab.${tool}`)));
  };

  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("vedit.title"))),
    tabs,
    h("div", { class: "workspace" }, controls, h("div", { class: "stage" }, results)),
  );
  renderTabs();
  renderControls();
  return () => disposers.forEach((d) => d && d());
}
