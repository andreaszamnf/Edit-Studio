// Full-screen viewer for a library item, with follow-up actions.
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, modal, toast, errorText, fmtDate, confirmBox, clear } from "../ui.js";
import { findModel } from "../state.js";
import { compareSlider } from "./compare.js";
import { navigate } from "../router.js";

export function isGif(item) { return item && item.file && item.file.toLowerCase().endsWith(".gif"); }

export function mediaEl(item, { controls = true, autoplay = false } = {}) {
  if (item.kind === "video" && !isGif(item)) {
    return h("video", { src: item.url, controls, autoplay, loop: true, playsInline: true, preload: "metadata" });
  }
  return h("img", { src: item.url, alt: "" });
}

async function quickJob(task, params) {
  try {
    await api.submit(task, params);
    toast(t("common.sent"), "ok");
  } catch (e) { toast(errorText(e), "err"); }
}

export async function openViewer(itemOrId, { list = [], onDeleted } = {}) {
  let item = typeof itemOrId === "string" ? await api.item(itemOrId) : itemOrId;
  let index = list.findIndex((x) => x.id === item.id);
  let source = null;

  const mediaBox = h("div", { class: "viewer-media" });
  const side = h("div", { class: "viewer-side" });
  const body = h("div", { class: "viewer" }, mediaBox, side);
  const m = modal({ title: "", body });
  const titleEl = m.box.querySelector(".modal-head h2");

  const keys = (e) => {
    if (e.key === "ArrowRight" && index >= 0 && index < list.length - 1) { index++; show(list[index]); }
    if (e.key === "ArrowLeft" && index > 0) { index--; show(list[index]); }
  };
  document.addEventListener("keydown", keys);
  const close = () => { document.removeEventListener("keydown", keys); m.close(); };
  m.box.querySelector(".modal-head button").onclick = close;

  const show = async (it) => {
    item = it;
    source = null;
    if (item.sources && item.sources.length && item.kind === "image") {
      try { source = await api.item(item.sources[0]); } catch { source = null; }
    }
    render();
  };

  const render = () => {
    const p = item.params || {};
    titleEl.textContent = t(`task.${item.task}`) !== `task.${item.task}` ? t(`task.${item.task}`) : (item.name || item.id);
    clear(mediaBox);
    let showingCompare = false;
    const canCompare = source && source.kind === "image" && Math.abs(source.width / source.height - item.width / item.height) < 0.02;
    const drawMedia = () => {
      clear(mediaBox);
      mediaBox.append(showingCompare ? compareSlider(source.url, item.url) : mediaEl(item, { autoplay: true }));
    };
    drawMedia();

    const model = item.model ? findModel(item.model) : null;
    const actions = h("div", { class: "actions-grid" });
    const add = (label, fn, cls = "") => actions.append(h("button", { class: `small ${cls}`, type: "button", onclick: fn }, label));

    add(item.favorite ? "★ " + t("viewer.unfavorite") : "☆ " + t("viewer.favorite"), async () => {
      item = await api.setFavorite(item.id, !item.favorite); render();
    });
    actions.append(h("a", { class: "btn small", href: item.url, download: item.file.split("/").pop() }, "⬇ " + t("common.download")));
    add("📂 " + t("common.open_folder"), () => api.open(item.id).catch((e) => toast(errorText(e), "err")));
    if (canCompare) add("◐ " + t("edit.compare"), () => { showingCompare = !showingCompare; drawMedia(); });
    if (item.kind === "image") {
      add("🪄 " + t("viewer.edit"), () => { close(); navigate("edit", { image: item }); });
      add("🎬 " + t("viewer.animate"), () => { close(); navigate("video", { image: item }); });
      add("🔍 " + t("viewer.upscale"), () => quickJob("image.upscale", { image: item.id, model: "realesrgan-x4", scale: 2 }));
      add("✂️ " + t("viewer.remove_bg"), () => quickJob("image.remove_bg", { image: item.id }));
    } else {
      add("✂️ " + t("viewer.vedit"), () => { close(); navigate("vedit", { video: item }); });
    }
    if (p.prompt && ["image.generate", "video.generate"].includes(item.task)) {
      add("♻ " + t("viewer.reuse"), () => { close(); navigate(item.task === "image.generate" ? "image" : "video", { params: { ...p, model: item.model } }); });
    }
    if (p.prompt) {
      add("📋 " + t("viewer.copy_prompt"), async () => {
        try { await navigator.clipboard.writeText(p.prompt); toast(t("common.copied"), "ok"); } catch { /* clipboard blocked */ }
      });
    }
    add("🗑 " + t("common.delete"), async () => {
      if (!(await confirmBox(t("common.confirm_delete")))) return;
      try {
        await api.deleteItem(item.id);
        onDeleted && onDeleted(item);
        list = list.filter((x) => x.id !== item.id);
        close();
      } catch (e) { toast(errorText(e), "err"); }
    }, "danger");

    const info = h("dl", { class: "kv" });
    const kv = (k, v) => { if (v !== undefined && v !== null && v !== "") info.append(h("dt", null, k), h("dd", null, String(v))); };
    kv(t("viewer.model"), model ? model.name : item.model);
    kv(t("viewer.size"), item.width ? `${item.width}×${item.height}` : "");
    if (item.kind === "video") kv(t("viewer.duration"), item.duration ? `${item.duration.toFixed(1)}s · ${Math.round(item.fps)} fps` : "");
    kv(t("viewer.seed"), p.seed);
    kv(t("field.steps"), p.steps);
    kv(t("field.guidance"), p.guidance);
    if (p.strength !== undefined && p.strength !== null) kv(t("field.strength"), p.strength);
    kv(t("viewer.created"), fmtDate(item.created));

    clear(side);
    side.append(
      p.prompt ? h("div", { class: "col" }, h("span", { class: "lbl" }, t("field.prompt")), h("div", { class: "prompt-box" }, p.prompt)) : null,
      p.negative_prompt ? h("div", { class: "col" }, h("span", { class: "lbl" }, t("field.negative")), h("div", { class: "prompt-box" }, p.negative_prompt)) : null,
      actions,
      h("div", { class: "col" }, h("span", { class: "lbl" }, t("viewer.info")), info),
    );
  };
  await show(item);
  return { close };
}
