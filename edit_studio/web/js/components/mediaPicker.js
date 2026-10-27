// Pick an image/video: drag & drop, file dialog, or from the library.
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, clear, modal, toast, errorText } from "../ui.js";

export async function uploadFile(file, opts) {
  return api.upload(file, opts);
}

export function galleryChooser(kind, { multiple = false } = {}) {
  return new Promise((resolve) => {
    const picked = new Map();
    let done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); m.close(); } };
    const grid = h("div", { class: "results" });
    const body = h("div", { class: "modal-body" }, grid);
    const load = async () => {
      const [outs, ups] = await Promise.all([
        api.gallery({ kind, limit: 120 }),
        api.gallery({ kind, uploads: true, limit: 60 }),
      ]);
      const items = [...outs.items, ...ups.items];
      clear(grid);
      if (!items.length) grid.append(h("div", { class: "empty" }, t("gallery.empty")));
      for (const it of items) {
        const tile = h("div", { class: "tile", onclick: () => {
          if (!multiple) return finish([it]);
          if (picked.has(it.id)) { picked.delete(it.id); tile.style.outline = ""; } else { picked.set(it.id, it); tile.style.outline = "3px solid var(--accent)"; }
        } }, h("img", { src: it.thumb_url, loading: "lazy" }));
        grid.append(tile);
      }
    };
    const m = modal({
      title: t("picker.choose_title"), body,
      footer: multiple ? [h("button", { class: "primary", onclick: () => finish([...picked.values()]) }, t("common.choose"))] : null,
      onClose: () => finish([]),
    });
    load().catch((e) => toast(errorText(e), "err"));
  });
}

/**
 * mediaPicker({kind: "image"|"video", value, onChange}) -> element
 * value / onChange use gallery items ({id, url, thumb_url, width, height, ...}).
 */
export function mediaPicker({ kind = "image", value = null, onChange, compact = false }) {
  const wrap = h("div", { class: "col" });
  let item = value;
  const input = h("input", { type: "file", accept: kind === "image" ? "image/*" : "video/*,.mkv,.mov", class: "hidden" });

  const set = (it) => { item = it; render(); onChange && onChange(it); };
  const handleFiles = async (files) => {
    const f = files && files[0];
    if (!f) return;
    render(true);
    try { set(await uploadFile(f)); } catch (e) { toast(errorText(e), "err"); render(); }
  };
  input.addEventListener("change", () => handleFiles(input.files));

  const render = (uploading = false) => {
    clear(wrap);
    wrap.append(input);
    if (uploading) {
      wrap.append(h("div", { class: "dropzone" }, t("picker.uploading")));
      return;
    }
    if (item) {
      const media = item.kind === "video" && !item.file.endsWith(".gif")
        ? h("video", { src: item.url, controls: true, muted: true, preload: "metadata" })
        : h("img", { src: item.url });
      wrap.append(h("div", { class: "picked" }, compact ? null : media,
        h("div", { class: "picked-bar" },
          compact ? h("img", { src: item.thumb_url, style: { width: "42px", height: "42px", objectFit: "cover", borderRadius: "6px" } }) : null,
          h("span", { class: "grow" }, item.width && item.height ? `${item.width}×${item.height}` : "", item.duration ? ` · ${item.duration.toFixed(1)}s` : ""),
          h("button", { class: "small", type: "button", onclick: () => input.click() }, t("common.change")),
          h("button", { class: "small", type: "button", onclick: async () => { const [it] = await galleryChooser(kind); if (it) set(it); } }, t("picker.from_gallery")),
          h("button", { class: "small ghost", type: "button", onclick: () => set(null) }, "✕"),
        )));
      return;
    }
    const dz = h("div", { class: "dropzone", onclick: () => input.click() },
      h("div", { style: { fontSize: "26px" } }, kind === "image" ? "🖼️" : "🎞️"),
      h("div", null, t(kind === "image" ? "picker.drop_image" : "picker.drop_video")),
      h("button", { class: "small", type: "button", onclick: async (e) => { e.stopPropagation(); const [it] = await galleryChooser(kind); if (it) set(it); } }, t("picker.from_gallery")),
    );
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); handleFiles(e.dataTransfer.files); });
    wrap.append(dz);
  };
  render();
  Object.defineProperty(wrap, "value", { get: () => item });
  wrap.setValue = (it) => { item = it; render(); };
  return wrap;
}

/** Several reference images (used by multi-image edit models). */
export function multiImagePicker({ value = [], onChange, max = 3 }) {
  const wrap = h("div", { class: "thumb-row" });
  let items = [...value];
  const input = h("input", { type: "file", accept: "image/*", multiple: true, class: "hidden" });
  input.addEventListener("change", async () => {
    for (const f of input.files) {
      if (items.length >= max) break;
      try { items.push(await uploadFile(f)); } catch (e) { toast(errorText(e), "err"); }
    }
    input.value = "";
    render(); onChange(items);
  });
  const render = () => {
    clear(wrap);
    wrap.append(input);
    for (const it of items) {
      wrap.append(h("div", { class: "mini" }, h("img", { src: it.thumb_url }),
        h("button", { type: "button", onclick: () => { items = items.filter((x) => x !== it); render(); onChange(items); } }, "✕")));
    }
    if (items.length < max) {
      wrap.append(h("button", { type: "button", class: "small", style: { width: "74px", height: "74px" }, onclick: () => input.click() }, "+"));
      wrap.append(h("button", { type: "button", class: "small ghost", onclick: async () => {
        const picked = await galleryChooser("image", { multiple: true });
        items = [...items, ...picked].slice(0, max); render(); onChange(items);
      } }, t("picker.from_gallery")));
    }
  };
  render();
  Object.defineProperty(wrap, "value", { get: () => items });
  return wrap;
}
