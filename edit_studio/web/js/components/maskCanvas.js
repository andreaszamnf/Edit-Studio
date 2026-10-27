// Paint a mask over an image. The mask is exported at the image's real resolution
// (white = change, black = keep) and uploaded as a hidden helper file.
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, slider } from "../ui.js";

export function maskCanvas({ onChange } = {}) {
  const wrap = h("div", { class: "col" });
  const stage = h("div", { class: "canvas-wrap" });
  const view = h("canvas");
  const cursor = h("div", { class: "brush-cursor hidden" });
  stage.append(view, cursor);

  let img = null;
  let mask = null; // offscreen canvas at natural size
  let brush = 60; // in image pixels
  let erasing = false;
  let painting = false;
  let last = null;
  let dirty = false;
  const history = [];

  const tools = h("div", { class: "canvas-tools" });
  const brushBtn = h("button", { type: "button", class: "small", onclick: () => setMode(false) }, "🖌️ " + t("edit.brush"));
  const eraseBtn = h("button", { type: "button", class: "small", onclick: () => setMode(true) }, "🧽 " + t("edit.eraser"));
  const setMode = (e) => {
    erasing = e;
    brushBtn.className = `small ${!e ? "primary" : ""}`;
    eraseBtn.className = `small ${e ? "primary" : ""}`;
  };
  const size = slider({ label: t("edit.brush_size"), min: 5, max: 300, value: brush, onInput: (v) => { brush = v; } });
  size.style.minWidth = "200px";
  tools.append(brushBtn, eraseBtn,
    h("button", { type: "button", class: "small", onclick: () => undo() }, "↶"),
    h("button", { type: "button", class: "small", onclick: () => clearMask() }, t("edit.clear")),
    h("button", { type: "button", class: "small", onclick: () => invert() }, t("edit.invert")),
    h("div", { class: "grow" }, size));
  setMode(false);

  const scale = () => (img ? img.naturalWidth / view.width : 1);

  const redraw = () => {
    if (!img) return;
    const ctx = view.getContext("2d");
    ctx.clearRect(0, 0, view.width, view.height);
    ctx.drawImage(img, 0, 0, view.width, view.height);
    ctx.save();
    ctx.globalAlpha = 0.55;
    // tint the mask red for display
    const tint = document.createElement("canvas");
    tint.width = view.width; tint.height = view.height;
    const tctx = tint.getContext("2d");
    tctx.drawImage(mask, 0, 0, view.width, view.height);
    tctx.globalCompositeOperation = "source-in";
    tctx.fillStyle = "#ff2d55";
    tctx.fillRect(0, 0, tint.width, tint.height);
    ctx.drawImage(tint, 0, 0);
    ctx.restore();
  };

  const fit = () => {
    if (!img) return;
    const maxW = Math.max(200, stage.clientWidth - 2);
    const maxH = Math.max(240, window.innerHeight * 0.62);
    const s = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 8);
    view.width = Math.round(img.naturalWidth * s);
    view.height = Math.round(img.naturalHeight * s);
    redraw();
  };

  const pos = (e) => {
    const r = view.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (view.width / r.width) * scale(), y: (e.clientY - r.top) * (view.height / r.height) * scale() };
  };

  const stroke = (a, b) => {
    const ctx = mask.getContext("2d");
    ctx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brush;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(b.x, b.y, brush / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const snapshot = () => {
    const c = document.createElement("canvas");
    c.width = mask.width; c.height = mask.height;
    c.getContext("2d").drawImage(mask, 0, 0);
    history.push(c);
    if (history.length > 30) history.shift();
  };
  const undo = () => {
    const prev = history.pop();
    if (!prev) return;
    const ctx = mask.getContext("2d");
    ctx.globalCompositeOperation = "copy";
    ctx.drawImage(prev, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    dirty = true; redraw(); onChange && onChange();
  };
  const clearMask = () => { if (!mask) return; snapshot(); mask.getContext("2d").clearRect(0, 0, mask.width, mask.height); dirty = true; redraw(); onChange && onChange(); };
  const invert = () => {
    if (!mask) return;
    snapshot();
    const c = document.createElement("canvas");
    c.width = mask.width; c.height = mask.height;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
    ctx.globalCompositeOperation = "destination-out";
    ctx.drawImage(mask, 0, 0);
    const m = mask.getContext("2d");
    m.globalCompositeOperation = "copy"; m.drawImage(c, 0, 0); m.globalCompositeOperation = "source-over";
    dirty = true; redraw(); onChange && onChange();
  };

  const moveCursor = (e) => {
    const r = view.getBoundingClientRect();
    const sr = stage.getBoundingClientRect();
    const d = (brush / scale()) * (r.width / view.width);
    Object.assign(cursor.style, { left: `${e.clientX - sr.left}px`, top: `${e.clientY - sr.top}px`, width: `${d}px`, height: `${d}px` });
  };

  view.addEventListener("pointerdown", (e) => {
    if (!img) return;
    view.setPointerCapture(e.pointerId);
    snapshot();
    painting = true;
    last = pos(e);
    stroke(last, last);
    redraw();
  });
  view.addEventListener("pointermove", (e) => {
    moveCursor(e);
    if (!painting) return;
    const p = pos(e);
    stroke(last, p);
    last = p;
    redraw();
  });
  const end = () => { if (painting) { painting = false; dirty = true; onChange && onChange(); } };
  view.addEventListener("pointerup", end);
  view.addEventListener("pointercancel", end);
  view.addEventListener("pointerenter", () => cursor.classList.remove("hidden"));
  view.addEventListener("pointerleave", () => cursor.classList.add("hidden"));
  window.addEventListener("resize", () => fit());

  wrap.append(stage, tools);

  wrap.setImage = (url) => {
    history.length = 0;
    if (!url) { img = null; const c = view.getContext("2d"); c.clearRect(0, 0, view.width, view.height); return; }
    const im = new Image();
    im.onload = () => {
      img = im;
      mask = document.createElement("canvas");
      mask.width = im.naturalWidth; mask.height = im.naturalHeight;
      brush = Math.max(10, Math.round(Math.max(im.naturalWidth, im.naturalHeight) / 18));
      size.setValue(Math.min(300, brush));
      dirty = true;
      requestAnimationFrame(fit);
    };
    im.src = url;
  };

  wrap.isEmpty = () => {
    if (!mask) return true;
    const small = document.createElement("canvas");
    small.width = 64; small.height = 64;
    const ctx = small.getContext("2d");
    ctx.drawImage(mask, 0, 0, 64, 64);
    const d = ctx.getImageData(0, 0, 64, 64).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 10) return false;
    return true;
  };

  let uploaded = null;
  wrap.upload = async () => {
    if (!dirty && uploaded) return uploaded;
    const out = document.createElement("canvas");
    out.width = mask.width; out.height = mask.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(mask, 0, 0);
    const blob = await new Promise((r) => out.toBlob(r, "image/png"));
    const file = new File([blob], "mask.png", { type: "image/png" });
    uploaded = await api.upload(file, { hidden: true });
    dirty = false;
    return uploaded;
  };
  return wrap;
}
