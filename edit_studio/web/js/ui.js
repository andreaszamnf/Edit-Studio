// Small DOM toolkit: element builder, toasts, modals, formatting.
import { t } from "./i18n.js";

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "html") el.innerHTML = v;
      else if (k === "value" || k === "textContent") el[k] = v;
      else if (k in el && typeof v !== "string") el[k] = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) { while (el.firstChild) el.firstChild.remove(); return el; }
export function add(el, ...children) { append(el, children); return el; }
export function mount(el, ...children) { clear(el); append(el, children); return el; }

export function toast(message, kind = "", ms = 4500) {
  const el = h("div", { class: `toast ${kind}` }, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), ms);
}

export function errorText(err) {
  if (!err) return t("err.error");
  const key = err.key || err.error_key || "error";
  const known = t(`err.${key}`);
  if (known !== `err.${key}`) {
    if (key === "invalid_params" && err.body && err.body.details) {
      return known + " " + err.body.details.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
    }
    return known;
  }
  return err.message || t("err.error");
}

export function modal({ title, body, footer, narrow = false, onClose } = {}) {
  const root = document.getElementById("modal-root");
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); onClose && onClose(); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const box = h("div", { class: `modal ${narrow ? "narrow" : ""}` },
    h("div", { class: "modal-head" }, h("h2", null, title || ""), h("button", { class: "ghost icon", onclick: close, title: t("common.close") }, "✕")),
    body,
    footer ? h("div", { class: "modal-foot" }, footer) : null,
  );
  const back = h("div", { class: "modal-back", onmousedown: (e) => { if (e.target === back) close(); } }, box);
  root.append(back);
  document.addEventListener("keydown", onKey);
  return { close, box };
}

export function confirmBox(message, { okText, danger = true } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); m.close(); } };
    const m = modal({
      title: t("common.confirm_delete"), narrow: true,
      body: h("div", { class: "modal-body" }, message),
      footer: [
        h("button", { onclick: () => finish(false) }, t("common.cancel")),
        h("button", { class: danger ? "danger" : "primary", onclick: () => finish(true) }, okText || t("common.delete")),
      ],
      onClose: () => finish(false),
    });
  });
}

export function fmtDuration(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "";
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m >= 60) return `${Math.floor(m / 60)}h ${m % 60}m`;
  return m ? `${m}:${String(s).padStart(2, "0")}` : `${s}s`;
}

export function fmtGB(gb) {
  if (!gb && gb !== 0) return "";
  return gb >= 10 ? `${Math.round(gb)} GB` : `${(+gb).toFixed(1)} GB`;
}

export function fmtDate(ts) {
  try { return new Date(ts * 1000).toLocaleString(); } catch { return ""; }
}

export function debounce(fn, ms = 300) {
  let id;
  return (...a) => { clearTimeout(id); id = setTimeout(() => fn(...a), ms); };
}

// Persist page form values between visits.
export function remember(key, defaults) {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("es:" + key) || "{}"); } catch { saved = {}; }
  const state = { ...defaults, ...saved };
  const save = debounce(() => { try { localStorage.setItem("es:" + key, JSON.stringify(state)); } catch { /* ignore */ } }, 400);
  return new Proxy(state, { set(obj, prop, value) { obj[prop] = value; save(); return true; } });
}

// Segmented control
export function seg(options, value, onChange, { full = false } = {}) {
  const wrap = h("div", { class: `seg ${full ? "full" : ""}` });
  const render = (v) => {
    clear(wrap);
    for (const o of options) {
      const opt = typeof o === "object" ? o : { value: o, label: String(o) };
      wrap.append(h("button", {
        type: "button", class: opt.value === v ? "on" : "", title: opt.title || "",
        onclick: () => { render(opt.value); onChange(opt.value); },
      }, opt.label));
    }
  };
  render(value);
  wrap.setValue = render;
  return wrap;
}

export function slider({ label, min, max, step = 1, value, onInput, format = (v) => v }) {
  const val = h("span", { class: "val" }, format(value));
  const input = h("input", { type: "range", min, max, step, value });
  input.addEventListener("input", () => { val.textContent = format(+input.value); onInput(+input.value); });
  const wrap = h("label", { class: "field" }, h("span", { class: "lbl" }, label, val), input);
  wrap.setValue = (v) => { input.value = v; val.textContent = format(v); };
  return wrap;
}

export function field(label, control, hint) {
  return h("label", { class: "field" }, h("span", { class: "lbl" }, label), control, hint ? h("span", { class: "hint" }, hint) : null);
}

export function section(title, body, open = false) {
  return h("details", { class: "section", open }, h("summary", null, title), h("div", { class: "section-body" }, body));
}

export function progressBar(value, indeterminate = false) {
  return h("div", { class: `progress ${indeterminate ? "indeterminate" : ""}` }, h("div", { style: { width: `${Math.round((value || 0) * 100)}%` } }));
}
