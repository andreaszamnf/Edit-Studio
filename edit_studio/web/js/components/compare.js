// Before / after slider.
import { h } from "../ui.js";

export function compareSlider(beforeUrl, afterUrl) {
  const before = h("img", { src: beforeUrl, alt: "" });
  const afterImg = h("img", { src: afterUrl, alt: "" });
  const after = h("div", { class: "after" }, afterImg);
  const handle = h("div", { class: "handle" });
  const wrap = h("div", { class: "compare" }, before, after, handle);
  let pct = 50;
  const apply = () => {
    after.style.clipPath = `inset(0 0 0 ${pct}%)`;
    handle.style.left = `${pct}%`;
    afterImg.style.width = `${before.clientWidth}px`;
  };
  const move = (e) => {
    const r = wrap.getBoundingClientRect();
    pct = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
    apply();
  };
  let drag = false;
  wrap.addEventListener("pointerdown", (e) => { drag = true; wrap.setPointerCapture(e.pointerId); move(e); });
  wrap.addEventListener("pointermove", (e) => { if (drag) move(e); });
  wrap.addEventListener("pointerup", () => { drag = false; });
  before.addEventListener("load", apply);
  window.addEventListener("resize", apply);
  requestAnimationFrame(apply);
  return wrap;
}
