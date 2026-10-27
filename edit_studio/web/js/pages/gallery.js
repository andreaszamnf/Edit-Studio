// Library of everything created (and uploaded).
import { t } from "../i18n.js";
import { api } from "../api.js";
import { h, clear, seg, debounce, toast, errorText, add } from "../ui.js";
import { on } from "../state.js";
import { openViewer } from "../components/viewer.js";

export function galleryPage(root) {
  let filter = "all";
  let q = "";
  let items = [];
  let total = 0;
  const grid = h("div", { class: "results" });
  const more = h("button", { class: "hidden", onclick: () => load(true) }, t("gallery.more"));

  const query = () => ({
    kind: filter === "images" ? "image" : filter === "videos" ? "video" : "",
    favorites: filter === "favorites" ? "true" : "",
    uploads: filter === "uploads" ? "true" : "",
    q, limit: 60,
  });
  const clean = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== "" && v !== undefined));

  const load = async (append = false) => {
    try {
      const res = await api.gallery(clean({ ...query(), offset: append ? items.length : 0 }));
      items = append ? [...items, ...res.items] : res.items;
      total = res.total;
      render();
    } catch (e) { toast(errorText(e), "err"); }
  };
  const tile = (it) => h("div", { class: "tile", onclick: () => openViewer(it, { list: items, onDeleted: () => load() }) },
    h("img", { src: it.thumb_url, loading: "lazy" }),
    it.kind === "video" ? h("div", { class: "tile-badge" }, h("span", { class: "tag" }, "▶ " + (it.duration ? it.duration.toFixed(1) + "s" : "GIF"))) : null,
    it.favorite ? h("div", { class: "tile-fav" }, "★") : null);
  const render = () => {
    clear(grid);
    if (!items.length) grid.append(h("div", { class: "empty", style: { gridColumn: "1 / -1" } }, h("div", { class: "big-ico" }, "🖼️"), t("gallery.empty")));
    for (const it of items) grid.append(tile(it));
    more.classList.toggle("hidden", items.length >= total);
  };

  const search = h("input", { type: "search", placeholder: t("gallery.search"), style: { maxWidth: "320px" } });
  search.addEventListener("input", debounce(() => { q = search.value; load(); }, 350));

  add(root, 
    h("div", { class: "page-head" }, h("h1", null, t("gallery.title")), h("span", { class: "spacer" }),
      h("button", { class: "small", onclick: () => api.open("outputs") }, "📂 " + t("common.open_folder"))),
    h("div", { class: "page-body" },
      h("div", { class: "row wrap", style: { marginBottom: "14px" } },
        seg([
          { value: "all", label: t("gallery.all") }, { value: "images", label: t("gallery.images") }, { value: "videos", label: t("gallery.videos") },
          { value: "favorites", label: "★ " + t("gallery.favorites") }, { value: "uploads", label: t("gallery.uploads") },
        ], filter, (v) => { filter = v; load(); }),
        search),
      grid,
      h("div", { class: "row", style: { justifyContent: "center", marginTop: "16px" } }, more),
    ),
  );
  load();
  const off = on("gallery", () => load());
  return off;
}
