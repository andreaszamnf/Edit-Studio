// App bootstrap: navigation, live events, jobs panel.
import { t, getLang, setLang } from "./i18n.js";
import { api, connectEvents } from "./api.js";
import { h, toast, errorText } from "./ui.js";
import { store, on, loadSystem, loadModels, loadJobs, handleEvent } from "./state.js";
import { route, start } from "./router.js";
import { mountJobsPanel, mountGpuStatus } from "./components/jobs.js";
import { imagePage } from "./pages/image.js";
import { editPage } from "./pages/edit.js";
import { videoPage } from "./pages/video.js";
import { veditPage } from "./pages/vedit.js";
import { galleryPage } from "./pages/gallery.js";
import { modelsPage } from "./pages/models.js";
import { settingsPage } from "./pages/settings.js";

const NAV = [
  ["image", "🎨"], ["edit", "🪄"], ["video", "🎬"], ["vedit", "✂️"], null, ["gallery", "🖼️"], ["models", "📦"], ["settings", "⚙️"],
];

function renderNav() {
  const nav = document.getElementById("nav-links");
  nav.replaceChildren(...NAV.map((n) => (n === null ? h("div", { class: "nav-sep" }) : h("a", {
    class: "nav-link", href: `#/${n[0]}`, dataset: { page: n[0] }, title: t(`nav.${n[0]}`),
  }, h("span", { class: "ico" }, n[1]), h("span", { class: "txt" }, t(`nav.${n[0]}`))))));
  const lang = document.getElementById("lang-toggle");
  lang.textContent = "🌐 " + t("lang.switch");
  lang.onclick = () => {
    setLang(getLang() === "el" ? "en" : "el");
    api.saveSettings({ language: getLang() }).catch(() => {});
    renderNav();
    start();
  };
}

route("image", imagePage);
route("edit", editPage);
route("video", videoPage);
route("vedit", veditPage);
route("gallery", galleryPage);
route("models", modelsPage);
route("settings", settingsPage);

async function boot() {
  try {
    const sys = await loadSystem();
    if (sys.settings && sys.settings.language) setLang(sys.settings.language);
  } catch (e) {
    document.getElementById("page").append(h("div", { class: "notice err", style: { margin: "24px" } }, errorText(e)));
    return;
  }
  renderNav();
  mountJobsPanel();
  mountGpuStatus();
  await Promise.all([loadJobs().catch(() => {}), loadModels().catch((e) => toast(errorText(e), "err"))]);
  start();

  let lostToast = null;
  connectEvents(handleEvent, (ok) => {
    if (!ok && !lostToast) { lostToast = true; toast(t("err.server"), "err", 3000); }
    if (ok && lostToast) { lostToast = null; loadJobs().catch(() => {}); loadSystem().catch(() => {}); }
  });

  // Tell the user when something fails, refresh memory status when work finishes.
  let sysTimer = null;
  on("job-finished", (j) => {
    if (j.status === "failed") toast(`${t(`task.${j.task}`)}: ${errorText({ key: j.error_key, message: j.error, info: j.error_info })}`, "err", 9000);
    clearTimeout(sysTimer);
    sysTimer = setTimeout(() => loadSystem().catch(() => {}), 400);
    if (j.task === "models.scan" || j.task === "model.download" || j.task === "tool.download") loadModels(true).catch(() => {});
  });
  on("job", (j) => { if (j.status === "running" && j.progress === 0) { clearTimeout(sysTimer); sysTimer = setTimeout(() => loadSystem().catch(() => {}), 800); } });
  if (!store.system.hardware.gpus.length && store.system.hardware.backend === "cpu") {
    toast(t("models.no_gpu"), "err", 8000);
  }
}

boot();
