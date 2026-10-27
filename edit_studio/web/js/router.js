// Hash router with "intents": data handed from one page to the next (e.g. "edit this image").
const routes = new Map();
let intent = null;
let current = null;

export function route(name, render) { routes.set(name, render); }

export function navigate(name, data = null) {
  intent = data ? { page: name, data } : null;
  if (location.hash === `#/${name}`) start();
  else location.hash = `#/${name}`;
}

export function takeIntent(page) {
  if (intent && intent.page === page) {
    const d = intent.data;
    intent = null;
    return d;
  }
  return null;
}

export function currentPage() { return current; }

let cleanup = null;
export function start() {
  const name = (location.hash.replace(/^#\/?/, "") || "image").split("?")[0];
  const render = routes.get(name) || routes.get("image");
  current = routes.has(name) ? name : "image";
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } }
  const page = document.getElementById("page");
  page.innerHTML = "";
  cleanup = render(page) || null;
  document.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", a.dataset.page === current));
}

window.addEventListener("hashchange", start);
