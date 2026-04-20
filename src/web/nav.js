const NAV_LINKS = [
  { label: "Control Panel", href: "/" },
  { label: "Chat",          href: "/chat/" },
  { label: "Training",      href: "/training/" },
  { label: "Tokenizer",     href: "/tokenizer/" },
];

export function insertNav() {
  const nav = document.createElement("nav");
  nav.className = "site-nav";

  const title = document.createElement("span");
  title.className = "nav-title";
  title.textContent = "Scout";
  nav.appendChild(title);

  const currentPath = window.location.pathname.replace(/\/?$/, "/").replace(/^([^/]*\/[^/]*).*/, "$1/");

  for (const { label, href } of NAV_LINKS) {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label;

    const normalizedHref = href.replace(/\/?$/, "/");
    const isHome = href === "/" && window.location.pathname === "/";
    const isActive = isHome || (href !== "/" && window.location.pathname.startsWith(href));
    if (isActive) a.classList.add("active");

    nav.appendChild(a);
  }

  document.body.insertBefore(nav, document.body.firstChild);
}