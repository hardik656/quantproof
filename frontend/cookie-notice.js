(function () {
  var KEY = "qp_cookie_ok";
  try {
    if (localStorage.getItem(KEY) === "1") return;
  } catch (e) {
    return;
  }
  var bar = document.createElement("div");
  bar.setAttribute("role", "dialog");
  bar.setAttribute("aria-label", "Cookie notice");
  bar.style.cssText = [
    "position:fixed",
    "left:12px",
    "right:12px",
    "bottom:12px",
    "z-index:10000",
    "max-width:720px",
    "margin:0 auto",
    "background:#0d1117",
    "color:#c9d1d9",
    "border:1px solid #30363d",
    "padding:12px 14px",
    "font:13px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",
    "display:flex",
    "gap:12px",
    "align-items:center",
    "flex-wrap:wrap",
    "box-shadow:0 8px 24px rgba(0,0,0,0.35)"
  ].join(";");
  var text = document.createElement("div");
  text.style.flex = "1";
  text.innerHTML = "This site uses Google Analytics cookies (<code>_ga</code>). Trade files are uploaded and checked on our server. <a href=\"/privacy-policy\" style=\"color:#00ff88\">Privacy policy</a>";
  var button = document.createElement("button");
  button.type = "button";
  button.textContent = "OK";
  button.style.cssText = "background:#00ff88;color:#0d1117;border:none;padding:8px 14px;font-weight:700;cursor:pointer";
  button.addEventListener("click", function () {
    try { localStorage.setItem(KEY, "1"); } catch (e) {}
    bar.remove();
  });
  bar.appendChild(text);
  bar.appendChild(button);
  document.body.appendChild(bar);
})();
