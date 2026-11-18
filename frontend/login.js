const loginForm = document.getElementById("login-form");
const loginButton = loginForm.querySelector('[type="submit"]');
const submitText = document.getElementById("submit-text");
const loginError = document.getElementById("error");
const passwordInput = document.getElementById("login-password");
const togglePassword = document.getElementById("toggle-password");
const next = new URLSearchParams(location.search).get("next");
function target(session) {
  if (next?.startsWith("/")) {
    try {
      const url = new URL(next, location.origin);
      const pages = ["/", "/admin", "/team", "/account", "/platform", "/organizations", "/dashboard", "/reports", "/members", "/stores", "/field"];
      if (url.origin === location.origin && pages.includes(url.pathname)) return url.pathname + url.search;
    } catch { /* fall back to the role's home screen */ }
  }
  return session.workspaces ? rastaHome(session.user) : "/";
}

function showError(message = "") {
  loginError.textContent = message;
  loginError.hidden = !message;
}

togglePassword.onclick = () => {
  const reveal = passwordInput.type === "password";
  passwordInput.type = reveal ? "text" : "password";
  togglePassword.setAttribute("aria-pressed", String(reveal));
  togglePassword.setAttribute("aria-label", reveal ? "Parolni yashirish" : "Parolni ko‘rsatish");
  togglePassword.innerHTML = icon(reveal ? "eye-off" : "eye");
};

rastaSession.then(session => {
  if (session.user || session.enabled === false) location.replace(target(session));
  if (session.unavailable) showError("Serverga ulanib bo‘lmadi. Keyinroq urinib ko‘ring.");
});

loginForm.onsubmit = async event => {
  event.preventDefault();
  loginButton.disabled = true;
  submitText.textContent = "Kirilmoqda…";
  showError();
  try {
    const session = await rastaJson("/api/session/login", "POST", Object.fromEntries(new FormData(loginForm)));
    location.replace(target(session));
  } catch (error) {
    showError(error.message);
    loginButton.disabled = false;
    submitText.textContent = "Kirish";
  }
};
loginButton.disabled = false;
