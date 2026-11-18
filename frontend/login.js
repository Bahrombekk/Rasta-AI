const loginForm = document.getElementById("login-form");
const loginButton = loginForm.querySelector("button");
const loginError = document.getElementById("error");
const next = new URLSearchParams(location.search).get("next");
const target = ["/", "/admin", "/team", "/account"].includes(next) ? next : "/";

rastaSession.then(session => {
  if (session.user || session.enabled === false) location.replace(target);
  if (session.unavailable) loginError.textContent = "Serverga ulanib bo‘lmadi. Keyinroq urinib ko‘ring.";
});

loginForm.onsubmit = async event => {
  event.preventDefault();
  loginButton.disabled = true;
  loginError.textContent = "";
  try {
    await rastaJson("/api/session/login", "POST", Object.fromEntries(new FormData(loginForm)));
    location.replace(target);
  } catch (error) {
    loginError.textContent = error.message;
    loginButton.disabled = false;
  }
};
