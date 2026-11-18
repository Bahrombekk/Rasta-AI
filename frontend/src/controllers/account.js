export default function mountPage() {
  const passwordForm = document.getElementById("password-form");
  const accountError = document.getElementById("account-error");
  const passwordButton = passwordForm.querySelector('[type="submit"]');

  rastaSession.then(session => {
    if (!session.enabled) { location.replace("/"); return; }
    if (session.user) document.getElementById("account-name").textContent = session.user.username;
  });
  passwordForm.onsubmit = async event => {
    event.preventDefault();
    passwordButton.disabled = true;
    accountError.textContent = "";
    try {
      await rastaJson("/api/session/password", "POST", Object.fromEntries(new FormData(passwordForm)));
      location.replace("/login");
    } catch (error) {
      accountError.textContent = error.message;
      passwordButton.disabled = false;
    }
  };
  passwordButton.disabled = false;
}
