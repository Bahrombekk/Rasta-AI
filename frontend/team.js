const teamList = document.getElementById("team-list");
const teamStatus = document.getElementById("team-status");
const addAdmin = document.getElementById("add-admin");
const dialog = document.getElementById("admin-dialog");
const adminForm = document.getElementById("admin-form");
const dialogError = document.getElementById("dialog-error");
let editing = null;

function openAdmin(user = null) {
  editing = user;
  adminForm.reset();
  dialogError.textContent = "";
  document.getElementById("dialog-title").textContent = user ? "Admin hisobini tahrirlash" : "Yangi admin";
  document.getElementById("password-label").textContent = user ? "Yangi parol" : "Parol";
  document.getElementById("password-hint").textContent = user
    ? "Parol amalda qolishi uchun maydonni bo‘sh qoldiring."
    : "Kamida 12 ta belgi.";
  adminForm.elements.username.value = user?.username || "";
  adminForm.elements.username.readOnly = Boolean(user);
  adminForm.elements.password.required = !user;
  adminForm.elements.active.checked = user?.active ?? true;
  document.getElementById("active-field").hidden = !user;
  dialog.showModal();
}

async function loadTeam() {
  try {
    const { users } = await rastaJson("/api/team");
    teamList.replaceChildren(...users.map(user => {
      const row = document.createElement("div");
      row.className = "team-row";
      row.innerHTML = `<span class="team-avatar">${icon("user")}</span>
        <div class="team-user"><strong>${esc(user.username)}</strong><small>${user.role === "super_admin" ? "Super admin" : "Admin"}</small></div>
        <span class="user-state ${user.active ? "active" : ""}">${user.active ? "Faol" : "To‘xtatilgan"}</span>`;
      if (user.role === "admin") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn";
        button.textContent = "Tahrirlash";
        button.onclick = () => openAdmin(user);
        row.append(button);
      }
      return row;
    }));
    teamStatus.hidden = users.length > 0;
    teamStatus.textContent = "Jamoada hisoblar yo‘q.";
    addAdmin.disabled = false;
  } catch (error) {
    teamStatus.hidden = false;
    teamStatus.textContent = error.message;
  }
}

addAdmin.onclick = () => openAdmin();
document.getElementById("dialog-close").onclick = () => dialog.close();
adminForm.onsubmit = async event => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  dialogError.textContent = "";
  const { username, password, active } = adminForm.elements;
  try {
    if (editing) {
      await rastaJson(`/api/team/${editing.id}`, "PATCH", { active: active.checked, password: password.value || null });
    } else {
      await rastaJson("/api/team", "POST", { username: username.value, password: password.value });
    }
    dialog.close();
    await loadTeam();
  } catch (error) {
    dialogError.textContent = error.message;
  } finally {
    button.disabled = false;
  }
};
rastaSession.then(session => { if (session.user?.role === "super_admin") loadTeam(); });
