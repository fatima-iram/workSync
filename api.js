const API = "http://localhost:5000";

export function logout() {

    localStorage.removeItem("loggedIn");
    localStorage.removeItem("userEmail");

    window.location.href = "index.html";
}

// ── EMAILS ───────────────────────────
export async function fetchEmails() {

    const res = await fetch(`${API}/emails`);

    if (!res.ok) {
        console.error("Failed to fetch emails");
        return [];
    }

    return await res.json();
}

// ── TODOS ────────────────────────────
export async function fetchTodos() {

    const res = await fetch(`${API}/todos`);

    if (!res.ok) {
        console.error("Failed to fetch todos");
        return [];
    }

    return await res.json();
}

// ── VOICE ────────────────────────────
export async function startVoiceBriefing(emails) {

    const res = await fetch(`${API}/voice`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ emails })
    });

    return await res.json();
}

export async function stopVoiceBriefing() {

    const res = await fetch(`${API}/voice/stop`, {
        method: "POST"
    });

    return await res.json();
}

export async function saveTodos(tasks) {

    const res = await fetch(`${API}/todos`, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(tasks)

    });

    return await res.json();
}