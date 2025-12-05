const API = "http://localhost:8000";

let lastSessionId = null;
let lastClothPath = null;

const el = (id) => document.getElementById(id);

el("tryon-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  el("status").textContent = "Uploading and generating...";
  const fd = new FormData();
  const person = el("person").files[0];
  const cloth = el("cloth").files[0];
  fd.append("person_image", person);
  fd.append("cloth_image", cloth);
  fd.append("garment_type", el("garment").value);
  fd.append("background_option", el("background").value);
  const res = await fetch(`${API}/api/tryon`, { method: "POST", body: fd });
  const data = await res.json();
  if (data.error) {
    el("status").textContent = `Error: ${data.error}`;
    return;
  }
  el("status").textContent = data.pose_quality_ok ? `Pose OK: ${data.pose_message}` : `Pose Warning: ${data.pose_message}`;
  el("person-img").src = `${API}${data.person_url}`;
  el("cloth-img").src = `${API}${data.cloth_url}`;
  el("result-a").src = `${API}${data.result_A_url}`;
  el("result-b").src = `${API}${data.result_B_url}`;
  el("compare").src = `${API}${data.comparison_url}`;
  lastSessionId = crypto.randomUUID();
  lastClothPath = `${API}${data.cloth_url}`;
  const recRes = await fetch(`${API}/api/recommendations?k=3&cloth_path=${encodeURIComponent(data.cloth_url)}`);
  const recs = await recRes.json();
  const recDiv = el("recs");
  recDiv.innerHTML = "";
  (recs.items || []).forEach((u) => {
    const img = document.createElement("img");
    img.src = `${API}${u}`;
    img.width = 150;
    recDiv.appendChild(img);
  });
});

const starRow = (id) => {
  const row = el(id);
  row.innerHTML = "";
  const stars = [];
  for (let i = 1; i <= 5; i++) {
    const s = document.createElement("button");
    s.textContent = "★";
    s.className = "text-gray-300 text-2xl";
    s.dataset.value = i;
    s.addEventListener("click", () => {
      stars.forEach((b, idx) => b.className = idx < i ? "text-yellow-400 text-2xl" : "text-gray-300 text-2xl");
      row.dataset.value = i;
    });
    row.appendChild(s);
    stars.push(s);
  }
  row.dataset.value = 5;
};
starRow("realism-stars");
starRow("fit-stars");

el("submit-feedback").addEventListener("click", async () => {
  const payload = {
    realism_rating: Number(el("realism-stars").dataset.value || 5),
    fit_rating: Number(el("fit-stars").dataset.value || 5),
    preferred_model: el("preferred").value,
    session_id: lastSessionId,
  };
  const res = await fetch(`${API}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) {
    el("summary").textContent = `Error: ${data.error}`;
    return;
  }
  const s = await fetch(`${API}/api/feedback_summary`);
  const sum = await s.json();
  el("summary").textContent = `Avg Realism: ${sum.avg_realism.toFixed(2)} | Avg Fit: ${sum.avg_fit.toFixed(2)} | Pref A: ${sum.pref_A_percent.toFixed(1)}% | Pref B: ${sum.pref_B_percent.toFixed(1)}% | Both: ${sum.pref_Both_percent.toFixed(1)}% | None: ${sum.pref_None_percent.toFixed(1)}%`;
});

document.getElementById("download-a").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = el("result-a").src;
  a.download = "tryon_model_A.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

document.getElementById("download-b").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = el("result-b").src;
  a.download = "tryon_model_B.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});
