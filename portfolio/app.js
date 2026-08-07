let trajectories = [];

document.addEventListener("DOMContentLoaded", () => {
  fetch("portfolio_trajectories.json")
    .then((res) => res.json())
    .then((data) => {
      trajectories = data;
      renderSidebar();
      if (trajectories.length > 0) {
        selectQuestion(0);
      }
    })
    .catch((err) => {
      console.error("Could not load portfolio trajectories:", err);
    });
});

function renderSidebar() {
  const listEl = document.getElementById("questionList");
  listEl.innerHTML = "";

  trajectories.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "question-item";
    if (index === 0) li.classList.add("active");

    const badgeClass = item.exact_match ? "badge-pass" : "badge-fail";
    const badgeText = item.exact_match ? "EM Match" : "Partial";

    li.innerHTML = `
      <div><strong>${item.question}</strong></div>
      <span class="badge ${badgeClass}">${badgeText}</span>
    `;

    li.addEventListener("click", () => {
      document.querySelectorAll(".question-item").forEach((el) => el.classList.remove("active"));
      li.classList.add("active");
      selectQuestion(index);
    });

    listEl.appendChild(li);
  });
}

function selectQuestion(index) {
  const q = trajectories[index];
  if (!q) return;

  document.getElementById("targetQuestion").innerText = q.question;
  document.getElementById("predictedAnswer").innerText = q.predicted_answer;
  document.getElementById("groundTruth").innerText = q.ground_truth;
  document.getElementById("jointF1").innerText = q.joint_f1;
  document.getElementById("totalSteps").innerText = q.step_count;

  const stepsContainer = document.getElementById("stepsContainer");
  stepsContainer.innerHTML = "";

  q.steps.forEach((s) => {
    const stepEl = document.createElement("div");
    stepEl.className = "step-card";

    let html = `<div class="step-header">Step ${s.step}: ${s.action}</div>`;
    html += `<div class="thought-box"><strong>Thought:</strong> ${s.thought}</div>`;
    html += `<div class="action-box"><strong>Action:</strong> ${s.action}</div>`;
    if (s.observation) {
      html += `<div class="obs-box"><strong>Observation:</strong><br>${s.observation}</div>`;
    }

    stepEl.innerHTML = html;
    stepsContainer.appendChild(stepEl);
  });
}
