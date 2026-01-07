const imageInput = document.getElementById("imageInput");
const runBtn = document.getElementById("runBtn");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const objectSelect = document.getElementById("objectSelect");
const recTableBody = document.querySelector("#recommendationsTable tbody");

let originalImage = null;
let lastDetections = [];

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      originalImage = img;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = false;
      statusEl.textContent = "Image loaded";
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

runBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) return;

  statusEl.textContent = "Detecting...";
  runBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  const selected = objectSelect.value;
  if (selected) {
    formData.append("selected_label", selected);
  }

  try {
    const res = await fetch("/detect", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    drawDetections(data);
    populateDropdown(data.detections);
    populateRecommendations(data.recommendations);
    statusEl.textContent = "Done";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error";
  } finally {
    runBtn.disabled = false;
  }
});

function drawDetections(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(originalImage, 0, 0);

  data.detections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;
    ctx.strokeStyle = "#4f7cff";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = "#4f7cff";
    ctx.font = "14px sans-serif";
    ctx.fillText(det.label, x1 + 4, y1 - 6);
  });
}

function populateDropdown(detections) {
  const labels = [...new Set(detections.map(d => d.label))];
  objectSelect.innerHTML = '<option value="">Select an object to shop for</option>';

  labels.forEach(label => {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    objectSelect.appendChild(opt);
  });
}

function populateRecommendations(items = []) {
  recTableBody.innerHTML = "";

  items.forEach(item => {
    const tr = document.createElement("tr");

    const imgTd = document.createElement("td");
    const img = document.createElement("img");
    img.src = item.image?.imageUrl || "";
    img.width = 80;
    imgTd.appendChild(img);

    const titleTd = document.createElement("td");
    titleTd.textContent = item.title || "Untitled";

    const linkTd = document.createElement("td");
    const a = document.createElement("a");
    a.href = item.itemWebUrl;
    a.textContent = "View";
    a.target = "_blank";
    linkTd.appendChild(a);

    tr.appendChild(imgTd);
    tr.appendChild(titleTd);
    tr.appendChild(linkTd);

    recTableBody.appendChild(tr);
  });
}
