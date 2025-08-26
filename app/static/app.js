const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const countsEl = document.getElementById('counts');
const detBody = document.querySelector('#detTable tbody');
const scoreEl = document.getElementById('score');

let originalImage = null;

imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) { runBtn.disabled = true; return; }
  const reader = new FileReader();
  reader.onload = function(evt) {
    const img = new Image();
    img.onload = function() {
      originalImage = img;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = false;
      statusEl.textContent = `Loaded ${file.name} (${img.width}×${img.height})`;
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
});

runBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) return;

  statusEl.textContent = 'Detecting...';
  runBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/detect', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Detection failed');
    const data = await res.json();
    renderDetections(data);
    statusEl.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error running detection.';
  } finally {
    runBtn.disabled = false;
  }
});

function renderDetections(data) {
  // Draw the image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(originalImage, 0, 0);

  // Boxes
  for (const det of data.detections) {
    const [x1, y1, x2, y2] = det.box;
    const w = x2 - x1;
    const h = y2 - y1;

    // box
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(84, 91, 255, 1)';
    ctx.strokeRect(x1, y1, w, h);

    // label background
    const label = `${det.label} ${(det.confidence*100).toFixed(1)}%`;
    ctx.font = '16px system-ui';
    const metrics = ctx.measureText(label);
    const lh = 20;
    const lw = metrics.width + 8;
    ctx.fillStyle = 'rgba(84, 91, 255, 0.9)';
    ctx.fillRect(x1, y1 - lh, lw, lh);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 4, y1 - 6);
  }

  // Counts
  countsEl.innerHTML = '';
  const entries = Object.entries(data.counts).filter(([k, v]) => v > 0);
  for (const [k, v] of entries) {
    const li = document.createElement('li');
    li.textContent = `${k}: ${v}`;
    countsEl.appendChild(li);
  }
  if (entries.length === 0) {
    const li = document.createElement('li');
    li.textContent = '(no selected features found)';
    countsEl.appendChild(li);
  }

  // Table
  detBody.innerHTML = '';
  for (const det of data.detections) {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td'); td1.textContent = det.label;
    const td2 = document.createElement('td'); td2.textContent = (det.confidence*100).toFixed(1) + '%';
    const td3 = document.createElement('td'); td3.textContent = det.box.map(v => v.toFixed(0)).join(', ');
    tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
    detBody.appendChild(tr);
  }

  // Score
  scoreEl.textContent = data.amenities_score.toFixed(1);
}
