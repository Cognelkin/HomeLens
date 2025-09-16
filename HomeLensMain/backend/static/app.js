const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const countsEl = document.getElementById('counts');
const detBody = document.querySelector('#detTable tbody');
const recBody = document.querySelector('#recTable tbody');
const scoreEl = document.getElementById('score');

let originalImage = null;

// Load image
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

// Run detection
runBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) return;

  statusEl.textContent = 'Detecting...';
  runBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/detect', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Detection failed');
    const data = await res.json();
    renderDetections(data);
    renderRecommendations(data.recommendations || []);
    statusEl.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error running detection.';
  } finally {
    runBtn.disabled = false;
  }
});

// Draw YOLO detections
function renderDetections(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(originalImage, 0, 0);

  for (const det of data.detections) {
    const [x1, y1, x2, y2] = det.box;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(84, 91, 255, 1)';
    ctx.strokeRect(x1, y1, w, h);

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

  countsEl.innerHTML = '';
  const entries = Object.entries(data.counts).filter(([k, v]) => v > 0);
  if (entries.length === 0) {
    countsEl.innerHTML = '<li>(no selected features found)</li>';
  } else {
    entries.forEach(([k,v]) => {
      const li = document.createElement('li');
      li.textContent = `${k}: ${v}`;
      countsEl.appendChild(li);
    });
  }

  detBody.innerHTML = '';
  data.detections.forEach(det => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${det.label}</td><td>${(det.confidence*100).toFixed(1)}%</td><td>${det.box.map(v=>v.toFixed(0)).join(', ')}</td>`;
    detBody.appendChild(tr);
  });

  scoreEl.textContent = data.amenities_score.toFixed(1);
}

// Render eBay recommendations
function renderRecommendations(items) {
  recBody.innerHTML = '';
  if (!items || items.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="3">(no recommendations)</td>';
    recBody.appendChild(tr);
    return;
  }

  items.forEach(item => {
    const tr = document.createElement('tr');
    const imgTd = document.createElement('td');
    const titleTd = document.createElement('td');
    const urlTd = document.createElement('td');

    imgTd.innerHTML = item.image?.imageUrl ? `<img src="${item.image.imageUrl}" alt="${item.title}" width="400">` : '(no image)';
    titleTd.textContent = item.title || '(no title)';
    urlTd.innerHTML = item.itemWebUrl ? `<a href="${item.itemWebUrl}" target="_blank">Link</a>` : '(no URL)';

    tr.appendChild(imgTd);
    tr.appendChild(titleTd);
    tr.appendChild(urlTd);
    recBody.appendChild(tr);
  });
}

