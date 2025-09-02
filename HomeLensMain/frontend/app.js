async function uploadImage() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) {
    alert("Please choose an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  displayResults(data.detections);
}

function displayResults(detections) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (!detections.length) {
    resultsDiv.innerHTML = "<p>No furniture detected.</p>";
    return;
  }

  const table = document.createElement("table");
  table.innerHTML = `
    <tr>
      <th>Item</th>
      <th>Style</th>
    </tr>
  `;

  detections.forEach(det => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${det.item}</td><td>${det.style}</td>`;
    table.appendChild(row);
  });

  resultsDiv.appendChild(table);
}
