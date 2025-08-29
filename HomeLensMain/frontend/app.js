async function uploadImage() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) {
    alert("Please select an image");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("http://127.0.0.1:8000/analyze/", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  console.log(data);

  let html = "<h2>Detections</h2><ul>";
  data.detections.forEach(d => {
    html += `<li>Class ID: ${d.class_id}, Confidence: ${d.confidence.toFixed(2)}</li>`;
  });
  html += "</ul>";

  if (data.recommendations && data.recommendations.predictions) {
    html += "<h2>Style Recommendations</h2><ul>";
    data.recommendations.predictions.forEach(p => {
      html += `<li>${p.name} (${(p.confidence*100).toFixed(1)}%)</li>`;
    });
    html += "</ul>";
  }

  document.getElementById("results").innerHTML = html;
}
