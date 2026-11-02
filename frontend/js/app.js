const API_BASE_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const browseButton = document.getElementById("browseButton");
const dropZone = document.getElementById("dropZone");
const uploadContent = document.getElementById("uploadContent");
const previewContent = document.getElementById("previewContent");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const removeImageButton = document.getElementById("removeImageButton");
const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");
const analysisStatus = document.getElementById("analysisStatus");
const resultsSection = document.getElementById("results");
const explanationSection = document.getElementById("explanation");
const predictedClass = document.getElementById("predictedClass");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const topPredictionsList = document.getElementById("topPredictionsList");
const predictionNotice = document.getElementById("predictionNotice");
const gradcamImage = document.getElementById("gradcamImage");
const gradcamPlaceholder = document.getElementById("gradcamPlaceholder");
const explanationContent = document.getElementById("explanationContent");
const chatMessages = document.getElementById("chatMessages");
const chatEmpty = document.getElementById("chatEmpty");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSendButton = document.getElementById("chatSendButton");
const toast = document.getElementById("toast");

let selectedFile = null;
let currentPredictedClass = null;
let currentConfidence = null;
let chatHistory = [];
let isAnalyzing = false;
let isChatting = false;
let toastTimer = null;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

function setStatus(message, type = "") {
  analysisStatus.textContent = message;
  analysisStatus.className = `inline-status ${type}`.trim();
}

function setAnalyzing(value) {
  isAnalyzing = value;
  analyzeButton.disabled = value || !selectedFile;
  analyzeButton.innerHTML = value
    ? "<span>Analyzing...</span><span>◌</span>"
    : "<span>Analyze image</span><span aria-hidden=\"true\">→</span>";
  browseButton.disabled = value;
  resetButton.disabled = value;
  removeImageButton.disabled = value;
}

function setChatEnabled(value) {
  chatInput.disabled = !value || isChatting;
  chatSendButton.disabled = !value || isChatting;
}

function validateFile(file) {
  const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const maxSize = 10 * 1024 * 1024;

  if (!file || !allowedTypes.includes(file.type)) {
    showToast("Please select a JPG, PNG, WEBP, or BMP image.");
    return false;
  }

  if (file.size > maxSize) {
    showToast("Image size must be 10 MB or less.");
    return false;
  }

  return true;
}

function selectFile(file) {
  if (!validateFile(file)) return;

  selectedFile = file;
  currentPredictedClass = null;
  currentConfidence = null;
  chatHistory = [];

  const previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;

  uploadContent.classList.add("hidden");
  previewContent.classList.remove("hidden");
  analyzeButton.disabled = false;
  setStatus("Image ready for analysis.", "success");

  resultsSection.classList.add("hidden");
  explanationSection.classList.add("hidden");
  setChatEnabled(false);
}

function clearSelectedFile() {
  if (imagePreview.src.startsWith("blob:")) {
    URL.revokeObjectURL(imagePreview.src);
  }

  selectedFile = null;
  imageInput.value = "";
  imagePreview.src = "";
  uploadContent.classList.remove("hidden");
  previewContent.classList.add("hidden");
  analyzeButton.disabled = true;
  setStatus("");

  resultsSection.classList.add("hidden");
  explanationSection.classList.add("hidden");
  currentPredictedClass = null;
  currentConfidence = null;
  chatHistory = [];
  setChatEnabled(false);
}

function resetResults() {
  predictedClass.textContent = "—";
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
  topPredictionsList.innerHTML = "";
  predictionNotice.textContent = "";
  gradcamImage.src = "";
  gradcamImage.classList.add("hidden");
  gradcamPlaceholder.classList.remove("hidden");
  explanationContent.textContent = "Explanation will appear after analysis.";
  chatMessages.innerHTML = "";
  chatMessages.appendChild(chatEmpty);
  chatEmpty.classList.remove("hidden");
  chatHistory = [];
}

async function checkApiHealth() {
  const dot = document.getElementById("apiStatusDot");
  const text = document.getElementById("apiStatusText");

  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) throw new Error("API unavailable");

    const data = await response.json();
    dot.classList.add("online");
    dot.classList.remove("offline");
    text.textContent = data.status === "healthy" ? "API connected" : "API available";
  } catch {
    dot.classList.add("offline");
    dot.classList.remove("online");
    text.textContent = "API offline";
  }
}

async function postImage(endpoint, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = data.detail || `Request failed (${response.status})`;
    throw new Error(message);
  }

  return data;
}

async function getExplanation(className, confidence) {
  const response = await fetch(`${API_BASE_URL}/api/explanation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      predicted_class: className,
      confidence: confidence
    })
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Could not generate AI explanation.");
  }

  return data;
}

function renderPrediction(data) {
  currentPredictedClass = data.predicted_class;
  currentConfidence = data.confidence;

  predictedClass.textContent = formatClassName(data.predicted_class);
  confidenceValue.textContent = `${Number(data.confidence).toFixed(2)}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, Number(data.confidence)))}%`;
  predictionNotice.textContent = data.notice || "";

  topPredictionsList.innerHTML = "";
  (data.top_predictions || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "top-prediction-row";

    const label = document.createElement("span");
    label.textContent = formatClassName(item.class_name);

    const track = document.createElement("div");
    track.className = "mini-track";

    const fill = document.createElement("div");
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.confidence)))}%`;
    track.appendChild(fill);

    const value = document.createElement("strong");
    value.textContent = `${Number(item.confidence).toFixed(1)}%`;

    row.append(label, track, value);
    topPredictionsList.appendChild(row);
  });

  resultsSection.classList.remove("hidden");
  document.getElementById("resultState").textContent = "Completed";
}

function formatClassName(value) {
  return String(value || "")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function runGradCAM(file) {
  gradcamPlaceholder.classList.remove("hidden");
  gradcamPlaceholder.querySelector("p").textContent = "Generating Grad-CAM visualization...";

  try {
    const data = await postImage("/api/gradcam", file);
    const imageUrl = `${API_BASE_URL}${data.gradcam_image}?t=${Date.now()}`;

    gradcamImage.onload = () => {
      gradcamImage.classList.remove("hidden");
      gradcamPlaceholder.classList.add("hidden");
    };
    gradcamImage.onerror = () => {
      gradcamImage.classList.add("hidden");
      gradcamPlaceholder.classList.remove("hidden");
      gradcamPlaceholder.querySelector("p").textContent = "Could not load Grad-CAM image.";
    };
    gradcamImage.src = imageUrl;
  } catch (error) {
    gradcamPlaceholder.querySelector("p").textContent = error.message || "Grad-CAM generation failed.";
  }
}

async function runExplanation(className, confidence) {
  explanationContent.textContent = "Generating educational explanation...";
  try {
    const data = await getExplanation(className, confidence);
    explanationContent.textContent = data.explanation || "No explanation was returned.";
  } catch (error) {
    explanationContent.textContent = error.message || "AI explanation is temporarily unavailable.";
  }
}

async function analyzeImage() {
  if (!selectedFile || isAnalyzing) return;

  setAnalyzing(true);
  setStatus("Running image classification...", "");
  resetResults();

  try {
    const prediction = await postImage("/api/predict", selectedFile);
    renderPrediction(prediction);

    explanationSection.classList.remove("hidden");
    setStatus("Prediction completed. Generating visual and AI explanations...", "success");

    setChatEnabled(true);
    await Promise.allSettled([
      runGradCAM(selectedFile),
      runExplanation(prediction.predicted_class, prediction.confidence)
    ]);

    setStatus("Analysis workflow completed.", "success");
    showToast("Image analysis completed.");
  } catch (error) {
    setStatus(error.message || "Analysis failed. Please try again.", "error");
    showToast(error.message || "Analysis failed.");
  } finally {
    setAnalyzing(false);
  }
}

function appendChatMessage(role, content) {
  chatEmpty.classList.add("hidden");

  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  message.textContent = content;
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatQuestion(question) {
  if (!currentPredictedClass || !question.trim() || isChatting) return;

  isChatting = true;
  setChatEnabled(true);
  chatInput.disabled = true;
  chatSendButton.disabled = true;

  appendChatMessage("user", question);
  chatHistory.push({ role: "user", content: question });

  const pending = document.createElement("div");
  pending.className = "chat-message assistant";
  pending.textContent = "Thinking...";
  chatMessages.appendChild(pending);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const response = await fetch(`${API_BASE_URL}/api/explanation/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        predicted_class: currentPredictedClass,
        question,
        history: chatHistory.slice(-8)
      })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "AI chat request failed.");

    pending.remove();
    appendChatMessage("assistant", data.answer || "No answer was returned.");
    chatHistory.push({ role: "assistant", content: data.answer || "" });
    chatHistory = chatHistory.slice(-8);
  } catch (error) {
    pending.remove();
    appendChatMessage("assistant", error.message || "AI chat is temporarily unavailable.");
  } finally {
    isChatting = false;
    chatInput.disabled = !currentPredictedClass;
    chatSendButton.disabled = !currentPredictedClass;
    chatInput.focus();
  }
}

browseButton.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) selectFile(file);
});

removeImageButton.addEventListener("click", clearSelectedFile);
resetButton.addEventListener("click", () => {
  clearSelectedFile();
  resetResults();
});

analyzeButton.addEventListener("click", analyzeImage);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files?.[0];
  if (file) selectFile(file);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  await sendChatQuestion(question);
});

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((nav) => nav.classList.remove("active"));
    item.classList.add("active");
  });
});

checkApiHealth();
setInterval(checkApiHealth, 15000);