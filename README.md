
# GI Disease Detection

**AI-Assisted Gastrointestinal Tract Disease Detection Using Computer Vision and Explainable AI**

An educational and research prototype that classifies gastrointestinal endoscopic images using EfficientNetB0 transfer learning and provides Grad-CAM visualizations to help explain model predictions.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical diagnostic system and must not be used to make clinical decisions.

## 1. Project Overview

Gastrointestinal endoscopic images can contain different visual patterns associated with normal anatomical regions and gastrointestinal conditions. This project explores deep learning-based image classification using the Kvasir-v2 dataset.

The application is designed to:
- Accept an endoscopic image upload.
- Predict one of eight supported image classes.
- Display the predicted class and model confidence.
- Return the top three predicted classes.
- Generate a Grad-CAM visualization of regions that influenced the model's prediction.
- Expose prediction functionality through a FastAPI backend.

## 2. Technologies

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow 2.15.1, Keras
- **Model:** EfficientNetB0 with transfer learning
- **Image Processing:** Pillow, OpenCV
- **Evaluation:** NumPy, Pandas, scikit-learn
- **Explainability:** Grad-CAM
- **Backend:** FastAPI, Uvicorn
- **Testing:** Pytest, FastAPI TestClient
- **Version Control:** Git and GitHub
- **Frontend:** In development

## 3. Dataset

This project uses the Kvasir-v2 gastrointestinal endoscopic image dataset.

Dataset source: https://datasets.simula.no/kvasir/

The dataset contains 8 classes with 1,000 images per class, for a total of 8,000 images.

Supported classes:
1. `dyed-lifted-polyps`
2. `dyed-resection-margins`
3. `esophagitis`
4. `normal-cecum`
5. `normal-pylorus`
6. `normal-z-line`
7. `polyps`
8. `ulcerative-colitis`

### Dataset Split

| Split | Images |
|---|---:|
| Training | 5,600 |
| Validation | 1,200 |
| Testing | 1,200 |
| **Total** | **8,000** |

The held-out test set is used for final evaluation rather than model training.

Download the dataset from the official source and place it in the expected local dataset directory. Dataset images are not included in this repository.

## 4. Model and Evaluation

The project uses EfficientNetB0 transfer learning for eight-class image classification.

Recorded evaluation results on the held-out test set:

| Metric | Result |
|---|---:|
| Test accuracy | 87.58% |
| Macro precision | 0.8768 |
| Macro recall | 0.8758 |
| Macro F1-score | 0.8751 |
| Weighted F1-score | 0.8751 |
| Test images | 1,200 |

These results describe performance on this dataset split only and do not establish clinical reliability or generalization to real-world clinical settings.

Generated evaluation artifacts include:
- `reports/evaluation_metrics.json`
- `reports/classification_report.csv`
- `reports/confusion_matrix.png`

## 5. Explainability

Grad-CAM generates a visualization highlighting image regions that contribute to the model's predicted class.

The visualization is intended to support research interpretation. It does not prove that the prediction is correct or that the highlighted regions represent a clinically meaningful finding.

## 6. API Endpoints

The FastAPI backend currently provides the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check API health |
| POST | `/api/upload` | Validate an uploaded image |
| POST | `/api/predict` | Predict the image class and return top predictions |
| POST | `/api/gradcam` | Generate a Grad-CAM visualization |
| GET | `/api/gradcam/image/{filename}` | Retrieve a generated Grad-CAM image |

Interactive API documentation is available at `/docs` while the server is running.

Image upload endpoints accept supported image formats and enforce a 10 MB upload limit.

### Example Prediction Response

```json
{
  "predicted_class": "polyps",
  "confidence": 92.5,
  "top_predictions": [
    {
      "class_name": "polyps",
      "confidence": 92.5
    }
  ],
  "message": "Prediction completed successfully.",
  "notice": "Research prototype notice"
}
```

The values above are illustrative only; actual predictions depend on the uploaded image and trained model.

## 7. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- A local copy of the Kvasir-v2 dataset
- The trained model file

### Clone the Repository

```bash
git clone https://github.com/Bhavish-rai/GI-Disease-Detection.git
cd GI-Disease-Detection
```

### Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure the trained model and class mapping are available in their expected locations:

- `models/best_efficientnetb0.keras`
- `src/ml/class_mapping.json`

The model is loaded when the API starts. If the model file or required dataset files are unavailable, some operations may not work.

### Start the API

Run from the repository root:

```powershell
uvicorn src.api.main:app --reload
```

Open the following addresses in your browser:

- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health

## 8. Running Tests

With the virtual environment activated, run:

```powershell
python -m pytest tests/test_api.py -v
```

The API test suite covers health checks, uploads, prediction, Grad-CAM generation and retrieval, unsupported formats, corrupted images, extension mismatches, oversized uploads, and missing files.

The current test suite has passed 10 tests in the development environment.

## 9. Project Structure

```text
GI-Disease-Detection/
├── dataset/                 # Local dataset (not tracked)
├── models/                  # Trained model files
├── reports/                 # Evaluation and Grad-CAM outputs
├── src/
│   ├── api/                 # FastAPI application and schemas
│   ├── inference/           # Prediction and Grad-CAM
│   └── ml/                  # Data loading, training and evaluation
├── tests/                   # API tests
├── requirements.txt
└── README.md
```

The frontend directory and its final structure will be documented when frontend development is complete.

## 10. Team Responsibilities

- **Member 1:** Dataset, preprocessing, model training, evaluation and integration coordination.
- **Member 2:** Prediction pipeline validation and Grad-CAM explainability.
- **Member 3:** FastAPI backend, validation and API testing.
- **Member 4:** Frontend development and API integration.

## 11. Limitations

- The model is evaluated on a particular Kvasir-v2 dataset split.
- Predictions on unrelated images may be meaningless.
- Confidence scores are model outputs, not calibrated medical probabilities.
- Grad-CAM is an approximate visualization, not proof of model reasoning.
- The system has not been established as clinically validated.
- The application is a research prototype and is not intended for medical diagnosis or treatment decisions.
