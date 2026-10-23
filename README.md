
# GI Disease Detection

AI-assisted gastrointestinal disease classification from endoscopic images using deep learning, transfer learning, and Grad-CAM explainability.

## Project Overview

This educational and research prototype explores the classification of gastrointestinal endoscopic images using the Kvasir-v2 dataset and a transfer-learning model such as EfficientNetB0.

The system is intended to display a predicted image class, model confidence, and a Grad-CAM visualization highlighting image regions that influenced the prediction.

This project is not a medical diagnostic system and must not be used as a substitute for professional medical assessment.

## Planned Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0 transfer learning
- NumPy, Pandas, scikit-learn and OpenCV
- Grad-CAM explainability
- FastAPI
- HTML, CSS and JavaScript
- Git and GitHub

## Team Responsibilities

- Member 1: Dataset, preprocessing, model training and evaluation
- Member 2: Prediction pipeline and Grad-CAM
- Member 3: FastAPI backend and integration
- Member 4: Frontend and API integration

## Project Structure

- `dataset/` - Local dataset files; not tracked by Git
- `models/` - Saved model files; not tracked by Git
- `reports/` - Evaluation outputs and generated reports
- `src/ml/` - Data loading, preprocessing, training and evaluation
- `src/inference/` - Prediction and Grad-CAM
- `src/api/` - FastAPI application
- `frontend/` - Web interface
- `tests/` - Automated tests

## Dataset

Kvasir-v2: https://datasets.simula.no/kvasir/

Download and use the dataset according to its published terms. Dataset images are not stored in this repository.

## Status

Initial repository setup. Model training, evaluation, inference, API endpoints and frontend integration are planned.

## Disclaimer

Predictions are experimental model outputs and are not medical diagnoses.
