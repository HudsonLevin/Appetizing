# Food Appeal Classification Model 🍔

An end-to-end Computer Vision pipeline engineered to distinguish between "Appetizing" and "Unappetizing" food images based on crowd-sourced human preference data. 

🏆 **Achievement:** Achieved a **Top 37% finish (Ranked 20th out of 55)** with an official validation accuracy of **56%** in a peer group machine vision classification challenge.

---

## 🧠 Model Architecture & Tech Stack
- **Language:** Python 3.8+
- **Deep Learning Framework:** TensorFlow / Keras (MobileNetV2 Transfer Learning)
- **Core Libraries:** OpenCV (`opencv-python`), NumPy, Matplotlib, Scikit-learn

---

## 📂 Repository Structure & Core Modules

The project consists of the following key Python scripts located under the `superappetizing` directory:

- **`Apetizing.py`**: The core application script that loads the trained model and processes food image data for classification.
- **`apetizing_model.keras`**: The final optimized weights and model architecture saved after training, achieving the peak 56% accuracy.
- **`confusionmatrix.py`**: A model evaluation utility used to calculate and plot the confusion matrix, precision, recall, and F1-score to analyze model performance across classes.
- **`longkong.py`**: A development script utilized for hyperparameter tuning experiments, cross-validation runs, and testing alternative data augmentation pipelines.
- **`testquestionairs.py`**: A validation script designed to evaluate the model's predictions against a dedicated test set derived from human questionnaire data to verify alignment with real-world human preferences.

---

## ⚡ Key Engineering & Problem Solving
- **Edge-Case Dataset Strategy:** Curated an advanced dataset via Pinterest and Google, introducing extreme negative samples (e.g., dropped/ruined food) against standard positive samples to significantly sharpen the model's decision boundaries.
- **False-Positive Mitigation:** Discovered and resolved feature extraction errors caused by commercial studio lighting and professional compositions, which initially misled the model.
- **Visual Diagnostics:** Implemented custom visual validation checkpoints within the codebase to output side-by-side prediction comparisons, enabling continuous visual inspection against ground truth data throughout training.

---

## 🚀 Installation & How to Run

### 1. Prerequisites
Make sure you have Python 3.8 or higher installed on your machine. It is highly recommended to use a virtual environment (`venv`).

### 2. Install Dependencies
Open your terminal and install the required packages using the following command:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
