# 🩺 Diabetes Risk Prediction Dashboard

> A machine learning platform that predicts diabetes risk using classifiers built **from scratch** — no ML libraries — trained on 100,000+ real patient records from the CDC.

Built by **Guillermo Novillo, Jorge, and Anvi** as a final project for DSA at the University of Florida.

---

## 📌 Overview

Most ML projects call `sklearn.fit()` and call it a day. This one doesn't.

We implemented **Logistic Regression** and **Linear SVM** from the ground up using only NumPy — including gradient descent, sigmoid activation, hinge loss, and custom train/test splitting — then deployed them inside an interactive Streamlit dashboard where users can enter their own health data and get a real-time diabetes risk prediction.

The dashboard also benchmarks our custom models against 5 standard scikit-learn classifiers (kNN, RBF SVM, Naive Bayes, LDA, QDA), visualizes feature importance, and includes an animated BMI-by-race scatter plot across years.

---

## ✨ Features

### 🔢 Diabetes Risk Calculator
Enter your age, BMI, HbA1c level, blood glucose, and other health indicators to get an instant prediction from both models.

### 🏆 Model Performance Comparison
Side-by-side benchmarking of 7 classifiers ranked by MSE, Accuracy, and AUC-ROC score with interactive Plotly charts.

### 📊 Feature Importance
Visual breakdown of which health factors most influence the Logistic Regression model's predictions — color-coded by risk impact.

### 🌍 BMI by Race (Animated)
Animated scatter plot showing BMI vs. age distributions across racial groups over time, built with Plotly Express.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Models (from scratch) | NumPy, custom Logistic Regression & Linear SVM |
| Comparison Models | scikit-learn (kNN, SVM, Naive Bayes, LDA, QDA) |
| Data Processing | Pandas, NumPy |
| Frontend / Dashboard | Streamlit, Plotly, Altair |
| Dataset | CDC Diabetes Health Indicators (~100,000 records) |
| Language | Python 3.11 |

---

## 🧠 How the Models Work

### Logistic Regression (from scratch)
- Sigmoid activation function with gradient clipping (`np.clip` to prevent overflow)
- Gradient descent with adaptive learning rate — halves the LR if loss spikes
- Convergence based on tolerance threshold (`tol=1e-8`)
- Trained for up to 10,000 iterations

### Linear SVM (from scratch)
- Hinge loss with L2 regularization (`lambda_param=0.0005`)
- Subgradient descent optimization
- Same convergence logic as Logistic Regression

Both models are serialized with `pickle` after training and loaded by the dashboard for instant predictions.

---

## 🚀 How to Run

> **Python 3.11 required.** Delete any `.pkl` files before running from scratch.

```bash
# 1. Install dependencies
pip install streamlit numpy pandas scikit-learn altair plotly streamlit-option-menu

# 2. Train the custom models (~10 min)
python main.py

# 3. Train models needed for the risk calculator (~2 min)
python train_models.py

# 4. Precompute model comparison data (~30 sec)
python other_models.py

# 5. Launch the dashboard
streamlit run app.py
```

> 💡 If your browser is in dark mode, go to the top-right menu → Settings → select **Light mode** for the best experience.

---

## 📁 Project Structure

```
├── main.py                  # Core ML logic: data loading, custom models, training
├── train_models.py          # Trains & saves Logistic Regression + SVM as .pkl files
├── other_models.py          # Trains sklearn comparison models, saves precomputed results
├── app.py                   # Streamlit dashboard (4 pages)
├── Graph_Testing.py         # Visualization experiments
├── data/
│   └── diabetes_dataset.csv # CDC dataset (~100k records)
├── logistic_regression_model.pkl
└── svm_model.pkl
```

---

## 👥 Team

| Name | GitHub |
|---|---|
| Guillermo Novillo | [@Gnovillo1120](https://github.com/Gnovillo1120) |
| Jorge | — |
| Anvi | — |

---

*University of Florida — Data Structures & Algorithms Final Project, Fall 2025*
