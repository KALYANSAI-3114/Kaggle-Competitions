# 📊 Student Exam Score Prediction – Kaggle Competition

This repository contains my solution for a **Kaggle competition** focused on predicting students' exam scores based on demographic, academic, and behavioral features.

The project demonstrates a complete **end-to-end machine learning pipeline**, including data preprocessing, feature engineering, model training, evaluation, and submission generation.

---

## 🧠 Problem Statement

The goal of this competition is to predict the **exam score** of students using features such as:
- Study hours
- Class attendance
- Sleep patterns
- Course type
- Study methods
- Internet access
- Exam difficulty

This is a **supervised regression problem**.

---

## 📂 Repository Structure
```
├── code.ipynb          # Complete ML pipeline (EDA → Modeling → Prediction)
├── requirements.txt    # Python dependencies
├── submission.csv      # Final Kaggle submission file
└── README.md           # Project documentation
```

---

## 📊 Dataset

The dataset is provided by Kaggle.

🔗 **Dataset Link:**  
👉 https://www.kaggle.com/competitions/playground-series-s6e1/data


### Dataset Description
- **Train data:** Features + target (`exam_score`)
- **Test data:** Features only (used for final prediction)
- Large-scale tabular dataset with both numerical and categorical features

---

## ⚙️ Tech Stack & Libraries

- Python 
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

All required libraries are listed in `requirements.txt`.

---

## 🔄 Machine Learning Pipeline

1. **Data Cleaning**
   - Missing value checks
   - Data type validation

2. **Feature Encoding**
   - One-Hot Encoding for nominal categorical features
   - Label Encoding for ordinal features
   - Frequency Encoding for high-cardinality features

3. **Outlier Handling**
   - IQR-based detection on continuous numerical features

4. **Feature Scaling**
   - Applied only to continuous numerical variables

5. **Model Training**
   - Baseline: Linear Regression
   - Ensemble Models: Random Forest
   - Final Model: **XGBoost Regressor**

6. **Evaluation**
   - Mean Absolute Error (MAE)
   - R² Score

7. **Prediction & Submission**
   - Predictions generated on test dataset
   - Output saved as `submission.csv`

---

## 📈 Model Performance

| Model | MAE ↓ | R² ↑ |
|------|------|------|
| Linear Regression | ~8.00 | ~0.72 |
| Random Forest | ~7.71 | ~0.74 |
| **XGBoost (Final)** | **~7.60** | **~0.75** |

The XGBoost model provided the best overall performance and was selected for the final submission.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook
Open `code.ipynb` and run all cells to reproduce results and predictions.

---

## 📤 Kaggle Submission

The final predictions are saved in:
```
submission.csv
```

This file follows Kaggle's required format:
```
id,exam_score
```

Upload this file directly to the competition submission page.

---

## 🎯 Key Learnings

- Importance of consistent preprocessing between train and test data
- Proper handling of categorical features in large datasets
- Model comparison and selection for regression problems
- When to stop optimizing and avoid overfitting

---

## 👤 Author

**Kalyan Sai Atchi**  
Machine Learning Enthusiast | Kaggle Practitioner

