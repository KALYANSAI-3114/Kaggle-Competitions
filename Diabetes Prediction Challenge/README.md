# 🩺 Diabetes Prediction using Machine Learning

This project aims to predict whether an individual is likely to have diabetes based on various health indicators using Machine Learning models. The dataset is taken from the Kaggle Playground Series, and the model outputs probability scores for diabetes diagnosis for each person in the test dataset.

---

## 📊 Dataset Information

Dataset Source:  
🔗 https://www.kaggle.com/competitions/playground-series-s5e12/data

Files included:

| File | Description |
|------|-------------|
| **train.csv** | Contains health features + target (`diagnosed_diabetes`) |
| **test.csv** | Contains health features without target |
| **sample_submission.csv** | Format reference for final output |

**Target column:**  
`diagnosed_diabetes` → 1 (Diabetic), 0 (Not Diabetic)

---

## 🛠 Project Workflow

### 1. Data Understanding & Cleaning
- Loaded train & test CSV files
- Checked column types, missing values, data distribution
- Converted categorical variables to numeric
  - `gender` mapped: Male=1, Female=0, Other=2
  - One-Hot Encoding applied for features like ethnicity, employment, smoking status, income level etc.

### 2. Feature Engineering & Preprocessing
- Removed `id` from training features
- Converted all columns to numerical format
- Ensured train & test column alignment after encoding
- Visualized feature correlations using heatmap

### 3. Model Training
- Split train data: **80% training + 20% validation**
- Trained ML Model (RandomForest/XGBoost/LightGBM)
- Evaluated using Accuracy, F1-score, Precision, Recall

**Sample Metrics**  
Accuracy: ~0.68
Better recall for diabetics (class 1)


### 4. Prediction & Submission
- Model predicted probability using `predict_proba()`
- Created submission file:

id, diagnosed_diabetes
10001, 0.72345
10002, 0.26410
...


- Exported as `submission.csv` and uploaded to Kaggle

---

## 🧰 Technologies Used

| Tool | Purpose |
|------|----------|
| Python | Programming |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | Visualization |
| Scikit-Learn / XGBoost / LightGBM | Machine Learning |
| Jupyter/Kaggle Notebook | Execution environment |

---

## 📂 Project Structure

📦 Diabetes-Prediction-ML

┣ 📄 train.csv

┣ 📄 test.csv

┣ 📄 code.ipynb

┣ 📄 submission.csv

┗ 📄 README.md



---

## 🚀 Future Improvements

- Hyperparameter tuning for better accuracy
- Try ensemble/blended models
- Feature selection & importance analysis
- Model deployment using Streamlit/Flask
- SHAP explainability

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be used to analyze health metrics and predict the likelihood of diabetes. After preprocessing, training, evaluation, and prediction, a final submission file was generated for Kaggle. This serves as a good foundation for classification problems in healthcare analytics.

---

### ⭐ If you like this project, give the repo a star!


