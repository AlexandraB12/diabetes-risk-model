# 📊 Diabetes Analysis — Exploring Patient Data and Building Predictive Models

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 🗂️ Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Domain Knowledge](#domain-knowledge)
- [Tools & Libraries](#tools--libraries)
- [How to Run](#how-to-run)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Modeling & Grid Search](#modeling--grid-search)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Key Takeaways](#key-takeaways)
- [Next Steps](#next-steps)
- [Author](#author)

---

## 🧠 Overview
This project performs a comprehensive **Exploratory Data Analysis (EDA)** and **predictive modeling** on a diabetes dataset.  
We analyze patient data such as **age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level** to predict diabetes using a **Random Forest Classifier**.  

The project demonstrates:
- ✅ Data cleaning & preparation  
- 📊 Static visualizations with **Matplotlib** and **Seaborn**  
- 🎨 Feature importance analysis  
- 🤖 Predictive modeling with **Random Forest** and **GridSearchCV hyperparameter tuning**  

---

## 📂 Dataset
- Approximately **96,000 patient records** 
- Features include:
  - 👥 Demographics: age, gender  
  - ⚕️ Health indicators: BMI, hypertension, heart disease  
  - 🚬 Lifestyle: smoking history  
  - 🩸 Lab measurements: HbA1c, blood glucose  
- Outcome: `diabetes` (0 = non-diabetic, 1 = diabetic)  

| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `gender` | Patient gender (Male/Female) |
| `hypertension` | Binary indicator (0 = no, 1 = yes) |
| `heart_disease` | Binary indicator (0 = no, 1 = yes) |
| `smoking_history` | Smoking history categories (`never`, `current`, `former`, `ever`, `not current`, `No Info`) |
| `bmi` | Body Mass Index |
| `HbA1c_level` | Glycated Hemoglobin Level |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | Target variable: diabetes (0 = no, 1 = yes) |

**Key Stats:**  
- 🧮 **Age:** Min 0.08, Max 80, Mean 41.8  
- ⚖️ **BMI:** Mean 27.3, Std 6.77  
- 📈 **Diabetes prevalence:** 9% positive, 91% negative  

> Dataset is for **synthetic/educational purposes**.

---

## 🎯 Objectives
1. Explore patterns and correlations in the dataset  
2. Preprocess data: handle categorical variables, scale numerical features  
3. Apply **SMOTE** to balance the classes  
4. Train a **Random Forest Classifier** and tune hyperparameters  
5. Evaluate model performance and interpret **feature importance**  

---

## 🧰 Tools & Libraries

### Python Installations
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn > /dev/null 2>&1
```

### Python Imports
```python
import pandas as pd            # Data manipulation
import numpy as np             # Numerical computations
import matplotlib.pyplot as plt  # Static visualizations
import seaborn as sns            # Statistical visualizations

# Scikit-learn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imbalanced-learn modules
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline
```

**Libraries Used:**
- 🐼 `pandas`, `numpy` → data manipulation  
- 📊 `matplotlib`, `seaborn` → visualization 
- 🤖 `scikit-learn` → modeling & metrics 
- ⚖️ `imbalanced-learn` → handling imbalanced classes

---

## ⚙️ How to Run

### Locally
-- Clone the repository

```bash
git clone https://github.com/AlexandraB12/diabetes-risk-model.git
```
Navigate into the project folder

``` bash
cd diabetes-risk-model

```
Launch the Jupyter Notebook

jupyter notebook main.ipynb


💡 Tip: Ensure all libraries are installed via pip install -r requirements.txt or individually as listed above.

---

##📈 Results & Visuals

<details>
<summary>🧹 Data Cleaning & Preparation</summary>

- Removed invalid or inconsistent values (e.g., `gender = Other`)  
- Checked for missing values → **none found**  
- Structured numeric (`age, bmi, HbA1c_level, blood_glucose_level`) and categorical (`gender, smoking_history`) features  
- Prepared dataset for **scaling and one-hot encoding**  

**Result:** Clean, structured dataset ready for modeling

</details>

<details>
<summary>👥 Demographic Overview</summary>

- **Age distribution:** wide range, median ~43 years  
- **Gender distribution:** majority male/female, 18 records removed for 'Other'  
- **Health conditions:** small percentage with hypertension (~8%) or heart disease (~4%)  

**Insight:** Dataset reflects general adult population with typical chronic disease prevalence

</details>

<details>
<summary>💉 Clinical & Lifestyle Features</summary>

- Explored **BMI, HbA1c, blood glucose** levels for distributions  
- Examined **smoking history** categories (`current`, `past_smoker`, `non-smoker`)  
- Checked correlations between **age, BMI, glucose levels, and HbA1c**  

**Insight:** Strong positive correlation between blood glucose and HbA1c, moderate correlation with BMI

</details>

<details>
<summary>👥 Binary Variables</summary>

- Hypertension, Heart Disease, Diabetes  
- Visualized using countplots to see class distribution  
- Identified class imbalance in Diabetes (~9% positive, 91% negative)  

**Insight:** Minority class requires balancing for modeling

</details>

<details>
<summary>💬 Categorical Variables</summary>

- Gender and Smoking History  
- Countplots used to analyze frequency of categories  
- Recategorized smoking history into: non-smoker, past smoker, current smoker  

**Insight:** Gender and smoking distributions are relatively balanced (except removed 'Other')

</details>

<details>
<summary>📊 Continuous Variables</summary>

- Age, BMI, HbA1c_level, Blood Glucose  
- Histograms and boxplots used to explore distributions  
- Identified outliers in BMI and blood glucose  

**Insight:** Continuous variables show expected variation; ready for modeling

</details>

<details>
<summary>🔗 Pairplots & Scatterplots</summary>

```python
# Example: pairplot of numerical features colored by diabetes
sns.pairplot(data=df[numeric_columns], hue='diabetes', palette='coolwarm', diag_kind='kde')
plt.show()

# Example: scatter plot age vs BMI
sns.scatterplot(data=df, x='age', y='bmi', hue='diabetes', palette='coolwarm')
plt.show()
```

- Examined relationships between numerical features
- Identified correlations and trends visually

**Insight:**: Age and BMI moderately correlated; blood measurements strongly differentiate diabetic vs non-diabetic

</details> 

<details> 
<summary>🧊 Correlation Heatmaps</summary>

```python
# Correlation matrix
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()
```

- Highlighted strong correlations (e.g., **HbA1c_level & Blood Glucose**)  
- Weak correlation with binary/categorical features  

**Insight:** Blood metrics are key predictors

</details>

<details>
<summary>🔧 Preprocessing & Modeling</summary>

- **ColumnTransformer:**  
  - Scaled numerical features: age, BMI, HbA1c_level, blood_glucose_level, hypertension, heart_disease  
  - One-hot encoded categorical features: gender, smoking_history  

- **Dataset Balancing:** SMOTE for minority class, RandomUnderSampler for majority class  
- **Pipeline:** Preprocessing → Resampling → RandomForestClassifier  
- **Hyperparameter tuning:** GridSearchCV  

**Result:** Model ready for evaluation

</details>

<details>
<summary>📈 Model Evaluation</summary>

- Random Forest achieved **~95.1% accuracy**  

**Precision:**  
- Class 0 (non-diabetic): 0.98  
- Class 1 (diabetic): 0.69  

**Recall:**  
- Class 0: 0.96  
- Class 1: 0.81  

- Confusion matrix and classification report generated  

**Insight:** Model performs well on both classes but minority class (diabetic) has lower precision

</details>

<details>
<summary>🌟 Feature Importance</summary>

| Feature | Importance |
|---------|------------|
| HbA1c_level | 0.44 |
| blood_glucose_level | 0.32 |
| age | 0.14 |
| BMI | 0.06 |
| hypertension | 0.02 |
| heart_disease | 0.01 |
| smoking_history_* | 0–0.01 |
| gender_* | 0 |

**✅ Key Insight:**  
- Blood-related measurements are the most critical predictors of diabetes  
- Age and BMI also contribute significantly  
- Gender and smoking history have minimal influence in this model

</details>

<details>
<summary>💡 Next Steps & Suggestions</summary>

- Collect additional lifestyle and family history features  
- Explore other models: **XGBoost**, **LightGBM**  
- Advanced feature engineering: interaction terms, polynomial features  
- Use **SHAP** for deeper feature interpretability  
- Explore other oversampling/cost-sensitive methods to improve minority class prediction

</details>



---

## 📌 Key Takeaways

- 🩺 **HbA1c** and **blood glucose** are the strongest predictors  
- 👴 **Age** and **BMI** contribute moderately  
- 🚭 Lifestyle factors have minimal impact in this dataset  
- 🌲 Random Forest achieves **~95% accuracy**, robust against imbalanced classes  

## 🔮 Next Steps

- Explore other models: **XGBoost**, **LightGBM**  
- Apply interpretability tools: **SHAP**, **Permutation Feature Importance**  
- Collect additional features: diet, physical activity, family history  
- Feature engineering: interaction terms, polynomial features  
- Explore advanced oversampling/undersampling methods

---

## 🧾 Author
**Alexandra Boudia**  
Data Scientist | Predictive Modeling | AI & ML Practitioner  
🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/alexandra-boudia/)
