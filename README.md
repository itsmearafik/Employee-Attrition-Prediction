# Employee Attrition Prediction — README

---

## Project Overview

**Business goal:** Build a model to predict which employees are likely to leave so HR can take proactive retention actions.  
**Target variable:** **Attrition** (yes/no).

![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/target_distribution.png)
---

## Table of Contents

- Project summary
- Dataset & columns
- Project structure
- Methodology
- Data preprocessing
- Feature engineering
- Modeling & evaluation
- Deployment
- Key findings & recommendations
- Future work
- How to reproduce

---

## Project summary

This project performs end-to-end analysis and modeling to predict employee attrition. It includes data understanding, preprocessing, exploratory data analysis (EDA), feature engineering, model training and tuning, evaluation, and exporting the final model for deployment.

![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/age_vs_attrition.png)

---

## Dataset & columns

Key concepts:

- **Attrition**: target (yes/no).
![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/attrition_distribution.png)

- **Attrition Label / Attrition Count**: auxiliary fields for analysis (e.g., aggregated counts or labels).  
Other typical columns used: Age, Gender, Education Field, Job Role, Department, Job Satisfaction, Work Life Balance, Monthly Income, average_monthly_hours, Hire Date / Attrition Date / Dynamic Date, Employee No., Placeholder.

Note: The datasets (HR_Attrition Data and HR_New_data) were joined using this column **Employee No.** and **emp_id** before proceeding with the model development.

---

## Project structure

Example folder layout:

- data/
  - HR_Attrition Data.csv (original dataset)
  - HR_New_data.csv
  - merged_dataset.csv
- model/
  - hr_attrition_model.joblib
- assets/
  - age_vs_attrition.png
  - attrition_distribution.png
  - department_vs_attrition.png
  - dept_distribution.png
  - education_vs_attrition.png
- main.ipynb
- LICENSE
- README.md
- requirements.txt

---

## Methodology

1. Data understanding and cleaning.
2. Exploratory data analysis — identify correlations and patterns in attrition.
3. Feature engineering — create meaningful predictors (e.g., tenure, aggregated satisfaction)
4. Model training — compare baseline and advanced models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).
5. Hyperparameter tuning (GridSearch / RandomizedSearch).
6. Model evaluation with relevant metrics.
7. Save & export final model for deployment.

![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/department_vs_attrition.png)

---

## Data preprocessing

- Check and handle missing values (imputation or drop).  
- Convert categorical features to numeric (one‑hot or label encoding depending on model).  
- Parse dates (Attrition Date, Dynamic Date, Hire Date); derive features such as tenure, days since last review, days since hire.  
- Drop identifiers and irrelevant columns (Employee No., Placeholder).  
- Scale numeric features (StandardScaler or MinMaxScaler) when required by the model.

![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/education_vs_attrition.png)
---

## Feature engineering

Examples of engineered features:

- Tenure = (Dynamic Date or Attrition Date) − Hire Date in days/years.
- Average satisfaction per department or job role.  
- Interaction terms (Job Satisfaction × Work Life Balance).  
- Aggregated counts: Attrition_Count by department/role.

---

## Exploratory Data Analysis (EDA)

Focus areas:

- Attrition rate overall and by subgroup (role, department, gender, education).  
- Relationships between attrition and Job Satisfaction, Work Life Balance, Monthly Income, average_monthly_hours.  
- Visualizations: histograms, boxplots, bar charts, correlation heatmap, and confusion-matrix-style analysis for model outputs.

![](https://github.com/itsmearafik/Employee-Attrition-Prediction/blob/main/assets/dept_distribution.png)

---

## Modeling & evaluation

Models evaluated:

- Logistic Regression (baseline)  
- Decision Tree  
- Random Forest  
- Gradient Boosting (e.g., XGBoost / LightGBM)

Evaluation metrics:

- **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**.  
Examine the confusion matrix to understand false positives vs false negatives; for attrition prediction, prioritize **Recall** (catching employees likely to leave) while controlling precision to avoid unnecessary interventions.

Hyperparameter tuning: GridSearchCV or RandomizedSearchCV with cross-validation.

---

## Final model export & deployment

- Save model and preprocessing pipeline together (joblib or pickle).  
- Provide an inference script that accepts new employee records, applies preprocessing, and returns probability and binary prediction.  
- Integration options: batch predictions, REST API (Flask/FastAPI), or direct integration with HR systems.

Example save:

```python
from joblib import dump
dump(pipeline, "models/final_model.joblib")
```

---

## Key findings & recommendations

- Most predictive features: e.g., **Job Satisfaction**, **Work Life Balance**, **Tenure**, **Monthly Income**, **Job Role**.  
- Actionable recommendations:
  - Targeted retention programs for high-risk roles or departments.
  - Improve job satisfaction and work–life balance initiatives.
  - Conduct stay interviews and compensation reviews for at-risk employees.

---

## Future work

- Integrate live HR data for continuous retraining and monitoring.  
- Build a feedback loop to capture post-intervention outcomes and improve model calibration.  
- Implement model explainability (SHAP/LIME) for per-employee actionable insights.  
- Monitor for concept drift and update features/models accordingly.

---

## How to reproduce

1. Clone repository.  
2. Create environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Place raw dataset in data/ and run preprocessing:

```bash
python src/data_prep.py --input data/raw/dataset.csv --output data/processed/
```

4. Train model:

```bash
python src/train.py --data data/processed/train.csv --model_out model/hr_attrition_model.pkl
```

5. Evaluate:

```bash
python src/evaluate.py --model model/hr_attrition_model.pkl --test data/processed/test.csv
```

---

Dependencies

- Python 3.8+  
- pandas, numpy, scikit-learn, matplotlib/seaborn, joblib

---

License
License (MIT) and data usage terms.

---
