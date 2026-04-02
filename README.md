<div align="center">

<br/>

```
██╗     ██╗   ██╗███╗   ██╗ ██████╗     ██████╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗
██║     ██║   ██║████╗  ██║██╔════╝    ██╔════╝██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗
██║     ██║   ██║██╔██╗ ██║██║  ███╗   ██║     ███████║██╔██╗ ██║██║     █████╗  ██████╔╝
██║     ██║   ██║██║╚██╗██║██║   ██║   ██║     ██╔══██║██║╚██╗██║██║     ██╔══╝  ██╔══██╗
███████╗╚██████╔╝██║ ╚████║╚██████╔╝   ╚██████╗██║  ██║██║ ╚████║╚██████╗███████╗██║  ██║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝
```

# Lung Cancer Risk Prediction Using Machine Learning

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=for-the-badge)](https://shap.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

> *Predicting lung cancer risk from patient demographics, lifestyle factors, and clinical symptoms — with full model explainability.*

<br/>

---

</div>

## 📌 Overview

Lung cancer remains one of the leading causes of cancer-related mortality worldwide. Early detection and risk stratification can dramatically improve patient outcomes by enabling timely clinical intervention.

This case study applies a **full production-grade ML pipeline** — from raw data to clinically actionable risk scores — using 4,163 patient records across 14 features. The pipeline covers supervised classification, unsupervised clustering, hyperparameter tuning, model explainability, and a three-tier clinical risk scoring system.

**Models trained and compared:**

| Model | Type | Purpose |
|-------|------|---------|
| 🌲 Random Forest (baseline) | Ensemble | Primary classifier |
| 📉 Logistic Regression | Linear | Interpretable baseline |
| 🔧 Random Forest (tuned) | Ensemble + GridSearchCV | Optimised classifier |
| ⚡ XGBoost | Gradient Boosting | Industry-standard tabular ML |
| 🎯 SVM (RBF kernel) | Kernel method | High-dimensional margin classifier |
| 🔵 K-Means Clustering | Unsupervised | Patient risk stratification |

<br/>

---

## 📂 Repository Structure

```
lung-cancer-risk-prediction/
│
├── 📓 CASESTUDY_FINAL_CLEAN.ipynb     ← Main notebook — run this
├── 📊 lung_cancer_data/               ← Patient cohort CSV files
│   ├── cohort_1.csv
│   ├── cohort_2.csv
│   └── ...
├── 📄 README.md                       ← You are here
│
└── outputs/
    ├── 📈 figures/                    ← All generated plots (PNG)
    ├── 📋 feature_importance.csv
    └── 📋 results_summary.csv
```

<br/>

---

## 📦 Dataset

**Multi-cohort Lung Cancer Patient Dataset**

| Detail | Value |
|--------|-------|
| 👤 Records | 4,163 patient encounters |
| 🧬 Features | 14 (13 predictors + 1 target) |
| 🎯 Target | `lung_cancer` — binary (0 = Low risk, 1 = Medium–High risk) |
| ⚖️ Class Balance | ~79% positive (imbalanced) — handled via stratified splits |
| 🔗 Source | Multiple hospital CSV cohorts, concatenated |

### Feature Summary

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Patient age (range: 14–87) |
| `gender` | Encoded Integer | 0 / 1 / 2 |
| `smoking` | Integer | Smoking habit indicator |
| `alcohol` | Integer | Alcohol consumption |
| `air_pollution` | Float | Exposure level (imputed) |
| `occupational_hazards` | Float | Workplace hazard score (imputed) |
| `genetic_risk` | Float | Genetic predisposition (imputed) |
| `chest_pain` | Integer | Clinician-rated severity (1–9) |
| `cough` | Float | Clinician-rated severity (imputed) |
| `shortness_of_breath` | Integer | Clinician-rated severity |
| `fatigue` | Float | Clinician-rated severity (imputed) |
| `wheezing` | Integer | Wheezing severity |
| `swallowing_difficulty` | Float | Clinician-rated (imputed) |
| `lung_cancer` | Integer (Target) | 0 = No/Low · 1 = Yes/Medium–High |

> **Note:** Features marked *(imputed)* contain ~48–52% missing values originating from different cohort CSV files. All are filled using column mean before training.

<br/>

---

## ⚙️ Setup

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost shap
```

### Clone & Launch

```bash
git clone https://github.com/YOUR_USERNAME/lung-cancer-risk-prediction.git
cd lung-cancer-risk-prediction
jupyter notebook CASESTUDY_FINAL_CLEAN.ipynb
```

> **Google Colab?** Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload Notebook → select `CASESTUDY_FINAL_CLEAN.ipynb`. Upload your CSV cohort files using the Files panel on the left sidebar.

<br/>

---

## ▶️ Pipeline Walkthrough

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

| Section | Description | Output |
|---------|-------------|--------|
| **1** | Imports & Setup | All libraries loaded |
| **2** | Data Loading | Multiple CSVs concatenated → 4,163 rows |
| **3** | Exploratory Data Analysis | Feature descriptions, heatmap, class distribution |
| **4** | Data Preprocessing | Mean imputation, StandardScaler, train/test split (80/20) |
| **5** | Supervised Learning | Random Forest + Logistic Regression trained & evaluated |
| **6** | K-Means Clustering | Elbow method → k=3, cluster profiles & cancer prevalence |
| **7** | Conclusion (baseline) | Summary of initial findings |
| **8** | 10-Fold Cross-Validation | Generalisation check — mean ± std accuracy |
| **9** | GridSearchCV Tuning | Optimal RF hyperparameters via 5-fold CV |
| **10** | XGBoost | Sequential boosting with `scale_pos_weight` for imbalance |
| **11** | SVM (RBF) | Kernel classifier with `class_weight='balanced'` |
| **12** | ROC-AUC Comparison | All 5 models on single ROC curve + AUC summary table |
| **13** | SHAP Explainability | Beeswarm, bar, and per-patient force plots |
| **14** | Clinical Risk Scoring | Low / Moderate / High tiers from predicted probabilities |
| **15** | Precision-Recall Curve | F1-optimal threshold selection for imbalanced data |

<br/>

---

## 📊 Results

### Model Performance (Test Set — 833 Samples)

| Model | Accuracy | AUC-ROC | Notes |
|-------|----------|---------|-------|
| 🌲 Random Forest (base) | **97.96%** | — | Strong ensemble baseline |
| 📉 Logistic Regression | 92.56% | — | Weaker recall on minority class |
| 🔧 Random Forest (tuned) | ≥ RF base | Highest | GridSearchCV-optimised |
| ⚡ XGBoost | Competitive | High | Native imbalance handling |
| 🎯 SVM (RBF) | Competitive | High | Balanced class weights |

### Cross-Validation (10-Fold Stratified)

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| Random Forest | ~0.97+ | < 0.02 |
| Logistic Regression | ~0.92+ | < 0.02 |

> Low standard deviation confirms models generalise well — accuracy is not inflated by a lucky split.

### K-Means Cluster Interpretation

| Cluster | Lung Cancer Rate | Patient Profile |
|---------|-----------------|----------------|
| **0** | ~73.5% | Older patients (avg. 54), low symptom severity, moderate risk |
| **1** | **100%** | Younger patients, very high symptom burden (fatigue, SOB) |
| **2** | ~92.2% | Younger patients, high lifestyle risk (smoking, alcohol, hazards) |

### Clinical Risk Tiers

| Risk Tier | Probability Threshold | Recommended Action |
|-----------|----------------------|-------------------|
| 🟢 Low | < 0.40 | Routine annual screening |
| 🟠 Moderate | 0.40 – 0.70 | Follow-up within 3 months + CT scan |
| 🔴 High | ≥ 0.70 | Immediate specialist referral |

<br/>

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to make the best model auditable by clinicians:

- **Beeswarm Plot** — shows the distribution of each feature's impact across all test patients
- **Bar Plot** — ranks features by mean absolute SHAP (global importance)
- **Force Plot** — explains a single high-risk patient's prediction feature by feature

> Top predictive features identified: `genetic_risk`, `smoking`, `air_pollution`, `chest_pain`, `shortness_of_breath`

<br/>

---

## 🧠 Key Concepts

| Term | Definition |
|------|------------|
| **Stratified Split** | Train/test split preserving the 79/21 class ratio in both sets |
| **Mean Imputation** | Replacing missing values with the column mean — preserves scale |
| **StandardScaler** | Transforms features to zero mean, unit variance |
| **GridSearchCV** | Exhaustive hyperparameter search with cross-validation |
| **XGBoost** | Gradient boosting — trees built sequentially to correct prior errors |
| **scale_pos_weight** | XGBoost parameter to handle class imbalance (neg/pos ratio) |
| **ROC-AUC** | Measures class separability (1.0 = perfect, 0.5 = random) |
| **F1-Score** | Harmonic mean of Precision & Recall — ideal for imbalanced data |
| **SHAP** | Game-theory attribution of each feature's contribution per prediction |
| **Elbow Method** | Selects optimal k for K-Means by plotting inertia vs. cluster count |

<br/>

---

## 📚 References

[1] Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

[2] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. https://arxiv.org/abs/1603.02754

[3] Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks.* Machine Learning, 20(3), 273–297. https://doi.org/10.1007/BF00994018

[4] Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS 2017. https://arxiv.org/abs/1705.07874

[5] Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

[6] MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.* Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability.

[7] Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357. https://arxiv.org/abs/1106.1813

[8] Fawcett, T. (2006). *An introduction to ROC analysis.* Pattern Recognition Letters, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

[9] Siegel, R. L., Miller, K. D., & Jemal, A. (2023). *Cancer Statistics, 2023.* CA: A Cancer Journal for Clinicians. https://doi.org/10.3322/caac.21763

[10] Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* Journal of the Royal Statistical Society: Series B, 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

<br/>

---

<div align="center">

*Submitted for **22AIE213 – Machine Learning** | Amrita Vishwa Vidyapeetham, Chennai*

<br/>

**Built with clinical purpose, statistical rigour, and full model transparency.**

</div>
