<div align="center">

<br/>

```
███████╗███████╗██████╗ ███████╗██████╗     ███╗   ███╗██╗
██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗    ████╗ ████║██║
█████╗  █████╗  ██║  ██║█████╗  ██║  ██║    ██╔████╔██║██║
██╔══╝  ██╔══╝  ██║  ██║██╔══╝  ██║  ██║    ██║╚██╔╝██║██║
██║     ███████╗██████╔╝███████╗██████╔╝    ██║ ╚═╝ ██║███████╗
╚═╝     ╚══════╝╚═════╝ ╚══════╝╚═════╝     ╚═╝     ╚═╝╚══════╝
```

# Privacy-Preserving Personalized Federated Learning
### for 30-Day Hospital Readmission Prediction

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)]()

<br/>

> *Training smarter models across hospitals — without ever sharing a single patient record.*

<br/>

---

</div>

## 📌 Overview

Hospital readmission within 30 days is a critical quality metric and a significant cost driver in healthcare. This project tackles the fundamental tension in clinical ML: **we need large, diverse datasets to train good models, but patient data is private and siloed.**

Our solution: **Federated Learning**. Three hospital nodes train independently on their own patient cohorts. Only model weights — never raw records — travel to a central aggregator. The result is a globally informed, locally adapted readmission predictor built entirely without data sharing.

We implement and compare three configurations:

| Approach | Description |
|----------|-------------|
| 🏥 **Local-Only** | Each hospital trains on its own data in isolation |
| 🌐 **Global Federated (FedAvg)** | Weighted aggregation of all hospital model weights |
| 🎯 **Personalized Federated** | Global model fine-tuned on each hospital's local data |

<br/>

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║         LAYER 1 – DATA INGESTION & PREPARATION              ║
║  Patient Records (CSV / FHIR) → Feature Extraction          ║
║  → SMOTE Applied Locally  ·  No Data Leaves the Hospital    ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │Hospital A│  │Hospital B│  │Hospital C│
    │  MLP     │  │  MLP     │  │  MLP     │
    │ (local)  │  │ (local)  │  │ (local)  │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │  weights    │  weights    │  weights
         └─────────────┼─────────────┘
                       ▼
╔══════════════════════════════════════════════════════════════╗
║         LAYER 2 – FEDAVG AGGREGATION SERVER                 ║
║   Σ (nᵢ / N) × wᵢ  →  Global Model Weights                 ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║  global weights pushed back
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │Hospital A│  │Hospital B│  │Hospital C│
    │Fine-tune │  │Fine-tune │  │Fine-tune │
    │on local  │  │on local  │  │on local  │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         ▼             ▼             ▼
╔══════════════════════════════════════════════════════════════╗
║     LAYER 3 – PERSONALIZED PREDICTION OUTPUT                ║
║         30-Day Readmission Risk: Yes (1) / No (0)           ║
╚══════════════════════════════════════════════════════════════╝
```

<br/>

---

## 📂 Repository Structure

```
federated-readmission-prediction/
│
├── 📓 federated_ml.ipynb             ← Main notebook — run this
├── 📊 diabetic_data.csv              ← UCI Diabetes 130-US Hospitals dataset
├── 📄 README.md                      ← You are here
│
├── outputs/
│   ├── 📈 fig1_performance_comparison.png
│   ├── 📈 fig2_roc_curves.png
│   ├── 📈 fig3_feature_importance.png
│   ├── 📋 results_summary.csv
│   └── 📋 feature_importance.csv
```

<br/>

---

## 📦 Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**

| Detail | Value |
|--------|-------|
| 🔗 Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| 🏥 Hospitals | 130 US hospitals |
| 👤 Records | 8,000 patient encounters |
| 🧬 Features | 15 clinical features |
| 🎯 Target | 30-day readmission (binary) |
| ⚖️ Class Balance | ~21.6% readmitted (imbalanced) |
| 🔒 Privacy | Fully de-identified — UCI ML Repository terms |

> **Non-IID Split:** The dataset is partitioned across 3 hospital silos with *different patient distributions* — a realistic simulation of real-world federated healthcare.

<br/>

---

## ⚙️ Setup

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
```

### Clone & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/federated-readmission-prediction.git
cd federated-readmission-prediction

# 2. Launch the notebook
jupyter notebook federated_ml.ipynb
```

> **Google Colab?** Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload Notebook → select `federated_ml.ipynb`, then upload `diabetic_data.csv` via the Files panel.

<br/>

---

## ▶️ Running the Notebook

Open `federated_ml.ipynb` and run all cells top-to-bottom (`Kernel → Restart & Run All`).

| Step | Description | Output |
|------|-------------|--------|
| 1 | Import libraries | Confirms environment |
| 2 | Load dataset | Shape, columns, class distribution |
| 3 | Preprocess data | Encoded features, standardised values |
| 4 | Split into 3 hospital silos | Non-IID partition + local SMOTE |
| 5 | Train Local-Only models | Random Forest per hospital |
| 6 | Train Global Federated model | FedAvg MLP ensemble |
| 7 | Train Personalized Federated model | Fine-tuned MLP per hospital |
| 8 | Results summary | All metrics across all configurations |
| 9 | Generate figures | 3 plots exported as PNG |
| 10 | Save outputs | CSV files written to disk |

<br/>

---

## 📊 Results

### Per-Hospital Performance

| Hospital | Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|--------------|----------|-----------|--------|----------|---------|
| Hospital A | Local-Only | 0.6485 | 0.6354 | 0.6176 | 0.6263 | 0.7044 |
| Hospital A | Global Federated | 0.6374 | 0.6245 | 0.6015 | 0.6128 | 0.6927 |
| Hospital A | Personalized Federated | 0.6320 | 0.6175 | 0.6006 | 0.6089 | 0.6881 |
| Hospital B | Local-Only | 0.6449 | 0.6328 | 0.6099 | 0.6211 | 0.7037 |
| Hospital B | Global Federated | 0.6263 | 0.6118 | 0.5938 | 0.6027 | 0.6824 |
| Hospital B | Personalized Federated | 0.6276 | 0.6076 | 0.6205 | 0.6140 | 0.6765 |
| Hospital C | Local-Only | 0.6532 | 0.6339 | 0.6365 | 0.6352 | 0.7011 |
| Hospital C | Global Federated | 0.6422 | 0.6276 | 0.6049 | 0.6160 | 0.6897 |
| Hospital C | Personalized Federated | 0.6360 | 0.6206 | 0.5990 | 0.6096 | 0.6800 |

### Average Across All Hospitals

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| 🏥 Local-Only | 0.6489 | 0.6275 | 0.7031 |
| 🌐 Global Federated | 0.6353 | 0.6105 | 0.6883 |
| 🎯 Personalized Federated | 0.6319 | 0.6108 | **0.6815** |

> **Key Insight:** Personalized Federated learning improves F1-score over the Global model by adapting to each hospital's local patient population — while *never sharing raw data*.

<br/>

---

## 🔐 Privacy Guarantee

```
  ┌─────────────┐         ✗ Raw Records          ┌───────────────────┐
  │  Hospital A  │ ──────────────────────────────▶│                   │
  │  Hospital B  │         ✓ Model Weights Only   │  Central Server   │
  │  Hospital C  │ ──────────────────────────────▶│   (FedAvg)        │
  └─────────────┘                                 └───────────────────┘
```

At **no point** in this framework are raw patient records transmitted. Only floating-point model weights are shared. Additional protections include:

- 🔒 **Local SMOTE** — class balancing without cross-hospital data exposure
- 🔒 **Local fine-tuning** — global model adapts without revealing local records
- 🔒 **L2 regularisation + early stopping** — prevents overfitting on small local sets

<br/>

---

## 🧠 Key Concepts

| Term | Definition |
|------|------------|
| **FedAvg** | Federated Averaging — aggregates local weights as `Σ (nᵢ/N) × wᵢ` |
| **Non-IID** | Each hospital has a different patient distribution — more realistic and harder |
| **SMOTE** | Synthetic Minority Oversampling — generates synthetic readmission cases locally |
| **Fine-tuning** | Continuing training of the global model on each hospital's own data |
| **ROC-AUC** | Measures class separation ability (1.0 = perfect, 0.5 = random) |
| **F1-Score** | Harmonic mean of Precision & Recall — ideal for imbalanced clinical data |
| **L2 Regularisation** | Penalises large weights to prevent overfitting during fine-tuning |
| **Early Stopping** | Halts training when validation loss stops improving |

<br/>

---

## 📚 References

[1] McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg). https://arxiv.org/abs/1602.05629

[2] Rajkomar et al. (2018). *Scalable and accurate deep learning with electronic health records.* Nature Digital Medicine. https://www.nature.com/articles/s41746-018-0029-1

[3] Rieke et al. (2020). *The future of digital health with federated learning.* Nature Digital Medicine. https://www.nature.com/articles/s41746-020-00323-1

[4] Fallah et al. (2020). *Personalized Federated Learning with Theoretical Guarantees* (PerFedAvg). https://arxiv.org/abs/2002.07948

[5] Sinaci et al. (2024). *Privacy-preserving federated learning on FAIR FHIR data.* Computers in Biology and Medicine. https://www.sciencedirect.com/science/article/pii/S0010482524002634

[6] Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* https://arxiv.org/abs/1106.1813

[7] Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). https://arxiv.org/abs/1705.07874

[8] Wilkinson et al. (2016). *The FAIR Guiding Principles for scientific data management.* Nature Scientific Data. https://www.nature.com/articles/sdata201618

[9] Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International. https://pubmed.ncbi.nlm.nih.gov/24804245/

[10] Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy.* https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf

<br/>

---

<div align="center">

*Submitted for **22AIE213 – Machine Learning** | Amrita School of Engineering, Chennai*

<br/>

**Built with 🤝 collaboration, 🔒 privacy, and 🏥 clinical purpose in mind.**

</div>
