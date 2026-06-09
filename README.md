# Fraud_Detection_
# Comparative Analysis of Machine Learning Models for Financial Fraud Detection in Zimbabwe
## Overview

This repository contains the dataset, implementation code, and experimental outputs used in the research project titled **"Comparative Analysis of Machine Learning Models for Financial Fraud Detection in Zimbabwe"** The study investigates how machine learning techniques can be applied to detect fraudulent financial transactions in Zimbabwe’s rapidly expanding digital payment environment.

With the increasing adoption of digital financial services such as mobile money, online banking, and point-of-sale (POS) transactions, financial fraud has become a significant challenge for financial institutions. Traditional rule-based fraud detection systems often struggle to identify complex and evolving fraud patterns. Machine learning methods offer a promising alternative by learning behavioural patterns from transaction data and identifying anomalies that indicate fraudulent activity.

The primary objective of this project is to evaluate the effectiveness of several supervised machine learning algorithms in detecting fraudulent transactions and to provide interpretable insights into the factors driving fraud predictions. The study also incorporates explainable artificial intelligence techniques to improve transparency and model interpretability.

---

## Research Objectives

The research was guided by the following objectives:

1. Explore patterns and trends in financial fraud within Zimbabwe’s digital financial sector.
2. Implement and evaluate multiple fraud detection models, including Logistic Regression,
Random Forest, XGBoost, LightGBM and a Soft Voting Ensemble, using performance
metrics such as precision, recall, F1-score, ROC-AUC and PR-AUC.
3. Apply SHAP-based explainability analysis to interpret model predictions and compare
feature importance across the implemented models.
4. Recommend model(s) for practical implementation in Zimbabwe’s financial platforms
based on the empirical findings.

---

## Dataset Description

The dataset used in this study consists of **100,000 anonymized financial transactions** representing both legitimate and fraudulent activities within digital payment systems. The dataset has been carefully processed to remove personally identifiable information while preserving behavioural and transactional characteristics necessary for machine learning analysis.

### Target Variable

- **Fraud_Label**
  - `0` = Legitimate transaction
  - `1` = Fraudulent transaction

### Example Features

- Transaction_Amount  
- Transaction_Type  
- Transaction_Channel  
- Authorization_Mode  
- Response_Code  
- Location  
- Customer_History_Frequency  
- Transaction timestamp (hour)  
- Location-based fraud risk indicators  
- Transaction behaviour ratios  

Additional engineered features such as fraud-rate encodings and behavioural ratios were created to capture deviations from normal transaction patterns.

---

## Machine Learning Models Implemented

The study evaluates the performance of several supervised learning algorithms commonly used in fraud detection:

- Logistic Regression
- Random Forest
- Extreme Gradient Boosting (XGBoost)
- Light Gradient Boosting Machine (LightGBM)
- Soft Voting Ensemble Model

Tree-based ensemble models demonstrated strong performance in modelling complex non-linear fraud patterns.

---

## Model Evaluation

Model performance was evaluated using several standard classification metrics:

- **Accuracy** – overall prediction correctness  
- **Precision** – proportion of predicted fraud cases that were actually fraudulent  
- **Recall** – ability of the model to correctly detect fraudulent transactions  
- **F1-Score** – harmonic mean of precision and recall  
- **ROC-AUC** – ability of the model to distinguish between fraudulent and legitimate transactions  

These metrics provide a comprehensive assessment of fraud detection performance.

---

## Explainable Artificial Intelligence (XAI)

To enhance transparency and interpretability, this study incorporates **SHapley Additive exPlanations (SHAP)**. SHAP is an explainable AI technique that quantifies the contribution of each feature to model predictions.

The SHAP analysis identified key factors influencing fraud predictions, including:

- Customer transaction behaviour patterns
- Historical fraud rates associated with transaction categories
- Authorization response anomalies
- Temporal transaction patterns
- Location-based fraud risk indicators

This interpretability enables financial institutions to better understand and trust model predictions.

---

### Directory Explanation

- **data/** – Contains the dataset used for training and evaluation  
- **code/** – Python scripts and notebooks implementing the fraud detection pipeline  
- **outputs/** – Generated figures and model evaluation results  
- **requirements.txt** – List of required Python libraries  

---

## Installation

Install required Python libraries using:
requirements.txt


The notebook includes the full pipeline:

1. Data loading
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Explainable AI analysis

---

## Research Contribution

This project contributes to research on financial fraud detection in emerging digital economies by:

- Applying machine learning methods to digital financial transaction data
- Comparing multiple predictive algorithms
- Integrating explainable AI techniques for model transparency
- Providing insights into behavioural patterns associated with fraudulent transactions

The findings demonstrate how data-driven approaches can strengthen fraud detection systems in Zimbabwe’s digital financial ecosystem.

---

## License

This repository is shared for **academic and research purposes only**.

---

## Author

**Maninga Blessing**  
MSc Big Data  

Supervisor: **Prof. Naison Gasela**
## Repository Structure
