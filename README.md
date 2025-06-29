# Credit Risk Probability Model for Alternative Data

## 🚀 Project Overview
**📌Business Context**:  
Bati Bank partners with an eCommerce platform to offer a **buy-now-pay-later (BNPL)** service. This project builds a credit scoring model using transactional data to predict default risk and assign credit scores.

**Goal**:  
Develop an end-to-end ML pipeline that:
1. Creates a proxy for "default risk" using RFM (Recency, Frequency, Monetary) analysis.
2. Predicts risk probabilities and optimal loan terms.

---
## 📁 Project Structure  
```
credit-risk-model/
├── data/
│ ├── raw/ # Original dataset (e.g., Xente Challenge data)
│ └── processed/ # Processed data for modeling
├── notebooks/
│ └── 1.0-eda.ipynb # Exploratory Data Analysis
├── src/
│ ├── data_processing.py # Feature engineering pipeline
│ ├── train.py # Model training script
│ ├── predict.py # Inference script
│ └── api/ # FastAPI deployment
│ ├── main.py
│ └── pydantic_models.py
├── tests/
│ └── test_data_processing.py # Unit tests
├── Dockerfile # Containerization
├── requirements.txt # Python dependencies
└── README.md # This file
```

### Credit Scoring Business Understanding  

### 1. Basel II’s Influence on Model Design  
Basel II mandates rigorous risk measurement and capital allocation. For Bati Bank:  
- **Interpretability**: Regulators require clear documentation of how risk probabilities are derived (e.g., Logistic Regression coefficients or SHAP values for complex models).  
- **Documentation**: Audits may demand evidence that the model aligns with Basel’s default definitions (e.g., using RFM proxies).  

### 2. Proxy Variable Necessity & Risks  
- **Why Proxy?** The dataset lacks a direct "default" label. RFM-based clustering (e.g., low-frequency/low-monetary customers) acts as a proxy for high-risk behavior.  
- **Business Risks**:  
  - **False Positives**: Mislabeling good customers as high-risk could lose revenue.  
  - **Regulatory Scrutiny**: Proxies must justify alignment with Basel’s default criteria.  

### 3. Simple vs. Complex Model Trade-offs  
| **Model Type**       | **Pros**                          | **Cons**                          |  
|-----------------------|-----------------------------------|-----------------------------------|  
| **Logistic Regression** | Interpretable (WoE bins), audit-friendly | Lower predictive power for complex patterns |  
| **Gradient Boosting**  | Higher accuracy (e.g., AUC)       | "Black-box" nature may hinder regulatory approval |  #Credit Scoring Business Understanding

