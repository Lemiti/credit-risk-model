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

###How does the Base II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented mode?

###Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

###What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated finance context?
