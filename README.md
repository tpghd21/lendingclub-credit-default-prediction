# Credit Risk Optimization and Robustness Analysis: A Stress-Test Approach

## Executive Summary
This project develops a credit default prediction model using LendingClub data (2007–2018), with a specific focus on **robustness against market regime changes** (specifically the 2008 Global Financial Crisis) and **data integrity verification**.

Unlike standard machine learning projects that prioritize raw accuracy, this research prioritizes **financial validity** in real-world credit underwriting. It rigorously identifies and eliminates "look-ahead bias" (data leakage) to ensure the model's performance metrics reflect realistic lending scenarios. The study compares Gradient Boosting (XGBoost/LightGBM) and Deep Learning (MLP) architectures to optimize risk-adjusted returns on unsecured personal loan portfolios.

---

## Key Research Highlights

### 1. Rigorous Data Leakage Analysis (Look-Ahead Bias Removal)
A critical component of this research was the identification of features that carried future information not available at the time of loan origination.
* **Problem:** The initial investigation revealed that features such as `last_fico_range_high` and `last_fico_range_low` (updated strictly after loan issuance) were causing artificial accuracy inflation (AUC ≈ 1.0).
* **Resolution:** Conducted a comprehensive feature audit to remove variables containing ex-post information.
* **Result:** While the nominal accuracy dropped, the resulting model provides a **statistically valid probability of default (PD)** that can be safely deployed.
* *Refer to `Appendix_Data_Leakage_Analysis.ipynb` for the detailed investigation.*

### 2. Stress Testing against the 2008 Financial Crisis
To verify the model's generalization capabilities under economic stress, I performed a temporal split validation focusing on the "Crisis Regime."
* **Methodology:** The model was evaluated specifically against the 2007–2010 timeline to measure performance degradation under high volatility.
* **Outcome:** Analyzed the **'F1 Score Defense Rate'** and **'Weighted Accuracy Drop'**. The XGBoost model demonstrated superior resilience compared to MLP, maintaining stability even when borrower default rates spiked, proving its suitability for risk management during economic downturns.

### 3. Addressing Class Imbalance via Cost-Sensitive Learning
Instead of generating synthetic data (e.g., SMOTE) which may introduce noise or artifacts in financial datasets, I implemented **Cost-Sensitive Learning** using the `scale_pos_weight` hyperparameter in XGBoost.
* **Methodology:** Dynamically tuned the penalty weight for misclassifying the minority class (defaults) during the Bayesian Optimization process (Optuna).
* **Advantage:** This approach preserves the original distribution of the financial data while forcing the model to prioritize recall on high-risk loans, effectively optimizing the risk-adjusted return.

---

## Repository Structure

* **`Final_project_Main.ipynb`**:
    The core research notebook containing the end-to-end pipeline: data preprocessing, Bayesian Optimization for hyperparameters, model training (XGBoost, LightGBM, MLP), and robust performance evaluation (Stress Testing).

* **`Appendix_Data_Leakage_Analysis.ipynb`**:
    **[Critical]** A specialized report detailing the detection of data leakage. It serves as a validation layer to prove the model's integrity and explains the statistical reasons for excluding post-origination features.

---

## Methodology & Mathematical Context

### Data Preprocessing & Feature Engineering
* **Imputation:** Handled missing values based on the distribution type of each variable.
* **Encoding:** Applied Target Encoding for high-cardinality categorical variables to preserve information density without exploding the feature space.
* **Scaling:** Utilized Standardization ($z = \frac{x - \mu}{\sigma}$) to ensure proper convergence for the MLP model.

### Model Selection Rationale
* **Gradient Boosting (XGBoost/LightGBM):** Selected for its ability to handle non-linear interactions between features and its robustness to outliers common in financial tabular data.
* **Multi-Layer Perceptron (MLP):** Implemented as a baseline to investigate whether deep representations could capture latent factors missed by tree-based models.
* **Conclusion:** Tree-based models generally outperformed Neural Networks in this domain, offering better interpretability and stability across different market regimes.

---

## Performance Evaluation
* **Metric Selection:** Focused on **Recall (Sensitivity)** and **F1-Score** rather than simple Accuracy. In credit risk modeling, a False Negative (classifying a defaulter as safe) incurs a direct capital loss, necessitating a high-recall approach.
* **Results:** The final model achieves an optimized balance between risk detection and approval rates, validated against Out-of-Time (OOT) datasets covering the financial crisis period.
