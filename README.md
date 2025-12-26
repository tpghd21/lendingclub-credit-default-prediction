# Credit Risk Optimization and Robustness Analysis: A Stress-Test Approach

## Executive Summary
This project develops a credit default prediction model using LendingClub data, with a specific focus on **robustness against market regime changes** (specifically the 2008 Global Financial Crisis) and **data integrity verification**.

Unlike standard machine learning projects that prioritize raw accuracy, this research prioritizes **financial validity**. It rigorously identifies and eliminates "look-ahead bias" (data leakage) to ensure the model's performance metrics reflect realistic trading environments. The study compares Gradient Boosting (XGBoost/LightGBM) and Deep Learning (MLP) architectures to optimize risk-adjusted returns on unsecured personal loan portfolios.

---

## Key Research Highlights

### 1. rigorous Data Leakage Analysis (Look-Ahead Bias Removal)
A critical component of this research was the identification of features that carried future information not available at the time of loan origination.
* **Problem:** The initial model achieved an artificially high accuracy (>99%) due to the inclusion of `total_rec_prncp` (Total payment received to date).
* **Resolution:** Conducted a comprehensive feature audit to remove variables containing ex-post information.
* **Result:** While the nominal accuracy dropped to a realistic level, the resulting model provides a **statistically valid probability of default (PD)** that can be safely deployed in production environments.
* *See `Appendix_Data_Leakage_Analysis.ipynb` for the detailed investigation.*

### 2. Stress Testing against the 2008 Financial Crisis
To verify the model's generalization capabilities under economic stress, I performed a temporal split validation.
* **methodology:** The model was trained on pre-crisis and post-crisis data but validated specifically against the 2007-2010 timeline.
* **Outcome:** The XGBoost model demonstrated superior resilience compared to MLP, maintaining a stable AUC score even during high-volatility periods, suggesting better suitability for tabular financial data with non-linear decision boundaries.

### 3. Handling High-Dimensional Imbalance via SMOTE
Given the inherent imbalance in default datasets (Defaults < 20%), I utilized Synthetic Minority Over-sampling Technique (SMOTE).
* **Mathematical Rationale:** Instead of simple oversampling, SMOTE interpolates between minority samples in the feature space, effectively approximating the underlying manifold of the default class and preventing overfitting to specific noise points.

---

## Repository Structure

* **`Final_project_Main.ipynb`**:
    The core research notebook containing the end-to-end pipeline: data preprocessing, SMOTE implementation, model training (XGBoost, LightGBM, MLP), and performance evaluation.

* **`Appendix_Data_Leakage_Analysis.ipynb`**:
    **[Critical]** A specialized report detailing the detection of data leakage. It serves as a validation layer to prove the model's integrity and explains the statistical reasons for feature exclusion.

---

## Methodology & Mathematical Context

### Data Preprocessing & Feature Engineering
* **Imputation:** Handled missing values based on the distribution type of each variable.
* **Encoding:** Applied Target Encoding for high-cardinality categorical variables to preserve information density without exploding the feature space.
* **Scaling:** utilized Standardization ($z = \frac{x - \mu}{\sigma}$) to ensure faster convergence for the MLP model.

### Model Selection Rationale
* **Gradient Boosting (XGBoost/LightGBM):** Selected for its ability to handle non-linear interactions between features and its robustness to outliers in tabular data.
* **Multi-Layer Perceptron (MLP):** Implemented as a baseline to compare whether a deep representation could capture latent factors missed by tree-based models.
* **Conclusion:** Tree-based models outperformed Neural Networks in this specific domain, likely due to the discrete nature of financial credit data and the distinct decision boundaries required for risk classification.

---

## Performance Evaluation
* **Metric Selection:** Focused on **Recall (Sensitivity)** and **AUC-ROC** rather than simple Accuracy. In credit risk modeling, a False Negative (classifying a defaulter as safe) incurs a direct capital loss, whereas a False Positive represents only an opportunity cost.
* **Results:** The final XGBoost model achieved an optimized balance between risk detection and approval rates, validated against the out-of-time (OOT) dataset covering the financial crisis.

## Future Work
* **Explainability (XAI):** Implementing SHAP (SHapley Additive exPlanations) values to interpret individual loan rejection reasons for regulatory compliance.
* **Macro-Economic Integration:** Incorporating external signals (e.g., Fed Funds Rate, Unemployment Rate) to further improve the model's predictive power during regime shifts.
