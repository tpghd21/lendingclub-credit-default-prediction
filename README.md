# LendingClub Loan Default Prediction

Credit default prediction using Tree-based and Deep Learning models with robustness analysis under economic regime shifts.

---

## Project Overview

This project analyzes credit default prediction performance using LendingClub's peer-to-peer lending data (2007-2018). We compare four state-of-the-art models under two distinct economic conditions:
- **Normal Period**: Stable economic environment  
- **Crisis Period**: 2008 Financial Crisis

**Key Research Question**: How do Tree-based and Deep Learning models perform when faced with distributional shifts in financial data?

---

## Key Findings

### Model Performance (AUC)

| Model | Normal Period | Crisis Period | Performance Drop |
|-------|---------------|---------------|------------------|
| **LightGBM** | 0.72 | 0.65 | **-10%**  |
| **XGBoost** | 0.72 | 0.65 | **-10%**  |
| **MLP + Embedding** | 0.72 | 0.58 | **-20%** |
| **DeepTable (GANDALF)** | 0.72 | 0.57 | **-21%** |

### Critical Insight
**Tree-based models (LightGBM, XGBoost) demonstrate superior robustness** against distributional shifts compared to Deep Learning models, maintaining relatively stable performance even during economic crises.

---

## 🔬 Technical Approach

### Models Implemented
1. **XGBoost** - Gradient Boosting with tree-based learners
2. **LightGBM** - Efficient gradient boosting framework
3. **MLP + Embedding** - Neural network with categorical embeddings
4. **DeepTable (GANDALF)** - Gated Adaptive Network for Deep Automated Learning of Features

### Hyperparameter Optimization
- Framework: **Optuna** with Bayesian optimization
- Objective: **AUC (Area Under ROC Curve)**
- Adaptive threshold optimization for F1 Score maximization

### Evaluation Metrics
- **AUC**: Primary metric for ranking ability
- **PSI** (Population Stability Index): Distribution shift detection
- **KS Statistic**: Discriminatory power
- **F1 Score, Recall, Weighted Accuracy**: Classification performance

### Data Leakage Prevention
Critical preprocessing to remove post-loan variables:
- Excluded: `recoveries`, `total_rec_prncp`, `total_pymnt`, `last_fico_range_*`
- See `Appendix_Data_Leakage_Analysis.ipynb` for detailed analysis

---

## Repository Structure

```
├── Final_project_Main.ipynb              
│   ├── 1. Introduction & Motivation
│   ├── 2. Exploratory Data Analysis (EDA)
│   ├── 3. Data Preprocessing
│   ├── 4. Model Training & Hyperparameter Tuning
│   │   ├── XGBoost
│   │   ├── LightGBM
│   │   ├── MLP + Embedding
│   │   └── DeepTable (GANDALF)
│   └── 5. Results & Discussion
│
├── Appendix_Data_Leakage_Analysis.ipynb  
│   ├── Impact analysis of post-loan variables
│   ├── Performance comparison with/without leakage
│   └── Variable selection guidelines
│
└── README.md                             
```

---

## Tech Stack

**Machine Learning**
- `XGBoost` - Extreme Gradient Boosting
- `LightGBM` - Light Gradient Boosting Machine
- `PyTorch` - Deep Learning framework
- `pytorch-tabular` - Tabular data deep learning

**Optimization & Evaluation**
- `Optuna` - Hyperparameter optimization
- `scikit-learn` - Metrics and preprocessing

**Data Processing & Visualization**
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

---

### Expected Runtime
- Full pipeline (4 models + Optuna tuning): ~2-3 hours on CPU
---

## Detailed Results

### Robustness Analysis
**Defense Rate** = Crisis Performance / Normal Performance

| Metric | LightGBM | XGBoost | MLP | DeepTable |
|--------|----------|---------|-----|-----------|
| AUC Defense | **90%** | **90%** | 80% | 79% |
| F1 Defense | 88% | 86% | 72% | 70% |

### Why Tree-based Models Excel
1. **Recursive splitting** captures nonlinear interactions naturally
2. **Less sensitive to distribution shifts** in feature space
3. **Robust to outliers** during economic turbulence
4. **Efficient handling** of high-dimensional tabular data

### Deep Learning Challenges
- Susceptible to **Out-of-Distribution (OOD)** problems
- Requires **larger datasets** for robust generalization
- **Over-parameterization** can lead to overfitting on specific distributions

---

## Data Preprocessing Highlights

### Critical Steps
1. **Missing Value Handling**
   - Features with >50% missing values removed
   - Strategic imputation for remaining features

2. **Data Leakage Prevention**
   - Systematic removal of post-loan variables
   - Validation through Appendix analysis

3. **Train-Test Split**
   - Normal Period: Random stratified split (80/20)
   - Crisis Period: Temporal split (2008 data)

4. **Feature Engineering**
   - Categorical encoding (Ordinal/Label Encoding)
   - Numerical scaling (StandardScaler)
   - Maintained categorical features for tree-based models

---

## 📜 License

This project is available under the MIT License. See `LICENSE` for details.

---

