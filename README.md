# 🧠 Depression Severity Prediction — Multimodal ML Analysis

> **Binary classification of depression severity (PHQ-9) using passive smartphone sensing, wearable biometrics, and daily mood surveys.**

---

## 📋 Overview

This project applies machine learning to predict depression severity in a cohort of Finnish immigrants using multimodal passive sensing data collected over 28 days. Four classifiers — SVM, Random Forest, XGBoost, and a Neural Network — are trained and evaluated on features extracted from:

- **Oura Ring** — sleep architecture, heart rate, HRV, activity, readiness scores
- **AWARE Smartphone App** — screen usage, call behavior, battery patterns
- **EMA Daily Surveys** — loneliness, social connectedness, isolation, positive/negative affect
- **PHQ-9 Weekly Questionnaire** — target label (binarized at clinical threshold of 10)

---

## 📁 Dataset

| Property | Detail |
|---|---|
| Source | Loneliness and Well-being in Finnish Immigrants |
| Participants | 39 (31 with valid PHQ-9 labels) |
| Study Duration | 28 days |
| Total Features | 55 (after imputation, no columns dropped) |
| Label | Binary: Low Depression (PHQ-9 < 10) vs. High Depression (PHQ-9 ≥ 10) |
| Train / Test Split | 23 train / 8 test (stratified, 75/25) |

### PHQ-9 Score Distribution & Class Balance

![PHQ-9 Distribution](phq9_distribution.png)

> The dataset is roughly balanced after binarization: 15 Low and 16 High depression cases among the 31 labeled participants.

---

## ⚙️ Pipeline

```
Raw Data (per participant)
    │
    ├── Oura Ring CSVs      → aggregate mean/std per feature over 28 days (36 features)
    ├── AWARE App CSVs      → screen, call, battery event aggregates (9 features)
    ├── EMA Survey CSVs     → mood mean/std + response count (11 features)
    └── PHQ-9 Survey CSVs  → sum q1–q9, average across weeks → binary label
                │
                ▼
    Merge on participant_id
    Drop rows with missing labels (n=31 retained)
    Drop features >50% missing (none dropped)
    Median imputation for remaining NaNs
    StandardScaler (SVM, MLP) | Raw (RF, XGBoost)
                │
                ▼
    Train/Test Split (stratified, 75/25)
                │
                ▼
    SVM | Random Forest | XGBoost | MLP
                │
                ▼
    Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, 5-Fold CV
```

---

## 📊 Results

### Model Performance Comparison

![Model Comparison](model_comparison.png)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.375 | 0.333 | 0.250 | 0.286 | 0.406 |
| **Random Forest** | **0.750** | **0.750** | **0.750** | **0.750** ⭐ | 0.656 |
| XGBoost | 0.625 | 0.667 | 0.500 | 0.571 | **0.750** ⭐ |
| Neural Network | 0.500 | 0.500 | 1.000 | 0.667 | 0.594 |

> ⭐ **Random Forest** leads on F1 and accuracy. **XGBoost** leads on ROC-AUC (probability calibration). The Neural Network collapsed to predicting all samples as positive (perfect recall, zero specificity). SVM underperformed on this test partition but recovered in cross-validation.

### Confusion Matrices

![Confusion Matrices](confusion_matrices.png)

### ROC Curves

![ROC Curves](roc_curves.png)

---

## 🔁 Cross-Validation

5-fold stratified cross-validation over all 31 labeled samples:

![CV Comparison](cv_comparison.png)

| Model | CV F1 (Mean ± Std) |
|---|---|
| SVM | 0.664 ± 0.203 |
| Random Forest | 0.711 ± 0.203 |
| XGBoost | 0.704 ± 0.188 |
| Neural Network | 0.712 ± 0.159 |

> All tree-based models and the neural network cluster tightly around F1 ≈ 0.70. High standard deviations across all models reflect instability from the very small sample size (n=31). SVM's CV performance (0.664) is notably higher than its single test-set result (0.286), confirming the test partition was an unlucky draw rather than a globally poor fit.

---

## 🔍 Feature Importance

### Top Features (Random Forest & XGBoost)

![Feature Importance](feature_importance.png)

**Key findings:**
- EMA daily mood features dominate both models — `ema_positive_mean`, `ema_lonely_mean`, `ema_isolate_mean`, `ema_connect_mean`, and `ema_negative_mean` account for ~40% of total RF importance
- Oura physiological features contribute from rank 6 onwards — notably `oura_sleep_hr_lowest_mean`, `oura_sleep_hr_avg_mean`, and `oura_sleep_rem_mean`
- AWARE smartphone behavioral features rank lowest across both models in this population

### Feature Correlation Heatmap

![Feature Correlation](feature_correlation.png)

> Strong intra-modality correlations are visible among Oura sleep sub-metrics (duration, total, deep, REM, light are highly correlated), and among EMA mood features (loneliness and isolation are positively correlated; connectedness is negatively correlated with both). Cross-modality correlations are generally weak, confirming that each sensing stream contributes partially independent information.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

**Python 3.8+** | Tested with scikit-learn 1.3, XGBoost 1.7

---

## 🚀 Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/depression-ml-analysis.git
cd depression-ml-analysis

# Set your dataset path in the notebook
DATASET_PATH = "/path/to/Loneliness_Dataset_Nov10"

# Run the notebook
jupyter notebook project4_depression_ml_final.ipynb
```

---

## ⚠️ Limitations

- **Very small sample**: n=31 labeled participants; each misclassification shifts accuracy by 12.5 percentage points
- **High dimensionality**: 55 features with 31 samples creates serious overfitting risk for all models
- **EMA circularity**: Self-reported mood features conceptually overlap with PHQ-9 items — this inflates apparent predictive power
- **Population specificity**: Finnish immigrants are a culturally specific cohort; generalizability is limited
- **Single test partition**: Test metrics are unreliable at n=8; cross-validation is a more trustworthy performance estimate

---

## 📄 License

This project is for academic/research purposes. Dataset is subject to original study data sharing agreements.

---

## 📬 Citation

If you use this pipeline, please cite the original dataset:

```
Loneliness and Well-being in Finnish Immigrants Dataset
Collected via Oura Ring, AWARE Framework, and EMA protocols
Study duration: 28 days | N=39 participants
```
