# Loan Status Classification using Machine Learning Models

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict loan approval status. The goal is to develop an accurate predictive model that can classify whether a loan applicant will have a successful loan status (approved) or not based on various personal and financial features. This is a binary classification problem where the target variable is `loan_status` (1 = approved, 0 = rejected).

The primary aim is to:
- Develop and train 6 different machine learning classification models
- Evaluate their performance using multiple evaluation metrics
- Compare the models to identify the best performing algorithm
- Provide insights into model behavior and prediction accuracy

---

## b. Dataset Description

**Dataset Name:** Loan Data Dataset

**Number of Records:** 45,002 loan applications

**Number of Features:** 13 input features + 1 target variable

**Target Variable:** `loan_status` (Binary: 0 = Loan Rejected, 1 = Loan Approved)

### Features Description:

1. **person_age** (Numeric): Age of the loan applicant
2. **person_gender** (Categorical): Gender of the applicant (Male/Female)
3. **person_education** (Categorical): Education level (High School, Bachelor, Master, Associate, etc.)
4. **person_income** (Numeric): Annual income of the applicant
5. **person_emp_exp** (Numeric): Employment experience in years
6. **person_home_ownership** (Categorical): Home ownership status (RENT, MORTGAGE, OWN)
7. **loan_amnt** (Numeric): Loan amount requested
8. **loan_intent** (Categorical): Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
9. **loan_int_rate** (Numeric): Interest rate for the loan
10. **loan_percent_income** (Numeric): Loan amount as percentage of annual income
11. **cb_person_cred_hist_length** (Numeric): Credit history length in years
12. **credit_score** (Numeric): Credit score of the applicant
13. **previous_loan_defaults_on_file** (Categorical): History of loan defaults (Yes/No)

### Data Preprocessing:

- **Missing Values:** Checked and handled appropriately
- **Categorical Encoding:** Label encoding applied to all categorical variables
- **Feature Scaling:** MinMax Scaler applied to normalize all features to [0, 1] range
- **Train-Test Split:** 80-20 split (36,001 training samples, 9,000 test samples)

---

## c. Models Used and Performance Comparison

### Classification Models Implemented:

1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Tree-based classification
3. **K-Nearest Neighbor (kNN)** - Instance-based learning (k=5)
4. **Naive Bayes Classifier (Gaussian)** - Probabilistic classification
5. **Random Forest** - Ensemble method (100 estimators)
6. **XGBoost** - Gradient boosting ensemble method (100 estimators)

### Performance Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | 0.9481 | 0.7650 | 0.7289 | 0.7465 | 0.6762 |
| Decision Tree | 0.9140 | 0.9605 | 0.8527 | 0.7433 | 0.7943 | 0.7430 |
| kNN | 0.8900 | 0.9185 | 0.7849 | 0.6990 | 0.7395 | 0.6719 |
| Naive Bayes | 0.7369 | 0.9360 | 0.4590 | 0.9965 | 0.6285 | 0.5490 |
| Random Forest (Ensemble) | 0.9287 | 0.9736 | 0.8904 | 0.7761 | 0.8293 | 0.7875 |
| XGBoost (Ensemble) | **0.9339** | **0.9777** | **0.8881** | **0.8055** | **0.8448** | **0.8044** |

**Legend:**
- **Accuracy:** Proportion of correct predictions among total predictions
- **AUC (Area Under ROC Curve):** Measure of model's ability to distinguish between classes (0-1, higher is better)
- **Precision:** Proportion of positive predictions that are actually correct
- **Recall:** Proportion of actual positive cases that were correctly identified
- **F1 Score:** Harmonic mean of Precision and Recall (balanced metric)
- **MCC (Matthews Correlation Coefficient):** Balanced measure for binary classification (-1 to 1, higher is better)

---

## d. Model Performance Analysis and Observations

### Performance Observations by Model:

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved 88.94% accuracy with a good AUC score of 0.9481. The model demonstrates balanced performance with 76.50% precision and 72.89% recall. It performs well for a linear model but is outperformed by tree-based and ensemble methods. The relatively lower recall suggests it may miss some loan approvals. Training time was efficient at 0.104 seconds. |
| **Decision Tree** | Decision Tree Classifier improved accuracy to 91.40% with AUC of 0.9605. It shows strong precision (85.27%) and moderate recall (74.33%), resulting in an F1-score of 0.7943. The model captures complex patterns in the data better than Logistic Regression. However, the lower recall compared to Random Forest indicates it may still miss some positive cases. Training time was 0.107 seconds. |
| **kNN** | K-Nearest Neighbor achieved 89.00% accuracy with the lowest AUC score (0.9185) among tree-based and ensemble models. The model has decent precision (78.49%) but the lowest recall (69.90%) among the top performers, resulting in an F1-score of 0.7395. This suggests kNN struggles to identify all positive cases. The relatively simple distance-based approach limits its effectiveness. Training was fast at 0.118 seconds. |
| **Naive Bayes** | Naive Bayes achieved the lowest accuracy (73.69%) but interestingly has the second-highest recall (99.65%), meaning it correctly identifies almost all positive cases but produces many false positives. The very low precision (45.90%) and F1-score (0.6285) indicate poor overall performance. The model is too conservative in its predictions. However, it has the fastest training time (0.026 seconds) and a respectable AUC of 0.9360. This model is not recommended for this dataset. |
| **Random Forest (Ensemble)** | Random Forest achieved 92.87% accuracy with an excellent AUC score of 0.9736, demonstrating strong discriminative power. Precision is high at 89.04% with recall at 77.61%, yielding an F1-score of 0.8293. The MCC of 0.7875 indicates good overall performance. The ensemble approach effectively combines multiple decision trees to reduce overfitting and improve generalization. Training took 0.991 seconds. This is a strong candidate model for deployment. |
| **XGBoost (Ensemble)** | XGBoost outperforms all other models with 93.39% accuracy and the highest AUC score (0.9777). It achieves excellent precision (88.81%) and the highest recall (80.55%) among all models, resulting in the best F1-score (0.8448) and MCC (0.8044). The gradient boosting approach optimally builds upon previous trees' errors, leading to superior predictive performance. Training took 0.437 seconds. **XGBoost is the recommended model for production deployment.** |

### Summary of Key Findings:

1. **Best Overall Model:** XGBoost achieved the highest accuracy (93.39%) and AUC (0.9777), making it the most reliable model for loan status prediction.

2. **Ensemble Methods Performance:** Both Random Forest and XGBoost significantly outperform linear and single tree-based models, demonstrating the advantage of ensemble learning.

3. **Trade-offs Observed:**
   - Naive Bayes prioritizes recall but sacrifices precision, making it unsuitable for this problem
   - kNN shows balanced but weaker performance compared to tree-based models
   - Decision Tree is a reasonable interpretable alternative to ensemble methods

4. **Recommended Model:** XGBoost should be selected for deployment due to:
   - Highest accuracy and AUC scores
   - Best recall (80.55%) - minimizes missed loan approvals
   - Excellent precision (88.81%) - minimizes false approvals
   - Best overall balanced performance (F1 = 0.8448, MCC = 0.8044)

5. **Model Ranking (Best to Worst):**
   - 1st: XGBoost (93.39% accuracy)
   - 2nd: Random Forest (92.87% accuracy)
   - 3rd: Decision Tree (91.40% accuracy)
   - 4th: kNN (89.00% accuracy)
   - 5th: Logistic Regression (88.94% accuracy)
   - 6th: Naive Bayes (73.69% accuracy)

---

## Conclusion

This project successfully implemented and compared 6 machine learning classification models on the loan status prediction dataset. Through comprehensive evaluation using multiple metrics (Accuracy, AUC, Precision, Recall, F1-Score, and MCC), XGBoost emerged as the superior model with the best overall performance. The analysis demonstrates that ensemble methods with gradient boosting significantly outperform simpler linear and single-tree approaches for this binary classification task. The selected XGBoost model is recommended for predicting loan approval status in a production environment, offering the best balance between accuracy and reliability.
