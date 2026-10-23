# Customer Churn Model Evaluation & Results

## 1. Executive Summary & Held-Out Test Metrics

The final production model (Tuned Random Forest with stratified class balancing and probability thresholding) was evaluated on the **untouched 20% held-out test set** (1,405 customers: 1,033 retained, 372 churned).

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **75.80%** | Overall fraction of correct classifications across both classes. |
| **Precision** | **52.99%** | When the model flags a customer for churn, it is right 53.0% of the time. |
| **Recall (Sensitivity)** | **76.34%** | Successfully captures 76.3% of all actual churners, preventing revenue leakage. |
| **F1-Score** | **0.6256** | Harmonic mean balancing precision and recall at optimal threshold (0.52). |
| **ROC-AUC** | **0.8427** | High separability ranking true churners above non-churners across all thresholds. |
| **PR-AUC** | **0.6632** | Robust performance on the minority positive class without distortion from class imbalance. |

### Confusion Matrix Breakdown (Test Set)
- **True Negatives (TN)**: 781 retained customers correctly left undisturbed.
- **False Positives (FP)**: 252 retained customers flagged for retention outreach (low-cost retention discount).
- **False Negatives (FN)**: 88 churners missed by the model.
- **True Positives (TP)**: 284 churners accurately intercepted before leaving.

---

## 2. Top 5 Churn Drivers (SHAP Explainability)

Based on global SHAP TreeExplainer values across all test instances, the top 5 factors driving customer churn risk are:

1. **`Contract_Month-to-month`**: High influence on churn probability. Represents tenure trajectory and contractual commitment.
2. **`is_month_to_month`**: Strong predictor indicating contract flexibility vs. multi-year lock-in. Month-to-month contracts lack switching barriers.
3. **`tenure`**: High monthly spend elevates churn sensitivity, especially without perceived service bundling value.
4. **`InternetService_Fiber optic`**: Indicator of internet connection type and tech support integration. Fiber customers without security add-ons churn at higher rates.
5. **`OnlineSecurity_No`**: Automatic payment methods anchor retention, whereas manual payment channels correlate with churn.

---

## 3. What These Metrics Mean for the Business

- **Direct ROI on Retention Campaigns**: Telecom customer acquisition costs ($300-$500) significantly exceed retention incentives ($20-$50 retention offers). By achieving a **76.3% recall**, the business captures **284 out of 372 potential churners** per 1,400 customers, allowing preemptive proactive customer success intervention.
- **Minimizing Marketing Waste**: With **53.0% precision**, retention credits and outreach resources are focused directly on genuine flight-risk customers rather than giving unnecessary discounts to loyal customers.
- **Threshold Agility**: The decision threshold (0.52) can be shifted dynamically: lower thresholds during high-risk quarters to prioritize recall, or higher thresholds when retention budgets are constrained to prioritize precision.

---

## 4. Limitations & Real-World Risks

To maintain rigorous data science ethics, the following limitations must be noted:

1. **Single Cross-Sectional Snapshot**: The IBM dataset is a static slice in time without event timestamp logs. Real-world telecom churn benefits from time-series sequences (e.g., changes in monthly usage, support ticket spikes over time).
2. **Absence of Customer Sentiment & Network Telemetry**: Critical real-world indicators—such as dropped calls, network latency, customer support transcript sentiment, or competitor promotional offers—are not present in this dataset.
3. **Demographic & Proxy Bias**: The dataset includes features like SeniorCitizen and Partner status. Retention models must adhere to fair-lending and non-discriminatory service regulations, ensuring pricing or discount decisions do not penalize protected classes.
