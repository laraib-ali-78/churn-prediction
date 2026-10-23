# Exploratory Data Analysis Insights

Key findings from the cleaned IBM Telco Customer Churn dataset (7,021 customers):

- **Baseline Class Imbalance**: Overall churn rate is **26.45%** (1,857 churned vs. 5,164 retained), establishing that 1 in 4 customers churn.
- **Contract Duration is the #1 Retention Anchor**: Customers on **Month-to-month contracts experience a 42.64% churn rate** (1,643 / 3,853), compared to **11.27%** on One-year contracts and only **2.83%** on Two-year contracts.
- **Tenure Risk Window**: Churned customers have a median tenure of only **10.0 months**, versus **38.0 months** for retained customers—indicating attrition clusters in the first year of onboarding.
- **Fiber Optic Attrition**: Customers with **Fiber optic internet service churn at 41.78%** (1,291 / 3,090), compared to **18.89% for DSL** and **7.21% for No Internet**, likely reflecting high price sensitivity and service expectations.
- **Monthly Charges Impact**: Churners pay a significantly higher median monthly bill of **$79.70** compared to retained customers at **$64.50** (a $15.20 or 23.6% premium).
- **Payment Method Vulnerability**: Customers paying via **Electronic check churn at 45.15%** (1,065 / 2,359), while automated payment methods (Bank transfer at **16.71%**, Credit card at **15.24%**) show nearly 3x lower churn.
- **Multi-Service & Bundling Stickiness**: Retained customers have higher overall service adoption (security, backup, tech support), showing strong negative correlation with churn.
