# Customer Churn Risk Dashboard

A Random Forest-based dashboard that predicts the probability a telecom customer will churn, explains why, and recommends a retention action.

Live app: https://5feb8gadhvotvqfvdxayz4.streamlit.app/

## Features

- Predicts churn probability for a single customer from account and service details
- Classifies risk into Low, Medium, or High bands based on a tuned probability threshold
- Recommends a specific retention action based on contract type, tenure, and risk band
- Supports batch scoring: upload a CSV of customers and get risk scores for all of them at once
- Shows model explainability via a top-10 feature importance chart
- Visualizes batch results with a risk distribution pie chart and a churn probability histogram
- Lets you download scored results as a CSV

## How it works

1. Enter a customer's details, or upload a CSV for batch mode
2. The Random Forest model outputs a churn probability
3. The probability is mapped to a risk band, using a threshold tuned for high recall on churners
4. A recommended action is generated based on contract type, tenure, and risk band

## Model performance

- ROC-AUC: 0.845
- - Recall: 87% at the chosen decision threshold (0.35)

## Tech stack

Python, Streamlit, scikit-learn, Pandas, Plotly, joblib

## Local setup

```
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Trained on the public Telco Customer Churn dataset, so it may not generalize well to other markets or products
- The decision threshold (0.35) favors catching more churners at the cost of some false positives, a deliberate tradeoff for a retention use case
- Batch mode expects the same columns as the training dataset; unexpected columns are ignored rather than validated
