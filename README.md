# OTT Subscription Churn Prediction

**Live Demo:** https://huggingface.co/spaces/Megzzz19/ott-churn-predictor

Predicts which customers are likely to cancel their subscription. Built with Random Forest and deployed as an interactive Streamlit app.

## Results

- **59% recall** - catches most customers who actually churn
- **F1 score: 0.265** - optimized for imbalanced data (22% churn rate)
- **Optimized threshold: 0.445** (instead of default 0.50)
- **Top predictor:** tenure - newer customers are at higher risk (42% feature importance)

## Features

The dashboard has two ways to get predictions:

1. **Sliders** - adjust tenure, seats, MRR, and billing type manually
2. **Chatbot** - type natural language like "tenure 6 monthly billing seats 10 mrr 200"

Gives you risk level (Critical/High/Medium/Low) and suggested action.

## Tech Stack

- Python, pandas, scikit-learn
- Streamlit for the web app
- Random Forest with 5K training samples
- Deployed on Hugging Face Spaces

## How to Run Locally

```bash
git clone https://github.com/Meghaa1901/OTT-churn-analysis.git
cd OTT-churn-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit dashboard
├── notebooks/          # EDA and model training
├── models/             # Saved models (.pkl files)
├── data/               # Raw CSV files
└── requirements.txt
```

## Dataset

Rivalytics OTT subscription data with account and subscription tables. Merged on account_id, engineered tenure feature from dates, one-hot encoded categoricals.

---

Made by Meghaa Arun | [GitHub](https://github.com/Meghaa1901) | 
