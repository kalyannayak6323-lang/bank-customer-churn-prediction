# Bank Customer Churn Intelligence

A Streamlit web application for educational analysis of bank customer churn, risk-score exploration, model comparison, and interactive churn-risk estimation.

## Features
- Executive dashboard with churn KPIs and charts
- Customer risk-score explorer with CSV export
- Interactive customer churn prediction form using a Random Forest pipeline
- Model performance comparison
- Feature-importance view when available in the workbook
- Workbook and risk-score downloads

## Run locally
1. Place the corrected workbook `Bank_Customer_Churn_Final_Analysis_CORRECTED.xlsx` in the same folder as `app.py`. The app also checks for `European_Bank_Cleaned.xlsx` and `European_Bank_Cleaned.xlsx.xlsx`.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

## Deploy on Streamlit Community Cloud
Push `app.py`, `requirements.txt`, and the workbook to the root of a GitHub repository, then create an app on Streamlit Community Cloud using `app.py` as the main file.

## Notes
- The prediction form learns from a customer-level worksheet containing a binary `Exited` or `Churn` target.
- It uses an 80/20 stratified split for holdout metrics and refits on all eligible records for interactive estimates.
- Predictions and example risk bands are educational estimates, not financial advice or a production decision system.
- Before publishing, confirm the workbook contains no confidential or personally identifiable customer data.
