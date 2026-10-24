import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.20);
        padding: 16px;
        border-radius: 12px;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(30,90,160,.18), rgba(20,160,130,.12));
        border: 1px solid rgba(120,120,120,.22);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🏦 Bank Customer Churn Intelligence</h1>'
    '<p>Customer analytics · Churn-risk exploration · Model evaluation · '
    'Interactive risk estimation</p></div>',
    unsafe_allow_html=True,
)

# The app accepts the corrected workbook or the raw cleaned dataset.
DATA_CANDIDATES = [
    Path("Bank_Customer_Churn_Final_Analysis_CORRECTED.xlsx"),
    Path("European_Bank_Cleaned.xlsx"),
    Path("European_Bank_Cleaned.xlsx.xlsx"),
]
DATA_FILE = next((p for p in DATA_CANDIDATES if p.exists()), None)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Executive Dashboard",
        "Customer Risk Explorer",
        "Predict Customer Churn",
        "Model Performance",
        "Feature Importance",
        "About & Downloads",
    ],
)
st.sidebar.caption("Educational analytics project. Predictions are estimates, not banking decisions.")

@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(path_string):
    return pd.read_excel(path_string, sheet_name=None)

@st.cache_resource(show_spinner="Preparing prediction model…")
def train_prediction_model(path_string):
    """Train a reproducible demo model from a workbook that contains customer-level target data."""
    sheets = pd.read_excel(path_string, sheet_name=None)
    data = None
    for sheet_name in ["Model_Data", "Full_Customer_Dataset", "Original_Model_Data_Source"]:
        candidate = sheets.get(sheet_name)
        if candidate is not None and not candidate.empty:
            data = candidate.copy()
            break
    if data is None:
        # Try a single-sheet raw workbook.
        data = next((v.copy() for v in sheets.values() if not v.empty), None)
    if data is None:
        raise ValueError("No non-empty worksheet was found.")

    # Identify target column by common names.
    target = next(
        (c for c in data.columns if str(c).strip().lower() in
         {"exited", "churn", "churn indicator", "exited churn indicator (target)", "churned"}),
        None,
    )
    if target is None:
        raise ValueError(
            "Could not identify the churn target column. Expected a column named "
            "'Exited' or 'Churn' in a customer-level data sheet."
        )

    y = pd.to_numeric(data[target], errors="coerce")
    valid = y.isin([0, 1])
    data = data.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if y.nunique() < 2 or len(data) < 20:
        raise ValueError("The dataset needs at least 20 rows and both churn classes (0 and 1).")

    X = data.drop(columns=[target])
    # Remove obvious identifiers and columns with a single value.
    id_like = [c for c in X.columns if str(c).strip().lower() in
               {"rownumber", "row_number", "customerid", "customer_id", "surname"}]
    X = X.drop(columns=id_like, errors="ignore")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        raise ValueError("No usable predictor columns remain after removing identifiers.")

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "categorical",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
            cat_cols,
        ))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    # Refit the same pipeline on all eligible rows for the interactive form.
    model.fit(X, y)
    return model, X, y, metrics, cat_cols, num_cols

def find_target_column(df):
    return next(
        (c for c in df.columns if str(c).strip().lower() in
         {"exited", "churn", "churn indicator", "exited churn indicator (target)", "churned"}),
        None,
    )

def numeric_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

if DATA_FILE is None:
    st.error(
        "Dataset workbook not found. Put the corrected analysis workbook "
        "'Bank_Customer_Churn_Final_Analysis_CORRECTED.xlsx' beside app.py, "
        "or include 'European_Bank_Cleaned.xlsx'."
    )
    st.stop()

try:
    sheets = load_workbook(str(DATA_FILE))
except Exception as exc:
    st.error(f"Unable to read the workbook: {exc}")
    st.stop()

def sheet(name):
    return sheets.get(name, pd.DataFrame())

# ----- Dashboard -----
if page == "Executive Dashboard":
    st.subheader("Executive Dashboard")
    data = sheet("Full_Customer_Dataset")
    if data.empty:
        data = sheet("Model_Data")
    if data.empty:
        data = next((x for x in sheets.values() if not x.empty), pd.DataFrame())

    target = find_target_column(data)
    total = len(data)
    churned = int(pd.to_numeric(data[target], errors="coerce").fillna(0).sum()) if target else 0
    churn_rate = (churned / total * 100) if total and target else 0.0

    a, b, c, d = st.columns(4)
    a.metric("Customer records", f"{total:,}")
    b.metric("Churned customers", f"{churned:,}" if target else "N/A")
    c.metric("Observed churn rate", f"{churn_rate:.2f}%" if target else "N/A")
    d.metric("Workbook sheets", str(len(sheets)))

    if target and total:
        plot = data.copy()
        plot["_churn"] = pd.to_numeric(plot[target], errors="coerce")
        plot = plot[plot["_churn"].isin([0, 1])]
        plot["Churn status"] = plot["_churn"].map({0: "Retained", 1: "Churned"})
        left, right = st.columns(2)
        with left:
            counts = plot["Churn status"].value_counts().rename_axis("Status").reset_index(name="Customers")
            st.plotly_chart(px.pie(counts, names="Status", values="Customers",
                                   hole=.48, title="Churn distribution"), use_container_width=True)
        with right:
            geo = numeric_column(data, ["Geography", "Country", "Region"])
            if geo:
                geo_data = plot.copy()
                geo_data[geo] = data.loc[plot.index, geo].astype(str)
                geo_counts = geo_data.groupby([geo, "Churn status"]).size().reset_index(name="Customers")
                st.plotly_chart(px.bar(geo_counts, x=geo, y="Customers", color="Churn status",
                                       barmode="group", title=f"Churn by {geo}"),
                                use_container_width=True)
            else:
                st.info("Geography column is not available in the selected customer sheet.")
    else:
        st.warning("A customer-level churn target column was not found in the selected sheet.")

    st.subheader("Workbook overview")
    st.dataframe(pd.DataFrame({
        "Worksheet": list(sheets.keys()),
        "Rows": [len(v) for v in sheets.values()],
        "Columns": [len(v.columns) for v in sheets.values()],
    }), use_container_width=True, hide_index=True)

# ----- Risk explorer -----
elif page == "Customer Risk Explorer":
    st.subheader("Customer Risk Explorer")
    risk = sheet("All_Customer_Risk_Scores")
    risk_source = "All_Customer_Risk_Scores"
    if risk.empty:
        risk = sheet("Customer_Risk_Scores_2000")
        risk_source = "Customer_Risk_Scores_2000"
    if risk.empty:
        st.warning("No risk-score worksheet was found in this workbook.")
        st.stop()

    st.caption(f"Source worksheet: {risk_source}. Showing a filtered view of the available scores.")
    score_col = next((c for c in risk.columns if any(k in str(c).lower() for k in ["risk score", "probability", "churn probability"])), None)
    category_col = next((c for c in risk.columns if "risk" in str(c).lower() and any(k in str(c).lower() for k in ["category", "level", "band"])), None)

    filtered = risk.copy()
    if category_col:
        categories = sorted(filtered[category_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect("Risk category", categories, default=categories)
        filtered = filtered[filtered[category_col].astype(str).isin(selected)]
    if score_col:
        values = pd.to_numeric(filtered[score_col], errors="coerce")
        min_value = float(values.min()) if values.notna().any() else 0.0
        max_value = float(values.max()) if values.notna().any() else 1.0
        if min_value < max_value:
            lo, hi = st.slider("Risk score range", min_value, max_value, (min_value, max_value))
            filtered = filtered[pd.to_numeric(filtered[score_col], errors="coerce").between(lo, hi)]

    st.metric("Records in current view", f"{len(filtered):,}")
    st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered risk scores (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_customer_risk_scores.csv",
        "text/csv",
    )

# ----- Interactive prediction -----
elif page == "Predict Customer Churn":
    st.subheader("Predict Customer Churn")
    st.write("Enter customer characteristics to estimate churn probability using a Random Forest model.")
    st.info("This is an educational estimate. Do not use it as the sole basis for real banking decisions.")

    try:
        model, model_X, model_y, holdout_metrics, cat_cols, num_cols = train_prediction_model(str(DATA_FILE))
        input_values = {}
        with st.form("prediction_form"):
            st.markdown("#### Customer details")
            cols = st.columns(2)
            for idx, column in enumerate(model_X.columns):
                with cols[idx % 2]:
                    series = model_X[column]
                    if column in cat_cols:
                        options = sorted(series.dropna().astype(str).unique().tolist())
                        if not options:
                            options = ["Unknown"]
                        input_values[column] = st.selectbox(str(column), options, key=f"input_{column}")
                    else:
                        numeric = pd.to_numeric(series, errors="coerce")
                        median = float(numeric.median()) if numeric.notna().any() else 0.0
                        min_v = float(numeric.min()) if numeric.notna().any() else median - 1
                        max_v = float(numeric.max()) if numeric.notna().any() else median + 1
                        if min_v == max_v:
                            min_v, max_v = median - 1, median + 1
                        input_values[column] = st.number_input(
                            str(column), min_value=float(min_v), max_value=float(max_v),
                            value=float(min(max(median, min_v), max_v)), key=f"input_{column}"
                        )
            submitted = st.form_submit_button("Estimate churn risk", type="primary", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([input_values], columns=model_X.columns)
            probability = float(model.predict_proba(input_df)[0, 1])
            if probability >= 0.70:
                band, guidance = "High", "Consider a suitable customer-retention review; verify the customer's context first."
            elif probability >= 0.35:
                band, guidance = "Medium", "Review relevant customer signals and consider appropriate service follow-up."
            else:
                band, guidance = "Low", "Continue routine customer service and monitoring."
            x1, x2 = st.columns(2)
            x1.metric("Estimated churn probability", f"{probability:.1%}")
            x2.metric("Illustrative risk band", band)
            st.progress(probability)
            st.write(guidance)
            st.caption("Risk-band cut-offs are illustrative and are not validated operational thresholds.")
    except Exception as exc:
        st.error(f"The prediction tool could not prepare a model: {exc}")
        st.write("Please verify that the workbook contains a customer-level sheet with an Exited/Churn target column.")

# ----- Model evaluation -----
elif page == "Model Performance":
    st.subheader("Model Performance Comparison")
    comparison = sheet("Model_Comparison")
    if comparison.empty:
        st.warning("The workbook does not contain a non-empty 'Model_Comparison' worksheet.")
    else:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        model_col = next((c for c in comparison.columns if str(c).strip().lower() == "model"), None)
        numeric_cols = comparison.select_dtypes(include="number").columns.tolist()
        if model_col and numeric_cols:
            metric = st.selectbox("Select metric to compare", numeric_cols)
            st.plotly_chart(px.bar(comparison, x=model_col, y=metric, title=f"Model comparison — {metric}"),
                            use_container_width=True)
    try:
        _, _, _, metrics, _, _ = train_prediction_model(str(DATA_FILE))
        st.subheader("Current app model: holdout evaluation")
        st.caption("These metrics are recalculated from a stratified 80/20 split (random_state=42) on the selected source worksheet.")
        st.dataframe(pd.DataFrame([metrics]).round(4), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"Holdout metrics could not be recalculated: {exc}")

# ----- Feature importance -----
elif page == "Feature Importance":
    st.subheader("Random Forest Feature Importance")
    importance = sheet("Feature_Importance")
    if importance.empty:
        importance = sheet("random forest feature importance")
    if importance.empty:
        st.warning("No feature-importance worksheet was found. Check the exact worksheet name in the workbook.")
    else:
        st.dataframe(importance, use_container_width=True, hide_index=True)
        numeric_cols = importance.select_dtypes(include="number").columns.tolist()
        feature_col = next((c for c in importance.columns if "feature" in str(c).lower()), None)
        score_col = next((c for c in importance.columns if "importance" in str(c).lower()), None)
        if feature_col and score_col:
            chart = importance.sort_values(score_col, ascending=True).tail(20)
            st.plotly_chart(px.bar(chart, x=score_col, y=feature_col, orientation="h",
                                   title="Top feature importances"), use_container_width=True)

# ----- About / downloads -----
elif page == "About & Downloads":
    st.subheader("About the Project")
    st.markdown("""
    **Project:** Predictive Modeling and Risk Scoring for Bank Customer Churn

    **Objective:** Explore churn patterns, compare model performance, and demonstrate
    how customer-level churn risk can be estimated.

    **Tools:** Python, Pandas, scikit-learn, Excel, Streamlit and Plotly.

    **Models covered in the analysis:** Logistic Regression, Decision Tree,
    Random Forest and Gradient Boosting.

    **Responsible use:** The application is an educational prototype. The prediction
    model and risk bands require further validation, calibration, fairness testing,
    privacy review and business approval before any real-world use.
    """)
    st.subheader("Available workbook sheets")
    st.dataframe(pd.DataFrame({
        "Worksheet": list(sheets.keys()),
        "Rows": [len(v) for v in sheets.values()],
        "Columns": [len(v.columns) for v in sheets.values()],
    }), use_container_width=True, hide_index=True)
    st.download_button(
        "Download analysis workbook",
        DATA_FILE.read_bytes(),
        DATA_FILE.name,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    risk = sheet("All_Customer_Risk_Scores")
    if not risk.empty:
        st.download_button("Download all customer risk scores",
                           risk.to_csv(index=False).encode("utf-8"),
                           "all_customer_risk_scores.csv", "text/csv")

st.divider()
st.caption("Bank Customer Churn Prediction & Risk Scoring · Educational analytics prototype")
