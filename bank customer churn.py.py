
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load customer data
folder = Path(__file__).parent
excel_files = list(folder.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError("Excel workbook not found.")

file_path = excel_files[0]

df = pd.read_excel(
    file_path,
    sheet_name="European_Bank"
)

df.columns = df.columns.astype(str).str.strip()

print("Workbook:", file_path.name)
print("Dataset shape:", df.shape)

# 2. Select model features using actual Excel column names
features = [
    "Customer credit score",
    "Geography France",
    "Geography Germany",
    "Geography_Spain",
    "Gender Male",
    "Customer age",
    "Years with the bankTenure",
    "Account balance",
    "Number of bank products",
    "Credit card ownership",
    "Activity indicat or Member",
    "Estimated annual salary",
    "Balance-to-Salary Ratio",
    "Products per Tenure",
    "Age-Tenure Interaction"
]

target = "Exited Churn indicator (target)"

# 3. Verify required columns
missing = [
    col for col in features + [target]
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 4. Prepare model data
X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")

valid_rows = y.isin([0, 1])
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

# 5. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Build and train Logistic Regression
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 7. Make predictions on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluate the model
print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Precision:",
      round(precision_score(
          y_test, y_pred, zero_division=0
      ), 4))

print("Recall:",
      round(recall_score(
          y_test, y_pred, zero_division=0
      ), 4))

print("F1-score:",
      round(f1_score(
          y_test, y_pred, zero_division=0
      ), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("\nLogistic Regression training completed.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Compare additional models on the same train/test split
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

results = [{
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}]

for name, candidate in models.items():
    candidate.fit(X_train, y_train)

    predictions = candidate.predict(X_test)
    probabilities = candidate.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

comparison = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    folder / "model_comparison.csv",
    index=False
)

print("\nSaved results to model_comparison.csv")

# STEP: Random Forest Feature Importance

import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": models["Random Forest"].feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
print(feature_importance.to_string(index=False))

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully.")

import matplotlib.pyplot as plt

# Load the saved feature importance results
importance_df = pd.read_csv("random_forest_feature_importance.csv")

# Select the top 10 features
top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Features Predicting Bank Customer Churn")
plt.tight_layout()

# Save chart
plt.savefig("random_forest_feature_importance.png", dpi=300)
plt.show()

print("Feature importance chart saved successfully.")

# STEP: CUSTOMER CHURN RISK SCORING

# Predict probability of churn for every test customer
churn_probability = models["Random Forest"].predict_proba(X_test)[:, 1]

# Create customer-level risk results
risk_results = pd.DataFrame({
    "Actual Churn": y_test.to_numpy(),
    "Churn Probability": churn_probability
})

# Convert probability to a score from 0 to 100
risk_results["Risk Score"] = (
    risk_results["Churn Probability"] * 100
).round(2)

# Categorize customers by predicted risk
risk_results["Risk Category"] = pd.cut(
    risk_results["Risk Score"],
    bins=[-0.01, 30, 60, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Save results
risk_results.to_csv("customer_churn_risk_scores.csv", index=False)

print("\n===== CUSTOMER CHURN RISK SCORING =====")
print("Customers scored:", len(risk_results))
print("\nRisk category distribution:")
print(risk_results["Risk Category"].value_counts().sort_index())
print("\nSample risk scores:")
print(risk_results.head(10).to_string(index=False))
print("\nRisk scores saved to customer_churn_risk_scores.csv")

# STEP: SCORE ALL 10,000 CUSTOMERS

# Use the same feature columns and order used during model training
all_customer_features = X.copy()

# Predict churn probability for every customer
all_churn_probabilities = models["Random Forest"].predict_proba(
    all_customer_features
)[:, 1]

# Create final customer-level results
all_customer_results = pd.DataFrame({
    "Customer Index": range(1, len(all_customer_features) + 1),
    "Actual Churn": y.to_numpy(),
    "Churn Probability": all_churn_probabilities
})

# Risk score from 0 to 100
all_customer_results["Risk Score"] = (
    all_customer_results["Churn Probability"] * 100
).round(2)

# Risk category
all_customer_results["Risk Category"] = pd.cut(
    all_customer_results["Risk Score"],
    bins=[-0.01, 30, 60, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Save full dataset for Power BI
all_customer_results.to_csv(
    "all_customer_churn_risk_scores.csv",
    index=False
)

print("\n===== ALL CUSTOMER RISK SCORING =====")
print("Total customers scored:", len(all_customer_results))
print("\nRisk category counts:")
print(all_customer_results["Risk Category"].value_counts().sort_index())
print("\nSaved: all_customer_churn_risk_scores.csv")
