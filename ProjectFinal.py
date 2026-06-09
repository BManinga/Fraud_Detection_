# =========================
# 0) Imports & Setup
# =========================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap

RANDOM_STATE = 42

# Output folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("Setup complete.")

# =========================
# 1) Loading Data
# =========================
# Update this path to your actual data location
DATA_PATH = r"ZimSwitchPOSTransactions.csv" 
data = pd.read_csv(DATA_PATH)

print("Shape:", data.shape)
print(data.head())
print(data.info())

# ================================
# 2) Cleaning & Type Fixes
# ================================
# 1. Standardize column names: strip whitespace and enforce lowercase
data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

# 2. Parse timestamp and create temporal features
if "timestamp" in data.columns:
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.day_name()
    print(f"Timestamp parsed. Range: {data['timestamp'].min()} to {data['timestamp'].max()}")
else:
    print("Warning: 'timestamp' column not found. Skipping temporal features.")

# 3. Ensure fraud label exists and is numeric binary
if "fraud_label" not in data.columns:
    raise ValueError("Target column 'fraud_label' not found. Please confirm target column name.")

data["fraud_label"] = pd.to_numeric(data["fraud_label"], errors="coerce").fillna(0).astype(int)

# Validate binary labels
if not set(data["fraud_label"].unique()).issubset({0, 1}):
    raise ValueError("fraud_label must contain only 0 and 1")

# 4. Log transform transaction amount to handle skewness
if "transaction_amount" in data.columns:
    data["transaction_amount"] = pd.to_numeric(data["transaction_amount"], errors="coerce")
    data["log_amount"] = np.log1p(data["transaction_amount"].clip(lower=0))
    print("Transaction amount log-transformed to 'log_amount'")
else:
    print("Warning: 'transaction_amount' column not found. Skipping log transform.")

# 5. Summary of cleaning results
print("\n=== Data Cleaning Summary ===")
print(f"Final dataset shape: {data.shape}")
print("\nFraud_Label distribution:")
print(data["fraud_label"].value_counts())
print(f"\nFraud rate: {data['fraud_label'].mean():.2%}")
print("=============================\n")

# =====================================
# 3) Dropping Irrelevant Columns
# =====================================
irrelevant_cols = [
    "market_preference", 
    "color", 
    "card_color", 
    "preferred_color",
    "user_id",
    "transaction_id"
]

cols_to_drop = [col for col in irrelevant_cols if col in data.columns]
cols_not_found = [col for col in irrelevant_cols if col not in data.columns]

if cols_to_drop:
    initial_shape = data.shape
    data = data.drop(columns=cols_to_drop)
    print("=== Feature Reduction Summary ===")
    print(f"Columns dropped: {cols_to_drop}")
    print(f"Shape before: {initial_shape} → Shape after: {data.shape}")
    print(f"Features removed: {len(cols_to_drop)}")
else:
    print("No irrelevant columns found to drop.")

if cols_not_found:
    print(f"\nNote: These columns were not in dataset: {cols_not_found}")
    
print("===================================\n")

# =============================================
# 4.1 EDA Group 1: Transaction Amount Patterns
# =============================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Analysis: Transaction Amount Patterns by Fraud Class', 
             fontsize=14, fontweight='bold', y=0.98)

data['fraud_label'] = data['fraud_label'].astype('category')
fraud_palette = ["#2E86AB", "#A23B72"]  

# (1) Overall Distribution of Transaction Amounts
if "transaction_amount" in data.columns:
    sns.histplot(
        data=data, x="transaction_amount", bins=50, kde=True, 
        ax=axes[0, 0], color="#2E86AB", edgecolor="black", alpha=0.7
    )
    axes[0, 0].set_title("(a) Distribution of Transaction Amounts")
    axes[0, 0].set_xlabel("Transaction Amount (USD)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_yscale('log')
    axes[0, 0].text(0.95, 0.95, f'n={len(data):,}', transform=axes[0, 0].transAxes, 
                    ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))

# (2) Transaction Amount by Fraud Label - Boxplot
if "transaction_amount" in data.columns and "fraud_label" in data.columns:
    sns.boxplot(
        x="fraud_label", y="transaction_amount", data=data, 
        ax=axes[0, 1], palette=fraud_palette, showfliers=False
    )
    axes[0, 1].set_title("(b) Transaction Amount by Fraud Class")
    axes[0, 1].set_xlabel("Fraud Label")
    axes[0, 1].set_ylabel("Transaction Amount (USD)")
    axes[0, 1].set_xticklabels(['Non-Fraud', 'Fraud'])

# (3) Log-Transformed Amount KDE by Fraud Label
if "log_amount" in data.columns and "fraud_label" in data.columns:
    sns.kdeplot(
        data=data, x="log_amount", hue="fraud_label", fill=True, 
        ax=axes[1, 0], palette=fraud_palette, alpha=0.6, linewidth=2
    )
    axes[1, 0].set_title("(c) Log-Transformed Amount Density")
    axes[1, 0].set_xlabel("Log(Transaction Amount + 1)")
    axes[1, 0].set_ylabel("Density")
    handles, _ = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles, ['Non-Fraud', 'Fraud'], title="Fraud Label")

# (4) Outlier View - Violin Plot
if "transaction_amount" in data.columns and "fraud_label" in data.columns:
    cap = data["transaction_amount"].quantile(0.99)
    sns.violinplot(
        x="fraud_label", y="transaction_amount", data=data[data["transaction_amount"] <= cap], 
        ax=axes[1, 1], palette=fraud_palette, inner="quartile", cut=0
    )
    axes[1, 1].set_title("(d) Amount Distribution with Quartiles")
    axes[1, 1].set_xlabel("Fraud Label")
    axes[1, 1].set_ylabel("Transaction Amount (USD)")
    axes[1, 1].set_xticklabels(['Non-Fraud', 'Fraud'])
    axes[1, 1].text(0.05, 0.95, f'99th percentile: ${cap:,.0f}', transform=axes[1, 1].transAxes,
                    ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/figures/Fig5_1_EDA_Amount_Patterns.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/figures/Fig5_1_EDA_Amount_Patterns.pdf", bbox_inches='tight')
plt.show()
print("Figure saved: outputs/figures/Fig5_1_EDA_Amount_Patterns.png and .pdf")

# ==================================================
# 4.2 EDA Group 2: Temporal Transaction Patterns
# ==================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Exploratory Analysis: Temporal Patterns in Fraudulent Activity', 
             fontsize=14, fontweight='bold', y=0.98)

data_plot = data.copy()
data_plot['fraud_label'] = data_plot['fraud_label'].astype('category')
fraud_palette = ["#2E86AB", "#A23B72"]

# (1) Transaction Count by Hour of Day
if "hour" in data_plot.columns:
    sns.countplot(
        x="hour", hue="fraud_label", data=data_plot, ax=axes[0, 0], 
        palette=fraud_palette, edgecolor="black", alpha=0.8
    )
    axes[0, 0].set_title("(a) Transaction Volume by Hour of Day")
    axes[0, 0].set_xlabel("Hour of Day (0-23)")
    axes[0, 0].set_ylabel("Transaction Count")
    axes[0, 0].legend(title="Fraud Label", labels=['Non-Fraud', 'Fraud'])
    axes[0, 0].set_xticks(range(0, 24, 2))

# (2) Transactions by Day of the Week
if "day_of_week" in data_plot.columns:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.countplot(
        x="day_of_week", hue="fraud_label", data=data_plot, order=day_order, 
        ax=axes[0, 1], palette=fraud_palette, edgecolor="black", alpha=0.8
    )
    axes[0, 1].set_title("(b) Transaction Volume by Day of Week")
    axes[0, 1].set_xlabel("Day of Week")
    axes[0, 1].set_ylabel("Transaction Count")
    axes[0, 1].legend(title="Fraud Label", labels=['Non-Fraud', 'Fraud'])
    axes[0, 1].tick_params(axis="x", rotation=30)

# (3) Fraud Rate by Hour
if "hour" in data.columns:
    hourly_fraud = data.assign(fraud_label=data['fraud_label'].astype(int))\
                     .groupby("hour")["fraud_label"].mean().reset_index(name="fraud_rate")
    
    sns.lineplot(
        x="hour", y="fraud_rate", data=hourly_fraud, ax=axes[1, 0], 
        marker="o", color="#A23B72", linewidth=2.5, markersize=6
    )
    axes[1, 0].set_title("(c) Fraud Rate by Hour of Day")
    axes[1, 0].set_xlabel("Hour of Day (0-23)")
    axes[1, 0].set_ylabel("Fraud Rate")
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    axes[1, 0].set_ylim(0, hourly_fraud["fraud_rate"].max() * 1.15)
    
    peak_hour = hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmax()]
    axes[1, 0].annotate(f"Peak: {peak_hour['fraud_rate']:.1%}\nat {int(peak_hour['hour'])}:00",
                        xy=(peak_hour['hour'], peak_hour['fraud_rate']),
                        xytext=(peak_hour['hour']+2, peak_hour['fraud_rate']),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# (4) Fraud Rate by Day of Week
if "day_of_week" in data.columns:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_fraud = data.assign(fraud_label=data['fraud_label'].astype(int))\
                    .groupby("day_of_week")["fraud_label"].mean().reindex(day_order).reset_index()
    daily_fraud.columns = ["day_of_week", "fraud_rate"]
    
    sns.lineplot(
        x="day_of_week", y="fraud_rate", data=daily_fraud, ax=axes[1, 1],
        marker="s", color="#A23B72", linewidth=2.5, markersize=8
    )
    axes[1, 1].set_title("(d) Fraud Rate by Day of Week")
    axes[1, 1].set_xlabel("Day of Week")
    axes[1, 1].set_ylabel("Fraud Rate")
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    axes[1, 1].set_ylim(0, daily_fraud["fraud_rate"].max() * 1.15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/figures/Fig5_2_EDA_Temporal_Patterns.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/figures/Fig5_2_EDA_Temporal_Patterns.pdf", bbox_inches='tight')
plt.show()
print("Figure saved: outputs/figures/Fig5_2_EDA_Temporal_Patterns.png and.pdf")

# ==============================================
# 5) Feature Engineering and Preprocessing
# ==============================================
print("\n=== Starting Feature Engineering ===")

# Select features for modeling
# Drop identifier columns and target
X = data.drop(['fraud_label', 'timestamp'], axis=1)
y = data['fraud_label']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ==============================================
# 6) Model Training and Evaluation
# ==============================================
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1-Score: {results[name]['f1']:.4f}")
    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")

# ==============================================
# 7) Handle Imbalanced Data with SMOTE
# ==============================================
print("\n=== Handling Imbalanced Data with SMOTE ===")

# Apply SMOTE to training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training size: {len(X_train)}")
print(f"Resampled training size: {len(X_train_resampled)}")
print(f"Resampled fraud rate: {y_train_resampled.mean():.4f}")

# Train models with SMOTE
smote_results = {}
for name, model in models.items():
    print(f"\n--- Training {name} with SMOTE ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train_resampled, y_train_resampled)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    smote_results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"Accuracy: {smote_results[name]['accuracy']:.4f}")
    print(f"Precision: {smote_results[name]['precision']:.4f}")
    print(f"Recall: {smote_results[name]['recall']:.4f}")
    print(f"F1-Score: {smote_results[name]['f1']:.4f}")
    print(f"ROC-AUC: {smote_results[name]['roc_auc']:.4f}")

# ==============================================
# 8) Results Visualization
# ==============================================
# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics))
width = 0.35

for i, (name, result) in enumerate(results.items()):
    values = [result[m] for m in metrics]
    axes[0].bar(x + i*width, values, width, label=name)

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance (Original Data)')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_ylim(0, 1)

for i, (name, result) in enumerate(smote_results.items()):
    values = [result[m] for m in metrics]
    axes[1].bar(x + i*width, values, width, label=name)

axes[1].set_xlabel('Metrics')
axes[1].set_ylabel('Score')
axes[1].set_title('Model Performance (SMOTE Resampled)')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metrics)
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("outputs/figures/Model_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/figures/Model_Comparison.pdf", bbox_inches='tight')
plt.show()
print("Figure saved: outputs/figures/Model_Comparison.png and .pdf")

# ==============================================
# 9) ROC Curves Comparison
# ==============================================
plt.figure(figsize=(10, 8))

for name, result in smote_results.items():
    pipeline = result['pipeline']
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Models with SMOTE')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/ROC_Curves.png", dpi=300, bbox_inches='tight')
plt.savefig("outputs/figures/ROC_Curves.pdf", bbox_inches='tight')
plt.show()
print("Figure saved: outputs/figures/ROC_Curves.png and .pdf")

# ==============================================
# 10) Best Model Selection and Saving
# ==============================================
# Find best model based on F1-score
best_model_name = max(smote_results, key=lambda x: smote_results[x]['f1'])
best_model = smote_results[best_model_name]['pipeline']

print(f"\n=== Best Model: {best_model_name} ===")
print(f"F1-Score: {smote_results[best_model_name]['f1']:.4f}")
print(f"ROC-AUC: {smote_results[best_model_name]['roc_auc']:.4f}")

# Save the best model
import joblib
joblib.dump(best_model, "outputs/models/best_fraud_detection_model.pkl")
print("Best model saved to: outputs/models/best_fraud_detection_model.pkl")

# ==============================================
# 11) Feature Importance (for tree-based models)
# ==============================================
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature names after preprocessing
    # Fit preprocessor to get feature names
    preprocessor.fit(X_train)
    
    # Get feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = np.concatenate([numerical_cols, cat_features])
    
    # Get feature importance from the best model
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(top_features['feature'], top_features['importance'], color='#2E86AB')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/figures/Feature_Importances.png", dpi=300, bbox_inches='tight')
    plt.savefig("outputs/figures/Feature_Importances.pdf", bbox_inches='tight')
    plt.show()
    print("Figure saved: outputs/figures/Feature_Importances.png and .pdf")
    
    # Save feature importance table
    feature_importance_df.to_csv("outputs/tables/feature_importance.csv", index=False)
    print("Feature importance saved to: outputs/tables/feature_importance.csv")

# ==============================================
# 12) SHAP Analysis (for tree-based models)
# ==============================================
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    print("\n=== SHAP Analysis ===")
    
    # Get a sample of test data for SHAP analysis
    X_test_processed = preprocessor.transform(X_test)
    
    # Create SHAP explainer
    if best_model_name == 'XGBoost':
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_processed)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_processed, feature_names=all_features, show=False)
        plt.tight_layout()
        plt.savefig("outputs/figures/SHAP_Summary.png", dpi=300, bbox_inches='tight')
        plt.savefig("outputs/figures/SHAP_Summary.pdf", bbox_inches='tight')
        plt.show()
        print("SHAP summary plot saved to: outputs/figures/SHAP_Summary.png and .pdf")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_processed, feature_names=all_features, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("outputs/figures/SHAP_Bar.png", dpi=300, bbox_inches='tight')
        plt.savefig("outputs/figures/SHAP_Bar.pdf", bbox_inches='tight')
        plt.show()
        print("SHAP bar plot saved to: outputs/figures/SHAP_Bar.png and .pdf")

print("\n=== Analysis Complete ===")
print("All outputs saved to 'outputs/' directory")