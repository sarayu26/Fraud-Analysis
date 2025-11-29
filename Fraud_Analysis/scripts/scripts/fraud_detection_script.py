import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- 1. Load & Audit Data -----
df = pd.read_excel('C:/Users/SARAYU/OneDrive/Desktop/Fraud_Analysis/data/Fraud_Analysis_Dataset.xlsx')

print("\nBasic Info:")
print(df.info())
print("\nMissing values:", df.isnull().sum())
print("\nClass balance:\n", df['isFraud'].value_counts())
print("\nType distribution:\n", df['type'].value_counts())
print("\nDuplicates:", df.duplicated().sum())

# ----- 2. Feature Engineering & Preprocessing -----
df_model = df.drop(['nameOrig', 'nameDest'], axis=1)
df_model['orig_bal_diff'] = df_model['oldbalanceOrg'] - df_model['newbalanceOrig']
df_model['dest_bal_diff'] = df_model['oldbalanceDest'] - df_model['newbalanceDest']
df_model['orig_ratio'] = df_model['amount'] / (df_model['oldbalanceOrg'] + 1)
df_model['dest_ratio'] = df_model['amount'] / (df_model['oldbalanceDest'] + 1)
df_model = df_model[(df_model['oldbalanceOrg'] >= 0) & (df_model['oldbalanceDest'] >= 0)]
df_model = pd.get_dummies(df_model, columns=['type'], drop_first=False)

from sklearn.preprocessing import StandardScaler
cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                 'orig_bal_diff', 'dest_bal_diff', 'orig_ratio', 'dest_ratio']
scaler = StandardScaler()
df_model[cols_to_scale] = scaler.fit_transform(df_model[cols_to_scale])

print("\nFeature audit after engineering/scaling:")
print(df_model.describe())
print(df_model.info())
print(df_model['isFraud'].value_counts())

# ----- 3. Train-Test Split & SMOTE -----
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nTrain/test shapes after SMOTE:", X_train_res.shape, X_test.shape)
print("Resampled train balance:", y_train_res.value_counts())

# ----- 4. Model Training -----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_train_res, y_train_res)
y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

lgbm = lgb.LGBMClassifier(n_estimators=200, random_state=42)
lgbm.fit(X_train_res, y_train_res)
y_pred_lgbm = lgbm.predict(X_test)
y_pred_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]

# ----- 5. Model Evaluation -----
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_score, recall_score, f1_score, roc_curve)

models = {
    'Logistic Regression': (y_pred_lr, y_pred_proba_lr),
    'Random Forest': (y_pred_rf, y_pred_proba_rf),
    'Gradient Boosting': (y_pred_gb, y_pred_proba_gb),
    'LightGBM': (y_pred_lgbm, y_pred_proba_lgbm)
}
for model_name, (y_pred, y_pred_proba) in models.items():
    print(f"\n--- {model_name} ---")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_pred_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance
def plot_feature_importance(model, X, title):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=X.columns[indices], y=importances[indices])
        plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_feature_importance(rf, X_train_res, 'Random Forest Feature Importance')
plot_feature_importance(gb, X_train_res, 'Gradient Boosting Feature Importance')
plot_feature_importance(lgbm, X_train_res, 'LightGBM Feature Importance')

# ----- 6. Financial Impact Analysis -----
test_results = X_test.copy()
test_results['true_label'] = y_test.values
test_results['pred_label'] = y_pred_lgbm  # Select your main model
test_results['amount'] = df.loc[X_test.index, 'amount']

false_negatives = test_results[(test_results['true_label']==1) & (test_results['pred_label']==0)]
false_positives = test_results[(test_results['true_label']==0) & (test_results['pred_label']==1)]
true_positives  = test_results[(test_results['true_label']==1) & (test_results['pred_label']==1)]

total_lost_due_to_fraud = false_negatives['amount'].sum()
total_investigation_cost = false_positives['amount'].sum()  # Or multiply by cost per investigation
total_saved_amount = true_positives['amount'].sum()

print("Estimated financial loss due to missed fraud:", total_lost_due_to_fraud)
print("Estimated operational cost from false alarms:", total_investigation_cost)
print("Potential fraud value detected and saved:", total_saved_amount)
