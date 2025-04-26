import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from covariateshift_module.base_cs import CovariateShiftMethod
from utils.dataset import PhotoDataset
from torch.utils.data import DataLoader
import pdb

# Load data
train_df = pd.read_csv("CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")
df = pd.concat([train_df, test_df], ignore_index=True)

# Drop non-useful columns
columns_to_drop = ['id','srcip', 'dstip', 'sport', 'dsport', 'Stime', 'Ltime','label']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Prepare features and new labels
X = df.drop('attack_cat', axis=1)
y = df['attack_cat'].fillna('None')  # Some rows might have NaN attack_cat

# Encode 'attack_cat' into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode nominal categorical features
X = pd.get_dummies(X, columns=['proto', 'state', 'service'], drop_first=True)

# Split into training and testing
X_train, X_test = X.iloc[:len(train_df)], X.iloc[-len(test_df):]
y_train, y_test = y_encoded[:len(train_df)], y_encoded[-len(test_df):]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare DataLoader
train_data = PhotoDataset(X_train, y_train, y_train)
test_data = PhotoDataset(X_test, y_test, y_test)
train_loader = DataLoader(train_data, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)

# Compute reweighting (if you want to use it)
cv_method = CovariateShiftMethod(randomseed=1, distance_method='KLIEP', scale_method='')
weights = cv_method.compute_similarity(train_loader, test_loader)

# Train Random Forest (no reweight)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Train Random Forest (with reweight)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train, sample_weight=1/weights)
rf_pred_w = rf2.predict(X_test)

# Report
print("=== Random Forest (Multi-Class) ===")
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

print("=== Random Forest Reweighted (Multi-Class) ===")
print(classification_report(y_test, rf_pred_w, target_names=label_encoder.classes_))

# Confusion Matrix Plot
def plot_cm(y_true, y_pred, title, filename, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

# Save confusion matrices
plot_cm(y_test, rf_pred, "Random Forest Multi-Class Confusion Matrix", filename='images/random_forest_multiclass.png', labels=label_encoder.classes_)
plot_cm(y_test, rf_pred_w, "Random Forest Reweight Multi-Class Confusion Matrix", filename='images/random_forest_reweight_multiclass.png', labels=label_encoder.classes_)
