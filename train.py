import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
columns_to_drop = ['id','srcip', 'dstip', 'sport', 'dsport', 'Stime', 'Ltime', 'attack_cat']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label'].astype(int)
## continuous features

# One-hot encode nominal categorical features: ['proto', 'state', 'service']
X = pd.get_dummies(X, columns=['proto', 'state', 'service'], drop_first=True)

X_train, X_test, y_train, y_test = X.iloc[:len(train_df)], X.iloc[-len(test_df):],y.iloc[:len(train_df)], y.iloc[-len(test_df):]


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

train_data = PhotoDataset(X_train, y_train.values, y_train.values)
test_data = PhotoDataset(X_test, y_test.values, y_test.values)
train_loader = DataLoader(train_data,batch_size=256)
test_loader = DataLoader(test_data,batch_size=256)
cv_method = CovariateShiftMethod(randomseed=1, distance_method='WassersteinRatio', scale_method='')
weights = cv_method.compute_similarity(train_loader, test_loader)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

## rweight version

# Train Random Forest
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train, sample_weight=1/weights)
rf_pred_w = rf2.predict(X_test)



# Report
print("=== Random Forest ===")
print(classification_report(y_test, rf_pred))

# Report
print("=== Random Forest Reweight ===")
print(classification_report(y_test, rf_pred_w))


# Confusion Matrix Plot
def plot_cm(y_true, y_pred, title, filename):
    

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Save the plot to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # # Optionally display it
    # plt.show()


plot_cm(y_test, rf_pred, "Random Forest Confusion Matrix",filename='images/random_forest.png')
plot_cm(y_test, rf_pred_w, "Random Forest Reweight Confusion Matrix",filename='images/random_forest_rewight.png')
