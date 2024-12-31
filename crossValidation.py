import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'path of dataset'  # Replace with your file path
df = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Drop rows with missing labels
df = df.dropna(subset=['label'])

# Separate features and target
X = df.drop(columns=['label'])  # Features (all columns except 'label')
y = df['label']                 # Target

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize (scale) the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 2: Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
train_errors = []
validation_errors = []

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train the model
    gb.fit(X_train, y_train)
    
    # Training Error
    train_pred = gb.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_errors.append(1 - train_accuracy)  # 1 - accuracy is the error
    
    # Validation Error
    val_pred = gb.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    validation_errors.append(1 - val_accuracy)

# Calculate average training and validation errors
avg_train_error = np.mean(train_errors)
avg_val_error = np.mean(validation_errors)

print(f"Average Training Error: {avg_train_error:.4f}")
print(f"Average Validation Error: {avg_val_error:.4f}")

# Step 3: Train-Test Split for Final Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# Step 4: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Compute the Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


