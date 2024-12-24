import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix
import matplotlib.pyplot as plt
import joblib
# Load the dataset
file_path = 'path of your Dataset'  # Replace with your file path
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

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = gb.predict(X_test)

# Step 5: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Compute the Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Save the trained model

joblib.dump(gb, 'path to save Model to .pkl format')
print("Model saved as 'gradient_boosting_model.pkl'")
