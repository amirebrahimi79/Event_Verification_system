import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'G:/term5/dev/DataSets/DataSet_Ring.csv'  # Adjust the file path if necessary
df = pd.read_csv(file_path)

# Step 1: Data Cleaning
# Drop rows with missing values (only non-null data will be used for training)
df_clean = df.dropna()

# Features and Target
X = df_clean.drop(columns=['label'])  # Features (numerical sensor data)
y = df_clean['label']  # Target variable

# Encode target variable (convert labels to numerical values)
y = y.astype('category').cat.codes  # Converts labels to 0, 1, etc.

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
random_forest.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = random_forest.predict(X_test)

# Accuracy and Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Feature Importance
feature_importances = random_forest.feature_importances_
feature_names = X.columns
print("\nFeature Importances:")
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance:.4f}")
