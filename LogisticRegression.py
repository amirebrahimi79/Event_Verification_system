import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
file_path = 'G:/term5/dev/DataSets/DataSet_Ring.csv'  # Adjust the file path if necessary
df = pd.read_csv(file_path)

# Step 1: Data Cleaning
# Drop rows with missing values
df_clean = df.dropna()

# Features and Target
X = df_clean.drop(columns=['label'])  # Features (numerical sensor data)
y = df_clean['label']  # Target variable

# Encode the target variable (label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode target variable (convert labels to numerical values)
y = y.astype('category').cat.codes  # Converts labels like 'Non-Event' to 0, 1, etc.

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)  # Increased iterations for convergence
log_reg.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = log_reg.predict(X_test)

# Accuracy and Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)