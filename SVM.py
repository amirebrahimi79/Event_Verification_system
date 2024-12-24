import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
# Load the dataset
file_path = 'G:/term5/dev/DataSets/DataSet_Ring.csv'
dataset = pd.read_csv(file_path)

# Drop rows with missing values
cleaned_data = dataset.dropna()

# Separate features (X) and target (y)
X = cleaned_data.drop(columns=['label'])
y = cleaned_data['label']

# Encode the target variable (label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train an SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

joblib.dump(svm_classifier, 'G:/term5/dev/svm_classifier_model.pkl')
print("Model saved as 'svm_classifier_model.pkl'")