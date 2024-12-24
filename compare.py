# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Step 1: Load the dataset
file_path = 'G:/term5/dev/DataSets/DataSet_shade_down.csv'
data = pd.read_csv(file_path)

# Step 2: Preprocess the dataset
# Encode the target variable (label) to numeric values
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

# Separate features (X) and target (y)
X = data.drop(columns=['label', 'label_encoded'], axis=1)
y = data['label_encoded']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Initialize models
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# Dictionary to store results
results = {}

# Step 4: Train and evaluate each model
for name, model in models.items():
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results[name] = {
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time,
        "Accuracy": accuracy
    }

# Step 5: Display results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Training Time: {metrics['Training Time (s)']:.4f} seconds")
    print(f"  Prediction Time: {metrics['Prediction Time (s)']:.4f} seconds")
    print(f"  Accuracy: {metrics['Accuracy']:.2%}\n")
