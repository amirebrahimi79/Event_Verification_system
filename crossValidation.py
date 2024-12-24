# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Step 3: Define the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)

# Step 4: Perform cross-validation
# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Step 5: Print the results
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))