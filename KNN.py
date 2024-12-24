import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'path of your Dataset'  # Adjust path if needed
df = pd.read_csv(file_path)

# Step 1: Data Cleaning
# Drop rows with missing values (only non-null data will be used for training)
df_clean = df.dropna()

# Features and Target
X = df_clean.drop(columns=['label'])  # Features (numerical sensor data)
y = df_clean['label']  # Target variable

# Encode target variable if necessary (convert labels to numerical values)
y = y.astype('category').cat.codes  # Converts labels to 0, 1, etc.

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train K-Nearest Neighbors (KNN) Model
k = 3  # Number of neighbors, you can tune this
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = knn.predict(X_test)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
