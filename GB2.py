import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'path of your Dataset'
df = pd.read_csv(file_path)

# Drop rows with missing labels
df = df.dropna(subset=['label'])

# Separate features and target
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize (scale) the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Binarize the output labels for multiclass ROC
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)

# Predict probabilities
y_prob = gb.predict_proba(X_test)

# Initialize a dictionary to store results
results = {}
total_samples = len(y_test)  # Total samples in the test set
samples_per_day = total_samples / 30  # Assume dataset spans 30 days

# Plot ROC curve for each class and calculate metrics
plt.figure(figsize=(10, 8))

for i, class_name in enumerate(lb.classes_):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    auc_score = roc_auc_score(y_test_binarized[:, i], y_prob[:, i])
    #plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.4f})")

    # Calculate DR, FAR, EER, and DR(#FAs per day < 1)
    fars, tprs = [], []
    confusion_matrices = []
    for thresh in thresholds:
        y_pred = (y_prob[:, i] >= thresh).astype(int)
        cm = confusion_matrix(y_test_binarized[:, i], y_pred)
        confusion_matrices.append(cm)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Rate
        dr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Detection Rate
        fars.append(far)
        tprs.append(dr)

    # Find EER (FAR = 1 - DR)
    fars = np.array(fars)
    tprs = np.array(tprs)
    eer_index = np.argmin(np.abs(fars - (1 - tprs)))
    eer = fars[eer_index]

    # Calculate DR(#FAs per day < 1)
    far_per_day = fars * samples_per_day
    valid_thresholds = np.where(far_per_day < 1)[0]
    dr_fas_per_day = tprs[valid_thresholds[-1]] if len(valid_thresholds) > 0 else 0

    # Store results for this class
    results[class_name] = {
        'AUC': auc_score,
        'EER': eer,
        'DR (FAR=0)': tprs[np.where(fars == 0)[0][0]] if 0 in fars else 0,
        'FAR (DR=99%)': fars[np.where(tprs >= 0.99)[0][0]] if np.any(tprs >= 0.99) else 1,
        'DR (#FAs per day < 1)': dr_fas_per_day,
        'Confusion Matrix (Threshold=0.5)': confusion_matrices[np.where(thresholds >= 0.5)[0][0]]
    }

# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

# Add labels and legend
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("ROC Curve for All Classes")
#plt.legend(loc="best")
#plt.show()

# Display results for each class
print("\nPerformance Metrics for Each Class:")
for class_name, metrics in results.items():
    print(f"\nClass: {class_name}")
    for metric_name, value in metrics.items():
        if metric_name == 'Confusion Matrix (Threshold=0.5)':
            print(f"  {metric_name}:\n{value}")
        else:
            print(f"  {metric_name}: {value:.4f}")
