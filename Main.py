import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, normalize='true')  # Normalized for better interpretation
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                fmt=".2%")  # Format as a percentage
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Load the data
data_csv = "diabetes_PIMA_preprocessed.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(data_csv, skiprows=1, names=column_names)  # Skip the first row

# Define features and target variable
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
logreg_model = LogisticRegression()

# Train the logistic regression model
logreg_model.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]

# Threshold for binary classification (you can choose based on your preference)
threshold = 0.5
y_pred = (y_pred_proba > threshold).astype(int)

# Compare model results vs true results
plot_confusion_matrix(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Decision Tree
# Initialize scikit-learn DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=3)

# Train the scikit-learn decision tree model
dt_model.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(16, 12))
plot_tree(dt_model, filled=True, feature_names=column_names[:-1], class_names=['No Diabetes', 'Diabetes'])
plt.show()

