# main.py

# 1. Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load and Prepare the Dataset ---
# The Breast Cancer dataset is a classic binary classification dataset.
print("Step 1: Loading and Preparing Data...")
data = load_breast_cancer()
X, y = data.data, data.target

# For visualization purposes, we will only use the first two features.
# This helps us plot the decision boundary on a 2D graph.
X_2d = X[:, :2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

# Scale the data for better SVM performance
# SVMs are sensitive to feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_2d_train = scaler.fit_transform(X_2d_train)
X_2d_test = scaler.transform(X_2d_test)
print("Data loading and preparation complete.\n")


# --- Step 2: Train SVM with Linear and RBF Kernels ---
print("Step 2: Training SVM with Linear and RBF Kernels...")
# Initialize the SVM classifiers
svm_linear = SVC(kernel='linear', random_state=42)
svm_rbf = SVC(kernel='rbf', random_state=42)

# Train the models on the 2D data for visualization
svm_linear.fit(X_2d_train, y_2d_train)
svm_rbf.fit(X_2d_train, y_2d_train)
print("Initial model training complete.\n")


# --- Step 3: Visualize the Decision Boundaries ---
print("Step 3: Visualizing Decision Boundaries...")
def plot_decision_boundary(clf, X, y, title):
    """Utility function to plot the decision boundary."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Visualize the Linear SVM Decision Boundary
plot_decision_boundary(svm_linear, X_2d_train, y_2d_train, 'Linear SVM - Decision Boundary')

# Visualize the RBF SVM Decision Boundary
plot_decision_boundary(svm_rbf, X_2d_train, y_2d_train, 'RBF SVM - Decision Boundary')
print("Visualization complete. Check the plots.\n")


# --- Step 4 & 5: Hyperparameter Tuning and Cross-Validation ---
print("Step 4 & 5: Tuning Hyperparameters with Cross-Validation...")
# Define the parameter grid for C and gamma
# C: Regularization parameter. Controls the trade-off between a smooth decision boundary and classifying training points correctly.
# gamma: Kernel coefficient for 'rbf'. Defines how much influence a single training example has.
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf'] # We focus on RBF as it is more complex to tune
}

# Use GridSearchCV for cross-validated hyperparameter tuning
# cv=5 means 5-fold cross-validation
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

# Train on the full dataset (not just 2D) for best performance
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f"Best Parameters found: {grid_search.best_params_}")
print("Hyperparameter tuning complete.\n")


# --- Final Model Evaluation ---
print("Final Model Evaluation:")
# Get the best model from the grid search
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))