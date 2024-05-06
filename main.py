# Import necessary libraries for data visualization, numerical operations, and machine learning
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the built-in Iris dataset from scikit-learn
iris = load_iris()
print("Iris Dataset: ")
print(iris)

# Extract the feature matrix (X) from the Iris dataset
X = iris.data  # Features
print("Content of X:")
print(X)

# Extract the target vector (y) from the Iris dataset
y = iris.target  # Target variable
print("Content of Y:")
print(y)

# Print the names of the features and target classes for clarity
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a StandardScaler to normalize the feature values
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_scaled = scaler.transform(X_test)

# Create a K-Nearest Neighbors classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the scaled training data
knn.fit(X_train_scaled, y_train)

# Predict the class labels for the scaled test data
y_pred = knn.predict(X_test_scaled)

# Calculate and print the accuracy of the classifier on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Begin plotting the classified items
plt.figure(figsize=(8, 6))

# Loop through each unique class predicted to plot them
for class_label in np.unique(y_pred):
    plt.scatter(X_test_scaled[y_pred == class_label, 0], X_test_scaled[y_pred == class_label, 1],
                label=f'Class {class_label}')  # Plot data points for each class

# Calculate and plot centroids for each class
for class_label in np.unique(y_pred):
    centroid = np.mean(X_test_scaled[y_pred == class_label], axis=0)  # Compute the centroid of each class
    plt.scatter(centroid[0], centroid[1], s=100, marker='x', color='black')  # Plot centroid

# Label the axes, add a title, legend, and grid to the plot
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('Final Classified Items')
plt.legend()
plt.grid(True)
plt.show()  # Display the plot
