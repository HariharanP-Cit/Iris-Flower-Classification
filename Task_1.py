import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data  # Features: SepalLength, SepalWidth, PetalLength, PetalWidth
    y = iris.target  # Target classes (0: Setosa, 1: Versicolor, 2: Virginica)

    # Split dataset into train and test subsets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Show example prediction
    example = X_test[0].reshape(1, -1)
    predicted_class = iris.target_names[clf.predict(example)[0]]
    actual_class = iris.target_names[y_test[0]]
    print(f"Example Test Sample Prediction:")
    print(f"Features (SepalLength, SepalWidth, PetalLength, PetalWidth): {example.flatten()}")
    print(f"Predicted Species: {predicted_class}")
    print(f"Actual Species: {actual_class}")

if __name__ == "__main__":
    main()

