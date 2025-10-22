import numpy as np
import matplotlib.pyplot as plt
from data import load_iris_data, load_wine_data
from utils import train_test_split
from knn_classifier import KNNClassifier

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_knn(X, y, k, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    return calculate_accuracy(y_test, predictions)

def main():
    X, y = load_iris_data()
    
    print("KNN Classification for iris dataset (k=3)")
    accuracy = evaluate_knn(X, y, k=3)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    print("Hyperparameter Tuning")
    k_values = [1, 3, 5, 7, 9, 11, 15]
    accuracies = []
    
    for k in k_values:
        accuracy = evaluate_knn(X, y, k)
        accuracies.append(accuracy)
        print(f"k = {k:2d}, Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-')
    plt.xlabel('k-value')
    plt.ylabel('Accuracy')
    plt.title('KNN Classifier Performance vs k-value')
    plt.grid(True)
    
    best_k_idx = np.argmax(accuracies)
    best_k = k_values[best_k_idx]
    best_accuracy = accuracies[best_k_idx]
    
    plt.plot(best_k, best_accuracy, 'ro', markersize=10, label=f'Best k={best_k}')
    plt.legend()
    
    print(f"\nBest Performance: k={best_k}, Accuracy={best_accuracy:.4f}")
    plt.show()

    print("\nEvaluating on Wine dataset using best k")
    X_wine, y_wine = load_wine_data()
    wine_accuracy = evaluate_knn(X_wine, y_wine, k=best_k)
    print(f"Wine Dataset Accuracy (k={best_k}): {wine_accuracy:.4f}")

main()