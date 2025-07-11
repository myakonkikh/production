import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from inference import predict
from preprocessing import split_data

def evaluate_pipline(pipe: Pipeline, X_test: np.array, y_test: np.array) -> float:
    y_pred = predict(pipe, X_test)

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    return accuracy_score(y_true=y_test, y_pred=y_pred)
