import os
import pandas as pd
import pickle
import numpy as np
from main import load_and_preprocess_data, train_test_split, standardize_features, LinearSVM, LogisticRegression


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_dataset.csv")
df = pd.read_csv(data_path)


#This script trains our models that we implemented ourselves! This is from main but it makes it more clear and distinct here :)
def save_trained_models(): 
    # Copied from main
    X, y, feature_names = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train_std, X_test_std, scaler_value = standardize_features(X_train, X_test)
    X_train_std = np.column_stack((np.ones(X_train_std.shape[0]), X_train_std))
    X_test_std = np.column_stack((np.ones(X_test_std.shape[0]), X_test_std))
    feature_names = ['intercept'] + feature_names

    lr = LogisticRegression(learning_rate=0.001, max_iter=10000, tol=1e-8)
    lr.fit(X_train_std, y_train)
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.0005, max_iter=10000, tol=1e-8)
    svm.fit(X_train_std, y_train)

    #Saves the trained models with pickle!
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump((lr, feature_names, scaler_value), f)

    with open("svm_model.pkl", "wb") as f:
        pickle.dump((svm, feature_names, scaler_value), f)


if __name__ == "__main__":
    save_trained_models()