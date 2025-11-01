import pickle
from main import load_and_preprocess_data, train_test_split, standardize_features, LinearSVM, LogisticRegression

def save_trained_models():
    #copied from main
    X, y, feature_names = load_and_preprocess_data('diabetes_dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    lr = LogisticRegression(learning_rate=0.1, max_iter=2000, tol=1e-6)
    lr.fit(X_train_std, y_train)
    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, max_iter=2000, tol=1e-6)
    svm.fit(X_train_std, y_train)

    #save the trained models with pickle
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump((lr, feature_names), f)

    with open("svm_model.pkl", "wb") as f:
        pickle.dump((svm, feature_names), f)


if __name__ == "__main__":
    save_trained_models()





















