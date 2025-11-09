import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_dataset.csv")
df = pd.read_csv(data_path)


def custom_train_test_split(X, y, test_size=0.5, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=2000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        prev_loss = float('inf')
        lr = self.learning_rate
        print(f"Starting Logistic Regression training with {n_features} features...")
        for i in range(self.max_iter):
            z = X @ self.weights
            predictions = self.sigmoid(z)

            gradient = (X.T @ (predictions - y)) / n_samples
            self.weights -= lr * gradient

            predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            if loss > prev_loss * 1.5:
                lr *= 0.5
                print(f"loss occured at {i}")
                break
            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at iteration {i}")
                break
            prev_loss = loss
            if i % 500 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.3, return_proba=False):
        prob = self.predict_proba(X)
        preds = np.where(prob >= threshold, 1, 0)
        return (preds, prob) if return_proba else preds


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_iter=2000, tol=1e-6):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def fit(self, X, y):
        y_svm = 2 * y - 1
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        prev_loss = float('inf')

        for i in range(self.max_iter):
            margins = y_svm * (X @ self.weights)
            hinge_loss = np.maximum(0, 1 - margins)
            misclassified = margins < 1
            if np.any(misclassified):
                subgrad = -np.sum(X[misclassified] * y_svm[misclassified, None], axis=0) / n_samples
            else:
                subgrad = np.zeros(n_features)
            reg_grad = self.lambda_param * self.weights
            gradient = reg_grad + subgrad
            self.weights -= self.learning_rate * gradient

            loss = np.mean(hinge_loss) + 0.5 * self.lambda_param * np.sum(self.weights ** 2)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

    def predict_proba(self, X):
        decision = X @ self.weights
        return 1 / (1 + np.exp(-decision))

    def predict(self, X):
        linear_output = X @ self.weights
        return np.where(linear_output >= 0, 1, 0)


def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)

    cat_columns = ['gender', 'smoking_history']
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            if col == 'gender':
                df[col] = df[col].map({'Female': 0, 'Male': 1, 'Other': 2}).fillna(2)
            else:  # smoking_history
                df[col] = df[col].map({'never': 0, 'former': 1, 'current': 2, 'No Info': 3, 'no info': 3}).fillna(3)

    numeric_cols = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level',
                    'hypertension', 'heart_disease',
                    'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']

    # Create feature columns list
    feature_cols = []
    for col in numeric_cols + ['gender', 'smoking_history']:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].values
    y = df['diabetes'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler  # Fixed return statement


def evaluate_model(model, X_test, y_test):
    try:
        probs = model.predict_proba(X_test)
        if probs.ndim > 1:  # Handle binary classification probability arrays
            probs = probs[:, 1]
    except AttributeError:
        try:
            decision = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-decision))
        except AttributeError:
            preds = model.predict(X_test)
            probs = preds
    mse = mean_squared_error(y_test, probs)
    return mse, probs


def main():
    # Fixed: Correct number of return values
    X, y, features, scaler = load_and_preprocess_data(data_path)

    # Use custom split to avoid sklearn conflict
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.5, random_state=42)

    # Our models, LR and LSVM
    lr = LogisticRegression(learning_rate=0.1, max_iter=2000, tol=1e-6)
    lr.fit(X_train, y_train)
    mse_lr = mean_squared_error(y_test, lr.predict_proba(X_test))

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, max_iter=2000, tol=1e-6)
    svm.fit(X_train, y_train)
    decision = X_test @ svm.weights
    probs_svm = 1 / (1 + np.exp(-decision))
    mse_svm = mean_squared_error(y_test, probs_svm)

    # Scikit-learn models
    sklearn_models = {
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'RBF SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }
    sklearn_mse = {}
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        mse, _ = evaluate_model(model, X_test, y_test)
        sklearn_mse[name] = mse

    all_results = {'Logistic Regression': mse_lr, 'Linear SVM': mse_svm}
    all_results.update(sklearn_mse)

    print("\nModel Mean Squared Errors:")
    for name, mse in all_results.items():
        print(f"{name}: {mse:.6f}")

    best_model_name = min(all_results, key=all_results.get)
    print(f"\nBest model by MSE: {best_model_name}")

    print("\nFeature coefficients (or importance) for best model:")
    if best_model_name == 'Logistic Regression':
        coefs = lr.weights
        feature_names = ['Intercept'] + features
        for feat, coef in zip(feature_names, coefs):
            print(f"{feat}: {coef:.6f}")
    elif best_model_name == 'Linear SVM':
        coefs = svm.weights
        feature_names = features  # No intercept for SVM in your implementation
        for feat, coef in zip(feature_names, coefs):
            print(f"{feat}: {coef:.6f}")
    else:
        best_model = sklearn_models[best_model_name]
        feature_names = features
        if hasattr(best_model, 'coef_'):
            coefs = best_model.coef_.ravel()
            for feat, coef in zip(feature_names, coefs):
                print(f"{feat}: {coef:.6f}")
        elif hasattr(best_model, 'feature_log_prob_'):  # Naive Bayes
            coefs = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
            for feat, coef in zip(feature_names, coefs):
                print(f"{feat}: {coef:.6f}")
        else:
            print("No coefficient information available")


if __name__ == "__main__":
    main()