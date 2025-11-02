import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error


def train_test_split(X, y, test_size=0.5, random_state=42):
    np.random.seed(random_state)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_features(X_train, X_test):
    X_train_std = X_train.copy().astype(np.float64)
    X_test_std = X_test.copy().astype(np.float64)
    means = {}
    stds = {}
    continuous_indices = [1, 5, 6, 7]  # age, bmi, hbA1c_level, blood_glucose_level
    for idx in continuous_indices:
        if idx < X_train.shape[1]:
            mean = np.mean(X_train[:, idx])  # get mean
            std = np.std(X_train[:, idx])  # get standard deviation
            means[idx] = mean
            stds[idx] = std
            if std > 0:
                X_train_std[:, idx] = (X_train[:, idx] - mean) / std  # turns it into scaled value (math formula)
                X_test_std[:, idx] = (X_test[:, idx] - mean) / std  # turns it into scaled value (math formula
    return X_train_std, X_test_std, (means, stds)


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
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        prev_loss = float('inf')
        lr = self.learning_rate
        print(f"Starting Logistic Regression training with {n_features} features...")
        for i in range(self.max_iter):  # gets gradient and adjusts weights
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
        return self.sigmoid(X @ self.weights)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.where(prob >= 0.5, 1, 0)


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


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'year' in df.columns:
        df = df.drop('year', axis=1)

    feature_cols = [col for col in numeric_cols + ['gender', 'smoking_history'] if col in df.columns]

    X = df[feature_cols].values
    y = df['diabetes'].values

    return X, y, feature_cols


def evaluate_model(model, X_test, y_test):
    # Try predict_proba, then decision_function, else fallback to predict
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            decision = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-decision))
        except AttributeError:
            preds = model.predict(X_test)
            probs = preds  # not probabilities
    mse = mean_squared_error(y_test, probs)
    return mse, probs


def main():
    # Load data
    X, y, features = load_and_preprocess_data("C:/Users/anvis/Downloads/diabetes_dataset.csv")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Standardize features
    X_train_std, X_test_std, (means, stds)= standardize_features(X_train, X_test)

    # Initialize and train your hand-coded models
    lr = LogisticRegression(learning_rate=0.1, max_iter=2000, tol=1e-6)
    lr.fit(X_train_std, y_train)
    mse_lr = mean_squared_error(y_test, lr.predict_proba(X_test_std))

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, max_iter=2000, tol=1e-6)
    svm.fit(X_train_std, y_train)
    decision = X_test_std @ svm.weights
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
        model.fit(X_train_std, y_train)
        mse, _ = evaluate_model(model, X_test_std, y_test)
        sklearn_mse[name] = mse

    # Combine all results
    all_results = {'Logistic Regression': mse_lr, 'Linear SVM': mse_svm}
    all_results.update(sklearn_mse)

    # Print all MSEs
    print("\nModel Mean Squared Errors:")
    for name, mse in all_results.items():
        print(f"{name}: {mse:.6f}")

    # Determine best model
    best_model_name = min(all_results, key=all_results.get)
    print(f"\nBest model by MSE: {best_model_name}")

    # Print coefficients / feature importances for best model
    print("\nFeature coefficients (or importance) for best model:")
    if best_model_name == 'Logistic Regression':
        coefs = lr.weights
        feature_names = ['Intercept'] + features
        for feat, coef in zip(feature_names, coefs):
            print(f"{feat}: {coef:.6f}")
    elif best_model_name == 'Linear SVM':
        coefs = svm.weights
        feature_names = ['Intercept'] + features
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