
import numpy as np
import pandas as pd

def load_and_preprocess_data(filename): #in summary, load the data, convernts inputs to numeric, dropps useless columns such as year, makes NAN 0, and creates a feature Matrix X and label Y
    df = pd.read_csv("C:/Users/Jorge/Downloads/diabetes_dataset.csv")

    print("Dataset shape:", df.shape) #ensuring the data set is properly loaded
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    df = df.copy()

    numeric_columns = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') #force changes columns to numeric, turning to NAN is missing

    if 'gender' in df.columns: #sets females to 0, males to 1, others to 2
        df['gender'] = df['gender'].astype(str)
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2}).fillna(2)

    if 'smoking_history' in df.columns: #sets never to 0, former to 1, current to 2, no info to 3
        df['smoking_history'] = df['smoking_history'].astype(str)
        df['smoking_history'] = df['smoking_history'].map({
            'never': 0, 'former': 1, 'current': 2, 'No Info': 3, 'no info': 3
        }).fillna(3)

    #lines 30 to 47 ensure columns exist, drop year because its not revelant, and sets NAN to 0.
    feature_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
        'bmi', 'hbA1c_level', 'blood_glucose_level',
        'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other'
    ]

    feature_columns = [col for col in feature_columns if col in df.columns]

    if 'year' in df.columns:
        df = df.drop('year', axis=1)

    df = df.fillna(0)
    #creates matrix of features X (inputs turned numeric) and Labels y (yes or not diabetic)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)

    X = df[feature_columns].values.astype(np.float64)
    y = df['diabetes'].values.astype(np.float64)

    X = np.column_stack([np.ones(X.shape[0]), X])

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Number of features (including intercept): {X.shape[1]}")
    print(f"Target distribution: {np.sum(y)} positive out of {len(y)} samples ({np.mean(y) * 100:.2f}%)")

    return X, y, feature_columns

def train_test_split(X, y, test_size=0.5, random_state=42): #makes unbias training sets
    np.random.seed(random_state)
    n = X.shape[0]
    indices = np.random.permutation(n) #creates random order of indeces to ensure distinct splits each time
    split_idx = int(n * (1 - test_size)) #gets point to split

    #splits
    train_idx = indices[:split_idx] 
    test_idx = indices[split_idx:]
    #Makes the sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test): #makes stander unit for all given that each row has differnt input sizes
    X_train_std = X_train.copy().astype(np.float64)
    X_test_std = X_test.copy().astype(np.float64)

    continuous_indices = [2, 6, 7, 8] #age, bmi, hbA1c_level, blood_glucose_level

    for idx in continuous_indices:
        if idx < X_train.shape[1]:
            mean = np.mean(X_train[:, idx]) #get mean
            std = np.std(X_train[:, idx]) #get standard deviation
            if std > 0:
                X_train_std[:, idx] = (X_train[:, idx] - mean) / std #turns it into scaled value (math formula)
                X_test_std[:, idx] = (X_test[:, idx] - mean) / std #turns it into scaled value (math formula

    return X_train_std, X_test_std

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.loss_history = []

    def sigmoid(self, z): #formula for linear output as probability (squishes values)
        z = np.asarray(z, dtype=np.float64)
        z_clipped = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def compute_loss(self, X, y): #computes loss for logistic regression (how wrong it is)
        z = X @ self.weights
        predictions = self.sigmoid(z)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def fit(self, X, y): #training loop
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)

        print(f"Starting Logistic Regression training with {n_features} features...")

        for i in range(self.max_iter): #gets gradient and adjusts weights
            z = X @ self.weights
            predictions = self.sigmoid(z)

            gradient = (X.T @ (predictions - y)) / n_samples
            self.weights -= self.learning_rate * gradient

            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                self.loss_history.append(loss)

            if len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                print(f"Logistic Regression converged at iteration {i}")
                break

            if i % 500 == 0:
                loss = self.compute_loss(X, y)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X): #given X, gets us the probability
        X = X.astype(np.float64)
        return self.sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5): #condition to considering something diabetic or not based on threshold
        return (self.predict_proba(X) >= threshold).astype(int)

class LinearSVM: #draws a boundry between diabetic and non diabetic
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        y_svm = 2 * y - 1

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)

        print(f"Starting SVM training with {n_features} features...")

        for i in range(self.max_iter):
            margins = y_svm * (X @ self.weights)
            hinge_loss = np.maximum(0, 1 - margins)

            misclassified = margins < 1
            subgradient = np.zeros(n_features, dtype=np.float64)
            if np.sum(misclassified) > 0:
                subgradient = -np.sum(X[misclassified] * y_svm[misclassified, np.newaxis], axis=0)
                subgradient /= n_samples

            reg_gradient = self.lambda_param * self.weights
            gradient = reg_gradient + subgradient

            new_weights = self.weights - self.learning_rate * gradient

            hinge_loss_total = np.mean(hinge_loss)
            reg_loss = 0.5 * self.lambda_param * np.sum(self.weights ** 2)
            total_loss = hinge_loss_total + reg_loss

            if i % 100 == 0:
                self.loss_history.append(total_loss)

            if len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                print(f"SVM converged at iteration {i}")
                break

            if i % 500 == 0:
                print(f"Iteration {i}, Loss: {total_loss:.4f}")

            self.weights = new_weights

    def predict(self, X):
        X = X.astype(np.float64)
        return (X @ self.weights >= 0).astype(int)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def error_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def main():
    try:
        X, y, feature_names = load_and_preprocess_data('diabetes_dataset.csv')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


        X_train_std, X_test_std = standardize_features(X_train, X_test)

        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Positive cases in training: {np.sum(y_train)}/{len(y_train)} ({np.mean(y_train) * 100:.2f}%)")
        print(f"Positive cases in test: {np.sum(y_test)}/{len(y_test)} ({np.mean(y_test) * 100:.2f}%)")

        print("\n" + "=" * 60)
        print("TRAINING LOGISTIC REGRESSION")
        print("=" * 60)

        lr = LogisticRegression(learning_rate=0.1, max_iter=2000, tol=1e-6)
        lr.fit(X_train_std, y_train)

        print("\n" + "=" * 60)
        print("TRAINING LINEAR SVM")
        print("=" * 60)

        svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, max_iter=2000, tol=1e-6)
        svm.fit(X_train_std, y_train)

        y_pred_lr = lr.predict(X_test_std)
        y_pred_svm = svm.predict(X_test_std)

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        lr_error = error_rate(y_test, y_pred_lr)
        lr_report = classification_report(y_test, y_pred_lr)

        print("\nLOGISTIC REGRESSION RESULTS:")
        print(f"Test Error Rate: {lr_error:.4f}")
        print(f"Accuracy: {lr_report['accuracy']:.4f}")
        print(f"Precision: {lr_report['precision']:.4f}")
        print(f"Recall: {lr_report['recall']:.4f}")
        print(f"F1-Score: {lr_report['f1_score']:.4f}")
        print(f"Confusion Matrix:")
        print(lr_report['confusion_matrix'])

        svm_error = error_rate(y_test, y_pred_svm)
        svm_report = classification_report(y_test, y_pred_svm)

        print("\nLINEAR SVM RESULTS:")
        print(f"Test Error Rate: {svm_error:.4f}")
        print(f"Accuracy: {svm_report['accuracy']:.4f}")
        print(f"Precision: {svm_report['precision']:.4f}")
        print(f"Recall: {svm_report['recall']:.4f}")
        print(f"F1-Score: {svm_report['f1_score']:.4f}")
        print(f"Confusion Matrix:")
        print(svm_report['confusion_matrix'])

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        if lr_error < svm_error:
            print("Logistic Regression performs better")
            improvement = ((svm_error - lr_error) / svm_error) * 100
            print(f"Improvement: {improvement:.2f}%")
        elif svm_error < lr_error:
            print("Linear SVM performs better")
            improvement = ((lr_error - svm_error) / lr_error) * 100
            print(f"Improvement: {improvement:.2f}%")
        else:
            print("Both models perform equally well")

        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        feature_importance = np.abs(lr.weights[1:])
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': lr.weights[1:len(feature_names) + 1],
            'Absolute_Importance': np.abs(lr.weights[1:len(feature_names) + 1])
        }).sort_values('Absolute_Importance', ascending=False)

        print("Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

        print(f"\nFinal Comparison:")
        print(f"Logistic Regression Error Rate: {lr_error:.4f}")
        print(f"Linear SVM Error Rate: {svm_error:.4f}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
        main()