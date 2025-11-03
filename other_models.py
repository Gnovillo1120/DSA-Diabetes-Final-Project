import os
import pickle
import numpy as np
import pandas as pd
from main import load_and_preprocess_data, train_test_split, standardize_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_dataset.csv")
df = pd.read_csv(data_path)


# All of these models use Library Scikit Learn in order to generate a model prediction to look at other potential models
# that we did not implement on our own. This is just for a nice comparison to other models that are out there in statistical
# learning. The main function of this script pre-calculates the model before it is presented and ranked and visually
# compared in app.py's 'Model Comparison' tab. Our own models that we implemented are also included, seen below.
def precompute_model_comparison():
    X, y, feature_names = load_and_preprocess_data(data_path)

    # Use smaller subset for faster computation if dataset is large
    if len(X) > 10000:
        print("Large dataset detected, using subset for faster computation...")
        X, _, y, _ = train_test_split(X, y, test_size=0.8, random_state=42)  # Use 20% of data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train_std, X_test_std, _ = standardize_features(X_train, X_test)

    # OUR MODELS: Logistic and Linear SVM model train for comparison
    with open("logistic_regression_model.pkl", "rb") as f:
        lr_model, feature_names, scaler_value = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm_model, _, _ = pickle.load(f)

    X_test_std_with_intercept = np.hstack([np.ones((X_test_std.shape[0], 1)), X_test_std])

    lr_probs = lr_model.predict_proba(X_test_std_with_intercept)
    svm_probs = svm_model.predict_proba(X_test_std_with_intercept)

    # Scikit-learn models! Not our work but for comparison purposes and funsies! The more the merrier, right?
    sklearn_models = {
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'RBF SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    sklearn_results = {}

    for name, model in sklearn_models.items():
        print(f"Training {name}...")
        try:
            if name == 'RBF SVM': #Radial SVM was taking a bit to load due to a larger training set and its own features
                if len(X_train_std) > 5000: # This lowers the training set values a little for ease and pace
                    sample_idx = np.random.choice(len(X_train_std), min(5000, len(X_train_std)), replace=False)
                    X_train_subset = X_train_std[sample_idx]
                    y_train_subset = y_train[sample_idx]
                    model.fit(X_train_subset, y_train_subset)
                else:
                    model.fit(X_train_std, y_train)
            else:
                model.fit(X_train_std, y_train)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test_std)[:, 1]
            else:
                probs = model.predict(X_test_std)

            sklearn_results[name] = probs
            print(f"   {name} completed successfully")

        except Exception as e:
            print(f" {name} failed: {str(e)} Using dummy variables ")
            # Use dummy predictions for failed models
            sklearn_results[name] = np.full_like(y_test, 0.5, dtype=float)

    # Combiness all prediction models, both ours and SKL lib
    all_predictions = {
        'Logistic Regression': lr_probs,
        'Linear SVM': svm_probs,
        **sklearn_results
    }

    # stats on stats, haha! In statistics, we use some of these to measure good fit, some use different measures than others, so we included all here
    metrics_data = []
    roc_data = {}

    for model_name, probs in all_predictions.items():
        try:
            preds = (probs > 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, probs)

            metrics_data.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, preds),
                'Precision': precision_score(y_test, preds, zero_division=0),
                'Recall': recall_score(y_test, preds, zero_division=0),
                'F1-Score': f1_score(y_test, preds, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, probs) if len(np.unique(probs)) > 1 else 0.5
            })

            roc_data[model_name] = {'fpr': fpr, 'tpr': tpr}

        except Exception as e:
            print(f" Metrics failed for {model_name}: {str(e)}. Replacing with default stats")
            # Add default metrics for failed models
            metrics_data.append({
                'Model': model_name,
                'Accuracy': 0.5,
                'Precision': 0.5,
                'Recall': 0.5,
                'F1-Score': 0.5,
                'AUC-ROC': 0.5
            })
            roc_data[model_name] = {'fpr': [0, 1], 'tpr': [0, 1]}

    # Save precomputed results
    precomputed_data = {
        'all_predictions': all_predictions,
        'y_test': y_test,
        'metrics_df': pd.DataFrame(metrics_data),
        'roc_data': roc_data
    }

    with open('precomputed_model_comparison.pkl', 'wb') as f:
        pickle.dump(precomputed_data, f)


if __name__ == "__main__":
    precompute_model_comparison()