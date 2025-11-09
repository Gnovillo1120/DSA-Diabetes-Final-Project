import os
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from main import load_and_preprocess_data, custom_train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_dataset.csv")

    # Load data and scaler from main.py
X, y, feature_names, scaler_value = load_and_preprocess_data(data_path)

    # Load models with error handling
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        lr_data = pickle.load(f)
        if isinstance(lr_data, tuple):
            if len(lr_data) == 3:
                lr_model, lr_feature_names, lr_scaler = lr_data
            else:
                lr_model, lr_feature_names = lr_data
        else:  # Dictionary format
            lr_model = lr_data['model']
            lr_feature_names = lr_data['feature_names']
except FileNotFoundError:
    st.error("Logistic Regression model not found. Please run train_models.py first.")
    lr_model, lr_feature_names = None, None

try:
    with open("svm_model.pkl", "rb") as f:
        svm_data = pickle.load(f)
        if isinstance(svm_data, tuple):
            if len(svm_data) == 3:
                svm_model, svm_feature_names, svm_scaler = svm_data
            else:
                svm_model, svm_feature_names = svm_data
        else:  # Dictionary format
            svm_model = svm_data['model']
            svm_feature_names = svm_data['feature_names']
except FileNotFoundError:
    st.error("SVM model not found. Please run train_models.py first.")
    svm_model, svm_feature_names = None, None

    # Use the scaler from main.py for scaling new inputs
def scale(input_data, scaler):
    if scaler is not None:
        return scaler.transform(input_data)
    else:
        st.warning("No scaler found. Using raw data.")
        return input_data


# Makes visual for whether diabetic or not
def result_card(value, title, non_diabetic=True):
    if non_diabetic:
        color = "#e6fffa"
    else:
        color = "#ffe6e6"
    icon = "✅" if non_diabetic else "⚠️"
    html_card = f"""<div style = "background-color: {color}; display:inline-block; border:1px solid #ddd; border-radius:10px; width:280px; height:140px; padding:5px; margin:10px; vertical-align:top;">
    <div style = "display:flex; align-items:center; justify-content:space-between;">
        <h4 style = "font-family:sans-serif; font-weight:bold; margin:0; font-size:16px">{title}</h4>
        <span style = "font-size:30px; right:30px;">{icon}</span>
    </div>
    <div style = "position:absolute; bottom:1px; left:15px;">
        <h3 style = "font-family:sans-serif; font-weight:bold; margin:0; text-align:center; white-space:nowrap; font-size:16px">{value}</h3>
    </div>
    </div>
    """
    return html_card


def train_sklearn_models(X_train_std, X_test_std, y_train, y_test): #Trains SKL models
    sklearn_models = {
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'RBF SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    model_results = {}
    for name, model in sklearn_models.items():
        model.fit(X_train_std, y_train)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test_std)[:, 1]
        else:
            probs = model.predict(X_test_std)
        model_results[name] = probs

    return model_results, sklearn_models


st.sidebar.title("Diabetes Prediction Dashboard") #Main visuals for app
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Diabetes Risk Calculator", "Model Comparison", "Feature Importance", "BMI by Race"] # Tab titles
    )

if selected == "Diabetes Risk Calculator": #Visual set up and user intake for calculator
    st.title("Diabetes Risk Calculator")
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=125, value=30)
    hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = st.selectbox("Heart Disease", options=["Yes", "No"])
    heart_disease = 1 if heart_disease == "Yes" else 0
    smoking = st.selectbox("Smoking Status", options=["Never", "Former", "Current", "No Information"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=252.0, value=20.0)
    h1ba1c = st.number_input("Hba1c Level", min_value=0.0, max_value=20.0, value=5.5)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=600.0, value=80.0)

    if st.button("Calculate Risk"):
        gender_options = {"Female": 0, "Male": 1, "Other": 2}
        smoking_options = {"Never": 0, "Former": 1, "Current": 2, "No Information": 3}
        user_info = {
            'age': age,
            'bmi': bmi,
            'hbA1c_level': h1ba1c,
            'blood_glucose_level': glucose,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'race:AfricanAmerican': 0,
            'race:Asian': 0,
            'race:Caucasian': 0,
            'race:Hispanic': 0,
            'race:Other': 0,
            'gender': gender_options[gender],
            'smoking_history': smoking_options[smoking],
        }
        user_input = np.array(list(user_info.values()), dtype=float).reshape(1, -1) #Takes in user input
        user_input_scaled = scale(user_input, scaler_value)
        formated_array = np.hstack([np.ones((user_input_scaled.shape[0], 1)), user_input_scaled])
        st.session_state["user_input"] = formated_array

        log_regression_probability = lr_model.predict_proba(formated_array)[0] #Evaluates based on Logistic and svm
        log_regression_prediction = lr_model.predict(formated_array)[0]
        svm_probability = svm_model.predict_proba(formated_array)[0]
        svm_prediction = svm_model.predict(formated_array)[0]

        st.subheader("Prediction Results: ")
        label_dictionary = {0: "Non-Diabetic", 1: "Diabetic"}
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                result_card(
                    f"{label_dictionary[log_regression_prediction]}",
                    "Logistic Regression",
                    non_diabetic=(log_regression_prediction == 0)),
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                result_card(
                    f"{label_dictionary[svm_prediction]}",
                    "Linear SVM",
                    non_diabetic=(svm_prediction == 0)),
                unsafe_allow_html=True)


elif selected == "Model Comparison":
    st.title("🏆 Model Performance Comparison")

    try:
        # Load precomputed results
        @st.cache_data
        def load_precomputed_results():
            with open('precomputed_model_comparison.pkl', 'rb') as f:
                return pickle.load(f)


        data = load_precomputed_results()
        all_predictions = data['all_predictions']
        y_test = data['y_test']
        metrics_df = data['metrics_df']

        valid_models = {}
        for model_name, probs in all_predictions.items():
            if len(np.unique(probs)) > 1 and not np.all(probs == 0.5):
                valid_models[model_name] = probs
            else:
                st.warning(f"⚠️ {model_name} had computation issues and was excluded")

        if not valid_models:
            st.error("❌ No valid models found in the precomputed data!")

        mse_scores = {}
        for model_name, probs in valid_models.items():
            mse = np.mean((y_test - probs) ** 2)
            mse_scores[model_name] = mse

        comparison_data = []
        for model_name in valid_models.keys():
            row = metrics_df[metrics_df['Model'] == model_name].iloc[0]
            comparison_data.append({
                'Model': model_name,
                'MSE': mse_scores[model_name],
                'Accuracy': row['Accuracy'],
                'AUC-ROC': row['AUC-ROC'],
                'Rank': 0
            })

        comparison_df = pd.DataFrame(comparison_data)

        comparison_df = comparison_df.sort_values('MSE')
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)

        best_model = comparison_df.iloc[0]
        st.subheader("🎯 Best Performing Model")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🏆 Winner",
                value=best_model['Model'],
                delta="Lowest MSE"
            )
        with col2:
            st.metric(
                label="MSE Score",
                value=f"{best_model['MSE']:.4f}",
                delta="Lower is better"
            )
        with col3:
            st.metric(
                label="Accuracy",
                value=f"{best_model['Accuracy']:.1%}",
            )

        st.divider()

        # Simple comparison table
        st.subheader("📊 Model Performance Ranking")

        # Format the table nicely
        display_df = comparison_df.copy()
        display_df['MSE'] = display_df['MSE'].apply(lambda x: f"{x:.4f}")
        display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.1%}")
        display_df['AUC-ROC'] = display_df['AUC-ROC'].apply(lambda x: f"{x:.3f}")

        st.info("""
        **📖 How to read this table:**
        - **MSE (Mean Squared Error)**: Lower values are better - measures prediction accuracy
        - **Accuracy**: Percentage of correct predictions - higher is better  
        - **AUC-ROC**: Measures how well the model separates diabetic vs non-diabetic - closer to 1.0 is better
        - **Rank**: Models are ranked by MSE (most important metric)
        """)

        st.subheader("📈 MSE Comparison (Lower is Better)")

        fig = px.bar(comparison_df,
                     x='Model',
                     y='MSE',
                     color='MSE',
                     color_continuous_scale='viridis_r',
                     title="Model Error Comparison")

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.info("Please run `precomputed_models.py` first to generate the comparison data.")
elif selected == "Feature Importance":
        st.title("Feature Importance in Logistic Regression")

        if lr_model is None:
            st.error("Logistic Regression model not loaded.")
        else:
            weights = lr_model.weights

            # Simple reliable approach
            if lr_feature_names and len(lr_feature_names) == len(weights):
                display_features = lr_feature_names
            else:
                # Use main.py features and handle intercept
                if len(weights) == len(feature_names) + 1:
                    display_features = ['intercept'] + feature_names
                else:
                    display_features = feature_names
                    # If we're still short, pad with generic names
                    while len(display_features) < len(weights):
                        display_features.append(f'feature_{len(display_features)}')
                    # If we have too many, truncate
                    display_features = display_features[:len(weights)]

            # Create the dataframe
            feature_dataframe = pd.DataFrame({
                "Feature": display_features,
                "Weight": weights,
                "Absolute Weight": np.abs(weights)
            }).sort_values(by="Absolute Weight", ascending=False)

            top_features = feature_dataframe.head(15)

            # Create the chart with better formatting
            chart = (alt.Chart(top_features).mark_bar().encode(
                x=alt.X("Absolute Weight:Q", title="Importance (Absolute Weight)"),
                y=alt.Y("Feature:N", sort='-x', title="Feature"),
                color=alt.condition(
                    alt.datum.Weight > 0,
                    alt.value("#ff6b6b"),  # Red for positive
                    alt.value("#4ecdc4")  # Green for negative
                ),
                tooltip=['Feature', 'Weight', 'Absolute Weight']
            ).properties(
                title="Top 15 Most Important Features for Diabetes Prediction",
                height=400
            ))

            st.altair_chart(chart, use_container_width=True)

            # Display detailed table
            st.subheader("Detailed Feature Weights")

            # Format the dataframe for better display
            display_df = feature_dataframe.copy()
            display_df['Weight'] = display_df['Weight'].round(6)
            display_df['Absolute Weight'] = display_df['Absolute Weight'].round(6)
            display_df['Impact'] = display_df['Weight'].apply(
                lambda x: "🟥 Increases Risk" if x > 0 else "🟩 Decreases Risk" if x < 0 else "⚪ Neutral"
            )

            # Reorder columns for better readability
            display_df = display_df[['Feature', 'Weight', 'Absolute Weight', 'Impact']]

            st.dataframe(display_df, use_container_width=True)

            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                positive_count = len(feature_dataframe[feature_dataframe['Weight'] > 0])
                st.metric("🟥 Risk Factors", positive_count)

            with col2:
                negative_count = len(feature_dataframe[feature_dataframe['Weight'] < 0])
                st.metric("🟩 Protective Factors", negative_count)

            with col3:
                top_feature = feature_dataframe.iloc[0]['Feature']
                top_weight = feature_dataframe.iloc[0]['Weight']
                st.metric("🏆 Most Important", top_feature)

            with col4:
                st.metric("📊 Total Features", len(feature_dataframe))


elif selected == "BMI by Race":
    st.title("BMI Visualization")
    df = pd.read_csv(data_path)
    race_cols = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
    available_race_cols = [col for col in race_cols if col in df.columns]


    def get_race(row):
        for col in available_race_cols:
            if row[col] == 1:
                return col.split(":")[1]
        return 'Invalid'


    if available_race_cols:
        df['race'] = df.apply(get_race, axis=1)
        df = df.drop(columns=available_race_cols)

    bmi_plotting_data_with_race = (
        df.groupby("year", group_keys=False).apply(lambda x: x.sample(min(len(x), 625), random_state=42)))
    bmi_plotting_data_with_race['year'] = bmi_plotting_data_with_race['year'].astype(int)
    bmi_plotting_data_with_race = bmi_plotting_data_with_race.sort_values(by='year')

    animated_scatter = px.scatter(bmi_plotting_data_with_race, x="age", y="bmi", color="race", animation_frame="year")
    animated_scatter.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
    animated_scatter.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2000
    st.plotly_chart(animated_scatter)