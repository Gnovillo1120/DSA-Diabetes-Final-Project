import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
from plotly import graph_objects as go
from main import load_and_preprocess_data


with open("logistic_regression_model.pkl", "rb") as f:
    lr_model, feature_names, scaler_value = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model, _, scaler_value = pickle.load(f)

def scale(user, scaler_val):
    means, stds = scaler_val
    for idx, mean in means.items():
        std = stds[idx]
        if std > 0:
            user[:, idx] = (user[:, idx] - mean) / std
    return user

def result_card(value, title, non_diabetic = True, ):
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

st.sidebar.title("Diabetes Prediction Dashboard")
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Diabetes Risk Calculator", "Model Comparison", "Feature Importance", "BMI by Race"]
    )

if selected == "Diabetes Risk Calculator":
    #input feilds
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

    #stuff the button will do
    if st.button("Calculate Risk"):
        gender_correspondance = {"Female": 0, "Male": 1, "Other": 2}
        smoking_correspondance = {"Never": 0, "Former": 1, "Current": 2, "No Information": 3}
        user_info = [1, gender_correspondance[gender], age, hypertension, heart_disease, smoking_correspondance[smoking], bmi, h1ba1c, glucose, 0, 0, 0, 0, 0]

        formated_array = np.array(user_info).reshape(1, -1)
        formated_array = scale(formated_array, scaler_value)
        st.session_state["user_input"] = formated_array

        linear_regression_probability = lr_model.predict_proba(formated_array)[0]
        linear_regression_prediction = lr_model.predict(formated_array)[0]
        svm_prediction = svm_model.predict(formated_array)[0]

        st.subheader("Prediction Results: ")
        label_dictionary = {0: "Non-Diabetic", 1: "Diabetic"}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                result_card(
                    f"{label_dictionary[linear_regression_prediction]} ({linear_regression_probability:.2%})", "Logistic Regression", non_diabetic=(linear_regression_prediction == 0)), unsafe_allow_html=True)
        with col2:
            st.markdown(
                result_card(
                    f"{label_dictionary[svm_prediction]}", "Linear SVM", non_diabetic = (svm_prediction == 0)), unsafe_allow_html=True)

elif selected == "Model Comparison":
    st.title("Your Comparison against others")
    #df = pd.read_csv("C:/Users/gsnov/Downloads/diabetes_dataset.csv")
    X, y, feature_names = load_and_preprocess_data('C:/Users/gsnov/Downloads/diabetes_dataset.csv')
    X = scale(X, scaler_value)
    patients_lr_prediction = lr_model.predict_proba(X)
    if "user_input" in st.session_state:
        user_probability = lr_model.predict_proba(st.session_state["user_input"])[0]
        graph = px.histogram(patients_lr_prediction, nbins=30, title="Logistic Regression Predictions Distribution")
        graph.add_vline(x=float(user_probability), line_color="red", line_dash="dash", annotation_text = f"You: {user_probability:.2%}", annotation_position = "top right")
        st.plotly_chart(graph, use_container_width=True)
    else:
        st.warning("Please use Diabetes risk calculator first for the values")

elif selected == "Feature Importance":
    st.title("Feature Importance")
    weights = lr_model.weights[1:len(feature_names)+1]
    feature_dataframe = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights,
        "Absolute Weight": np.abs(weights)
    }).sort_values(by = "Absolute Weight", ascending=False)

    st.subheader("Features ranked by overall importance (+ and -)")
    chart = (alt.Chart(feature_dataframe.head(10)).mark_bar().encode
             (
                 x = alt.X("Feature", sort = None), y = alt.Y("Absolute Weight"), color = alt.condition(
                     alt.datum.Weight > 0, alt.value("green"), alt.value("red")
        )
            )
                )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(feature_dataframe)

elif selected == "BMI by Race":
    st.title("BMI Visualization")
    df = pd.read_csv("C:/Users/gsnov/Downloads/diabetes_dataset.csv")
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
    bmi_plotting_data_with_race = (df.groupby("year", group_keys=False).apply(lambda x: x.sample(min(len(x), 625), random_state=42)))

    
    bmi_plotting_data_with_race['year'] = bmi_plotting_data_with_race['year'].astype(int)
    bmi_plotting_data_with_race = bmi_plotting_data_with_race.sort_values(by='year')

    animated_scatter = px.scatter(bmi_plotting_data_with_race, x="age", y="bmi", color="race", animation_frame= "year")
    animated_scatter.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000  
    animated_scatter.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2000 
    st.plotly_chart(animated_scatter)



