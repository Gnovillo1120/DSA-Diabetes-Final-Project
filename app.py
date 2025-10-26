import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import altair as alt

with open("logistic_regression_model.pkl", "rb") as f:
    lr_model, feature_names = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model, _ = pickle.load(f)

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
        options=["Diabetes Risk Calculator", "Model Comparison", "Feature Importance"]
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
    st.title("Model Comparison")
    st.write("Need to add a comparison tableee!!!!!!!!")


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






































