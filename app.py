import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px

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

        linear_regression_probability = lr_model.predict_proba(formated_array)[0]
        linear_regression_prediction = lr_model.predict(formated_array)[0]
        svm_prediction = svm_model.predict(formated_array)[0]

        st.session_state['user_input'] = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking": smoking,
            "bmi": bmi,
            "h1ba1c": h1ba1c,
            "glucose": glucose
        }

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
    if "user_input" in st.session_state:
        user_data = pd.DataFrame([st.session_state["user_input"]])
        user_age = user_data["age"].iloc[0]
        user_bmi = user_data["bmi"].iloc[0]
        user_h1ba1c = user_data["h1ba1c"].iloc[0]
        user_glucose = user_data["glucose"].iloc[0]

        comparison_data = pd.read_csv("C:/Users/Jorge/Downloads/diabetes_dataset.csv")


        similar_datapoints = comparison_data[
        (comparison_data["age"].between(user_age - 8, user_age + 8)) &
        (comparison_data["bmi"].between(user_bmi - 6, user_bmi + 6)) 
        
        
    ]
        
        similar_datapoints["age_bin"] = pd.cut(similar_datapoints["age"], bins=13)
        similar_datapoints["bmi_bin"] = pd.cut(similar_datapoints["bmi"], bins=13)

        heatmap_data = (
            similar_datapoints.groupby(["age_bin", "bmi_bin"])["diabetes"]
            .mean()
            .reset_index()
        )

        counts = similar_datapoints.groupby(["age_bin", "bmi_bin"]).size().reset_index(name="count")
        heatmap_data = heatmap_data.merge(counts, on=["age_bin", "bmi_bin"])


        min_val = heatmap_data["diabetes"].min()
        max_val = heatmap_data["diabetes"].max()
        heatmap_data["diabetes_norm"] = (heatmap_data["diabetes"] - min_val) / (max_val - min_val)

        pivot_table = heatmap_data.pivot(index="age_bin", columns="bmi_bin", values="diabetes_norm")

        pivot_table = pivot_table.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)

        pivot_smoothed = pivot_table.rolling(2, axis=0, min_periods=1).mean().rolling(2, axis=1, min_periods=1).mean()
        pivot_smoothed = pivot_smoothed.fillna(0)

        similar_datapoints["age_bin"] = similar_datapoints["age_bin"].astype(str)
        similar_datapoints["bmi_bin"] = similar_datapoints["bmi_bin"].astype(str)

        pivot_smoothed = pivot_smoothed.iloc[::-1]
        comparison_chart_1 = px.imshow(
    pivot_smoothed.values,
    x=pivot_smoothed.index.astype(str),
    y=pivot_smoothed.columns.astype(str),
    color_continuous_scale="RdBu",
    zmin=0,  
    zmax=1, 
    labels=dict(x="Age", y="BMI", color="Risk Compared to Similar Groups", ticktext=["Lowest Risk", "Highest Risk"]),
    title="Normalized Diabetes Risk Heatmap"
)
                
        st.plotly_chart(comparison_chart_1)


    

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

elif selected == "BMI by Race":
    st.title("BMI Visualization")
    df = pd.read_csv("C:/Users/Jorge/Downloads/diabetes_dataset.csv")
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



































