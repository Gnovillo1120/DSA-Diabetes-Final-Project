import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as pg

def load_and_preprocess_data(filename):
    df = pd.read_csv("C:/Users/gsnov/Downloads/diabetes_dataset.csv")

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    df = df.copy()

    numeric_columns = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'year']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str)
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2}).fillna(2)

    if 'smoking_history' in df.columns:
        df['smoking_history'] = df['smoking_history'].astype(str)
        df['smoking_history'] = df['smoking_history'].map({
            'never': 0, 'former': 1, 'current': 2, 'No Info': 3, 'no info': 3
        }).fillna(3)

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


    feature_columns = [
        'year', 'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
        'bmi', 'hbA1c_level', 'blood_glucose_level','race'
    ]

    feature_columns = [col for col in feature_columns if col in df.columns]

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')  
        df = df.dropna(subset=['year'])                         
        df['year'] = df['year'].astype(int)
        df = df.sort_values(by='year')           

    df = df.fillna(0)


    df = df.fillna(0)

    y = df['diabetes'].values.astype(np.float64)

    X = df[feature_columns] 

    X = np.column_stack([np.ones(X.shape[0]), X])


    return X, y, feature_columns

        


x, y, feature_columns = load_and_preprocess_data("Obesity_data")
bmi_plotting_data = pd.DataFrame(x[0:2500,[3,7]],columns=["age","bmi"]) 
np.random.seed(87912)
sample_size = min(2500, x.shape[0]) 
randoms = np.random.choice(x.shape[0], size=sample_size, replace=False)
bmi_plotting_data_year = pd.DataFrame(x[randoms][:,[1,3,7]],columns=["year","age","bmi"]) 
bmi_plotting_data_year['year'] = bmi_plotting_data_year['year'].astype(int)
bmi_plotting_data_year = bmi_plotting_data_year.sort_values(by='year')

bmi_plotting_data_with_race = pd.DataFrame(x[randoms][:,[1,3,7, 10]],columns=["year","age","bmi", "race"])
bmi_plotting_data_with_race['year'] = bmi_plotting_data_with_race['year'].astype(int)
bmi_plotting_data_with_race = bmi_plotting_data_with_race.sort_values(by='year')

bmi_plotting_data_blood_things = pd.DataFrame(x[randoms][:,[7,8, 9]],columns=["bmi", "hbA1c", "blood_glucose_level"])



st.title("Graphs Test")
st.text("Trying out some Streamlit Graph Applications")
st.header("Scatter Chart Representing Age to BMI Data from original dataset.")

st.scatter_chart(data= bmi_plotting_data, x="age", y= "bmi", x_label="Age", y_label="Body Mass Index",color=["#59d96c"], size=20, width=None, height=None, use_container_width=True) 


st.subheader("Some notes:")
st.text("I originally started by using all 100000 datapoints, dont do that. very laggy, lowered it down to 10 percent for testing." \
" This One is using The auto width and Auto Height idk how I like it yet." \
" Okay, I had to lower it again, 1000 it is, we should be careful with how many datapoints we do unless this is just an issue with my local machine." \
" You can actually zoom in and pan with these graphs, pretty cool. Bless Anvi for writing the pandas code 🙏")

st.header("Animations??")

st.text("Just wasted like an hour trying to use the native streamlit things for animations" \
" Turns out Streamlit does not have native support for animations :( , going to try Plotly.")

animated_maybe = px.histogram(bmi_plotting_data_year, x="bmi", nbins=30, color="year", animation_frame="year", range_x=[bmi_plotting_data_year['bmi'].min(), bmi_plotting_data_year['bmi'].max()])
st.plotly_chart(animated_maybe)

st.text("This works, its a little clunky since its a histogram and its also very zoomed in which might just" \
" be a data selection thing, so no biggie for now. I modified the load data function a little" \
" to include year. Will try Scatter Plot now, I think it will look better")

animated_scatter = px.scatter(bmi_plotting_data_with_race, x="age", y="bmi", color="race", animation_frame= "year")

animated_scatter.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000  
animated_scatter.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2000 

st.plotly_chart(animated_scatter)

st.text("Alright these look better, I made the race parameter in the load function just a string column" \
" This represents the age, bmi, year, and the race of the plot_points." \
"We can probably play around with toggling around some of these values for the specific person")

st.header("3D Graphs")

three_dimensional_chart = px.scatter_3d(bmi_plotting_data_blood_things, x="hbA1c", y="blood_glucose_level", z="bmi")

three_dimensional_chart.update_layout(title = dict(text="Temp Temp Temp"))

st.plotly_chart(three_dimensional_chart)

st.text("This honestly looks pretty wonky due to its trying to auto corrects its orientation and the " \
" way it set of the number ranges. Ima fix that Tommorow, im tired.")