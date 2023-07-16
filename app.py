import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.special import boxcox1p
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')


# Load the trained model from GitHub
url = 'final_model.pkl'  # Replace with your GitHub URL
model = pd.read_pickle(url)

# Define the transformation functions
def transform_to_normal(df):
    # Transform humidity using Box-Cox transformation
    # df['humidity'], _ = stats.boxcox(df['humidity'] + 1)

    # Transform lighting using log transformation
    df['lighting'] = np.log1p(df['lighting'])

    # Transform noise_level using Box-Cox transformation
    # df['noise_level'], _ = stats.boxcox(df['noise_level'] + 1)

    # Transform object_movements using Johnson SU transformation
    transformed_object_movements = stats.johnsonsu.fit(df['object_movements'])
    df['object_movements'] = stats.johnsonsu(*transformed_object_movements).cdf(df['object_movements'])

    return df

def impute_null_values(df):
    # Impute temperature using mean
    df['temperature'].fillna(df['temperature'].median(), inplace=True)

    # Impute building_type using mode
    df['building_type'].fillna(df['building_type'].mode()[0], inplace=True)

    # Impute window_type using mode
    df['window_type'].fillna(df['window_type'].mode()[0], inplace=True)

    return df

# Define the Streamlit app
def main():
        # Add background image to title and sidebar
    st.markdown(
        """
        <style>
        .main-title {
            background-image: url('https://github.com/gullayeshwantkumarruler/ind_env/blob/main/image_indoor.jpg');
            background-size: cover;
            padding: 2rem;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Indoor Healthy Environment Predictor")
    st.sidebar.title("Input Features")
    st.sidebar.markdown("Enter the values for the following features:")

    # Get user inputs for features
    air_quality = st.sidebar.number_input("Air Quality", min_value=0.0, max_value=5000.0, step=0.1, value=20.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=18.0, max_value=25.0, step=0.1, value=20.0)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=40.0, max_value=60.0, step=0.1, value=50.0)
    lighting = st.sidebar.number_input("Lighting Intensity", min_value=0, max_value=1000, step=1, value=500)
    noise_level = st.sidebar.number_input("Noise Level (dB)", min_value=30, max_value=70, step=1, value=50)
    green_spaces = st.sidebar.number_input("Access to Green Spaces (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    object_movements = st.sidebar.number_input("Number of Object Movements", min_value=0, max_value=10, step=1, value=5)
    pollution_controls = st.sidebar.selectbox("Pollution Controls", ['Yes', 'No'])
    elderly_health = st.sidebar.selectbox("Elderly Health", ['Excellent', 'Good', 'Fair', 'Poor'])
    wellbeing = st.sidebar.selectbox("Wellbeing", ['High', 'Medium', 'Low'])
    building_type = st.sidebar.selectbox("Building Type", ['House', 'Apartment', 'Office'])
    floor_level = st.sidebar.selectbox("Floor Level", ['Ground', '1st', '2nd', '3rd', '4th'])
    window_type = st.sidebar.selectbox("Window Type", ['Single', 'Double', 'Triple'])
    ventilation_type = st.sidebar.selectbox("Ventilation Type", ['Natural', 'Mechanical'])
    location = st.sidebar.selectbox("Location", ['Urban', 'Suburban', 'Rural'])

    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'air_quality':[air_quality],
        'temperature': [temperature],
        'humidity': [humidity],
        'lighting': [lighting],
        'noise_level': [noise_level],
        'green_spaces': [green_spaces],
        'object_movements': [object_movements],
        'pollution_controls': [pollution_controls],
        'elderly_health': [elderly_health],
        'wellbeing': [wellbeing],
        'building_type': [building_type],
        'floor_level': [floor_level],
        'window_type': [window_type],
        'ventilation_type': [ventilation_type],
        'location': [location]
    })

    # Apply the transformations to the input data
    input_data = transform_to_normal(input_data)
    input_data = impute_null_values(input_data)
    

    # Perform label encoding on the pollution_controls column
    label_encoder = LabelEncoder()
    input_data['pollution_controls'] = label_encoder.fit_transform(input_data['pollution_controls'])
    
    # Perform label encoding on the elderly_health column
    input_data['elderly_health'] = label_encoder.fit_transform(input_data['elderly_health'])
    
    # Perform label encoding on the wellbeing column
    input_data['wellbeing'] = label_encoder.fit_transform(input_data['wellbeing'])
    
    # Perform label encoding on the building_type column
    input_data['building_type'] = label_encoder.fit_transform(input_data['building_type'])
    
    # Perform label encoding on the floor_level column
    input_data['floor_level'] = label_encoder.fit_transform(input_data['floor_level'])
    
    # Perform label encoding on the window_type column
    input_data['window_type'] = label_encoder.fit_transform(input_data['window_type'])
    
    # Perform label encoding on the ventilation_type column
    input_data['ventilation_type'] = label_encoder.fit_transform(input_data['ventilation_type'])
    
    # Perform label encoding on the location column
    input_data['location'] = label_encoder.fit_transform(input_data['location'])
    # Perform one-hot encoding on the categorical features
    # input_data = pd.get_dummies(input_data, drop_first=True)
    print(input_data.head(1))
    # Make predictions using the trained model
    prediction = np.argmax(model.predict(input_data))

    # Display the prediction
    st.header("Prediction")
    if prediction == 1:
        st.success("The indoor environment is predicted to be healthy.")
    else:
        st.error("The indoor environment is predicted to be unhealthy.")

if __name__ == '__main__':
    main()
