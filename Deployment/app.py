import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration (must be called only once and at the top)
st.set_page_config(page_title="Solar Power Generation Prediction", layout="wide")

# Load the trained model
try:
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the dataset to get feature columns (without the target column)
try:
    data = pd.read_csv("solarpowergeneration.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Rename columns as per the training dataset
new_column_names = {
    'distance-to-solar-noon': 'distance_to_solar_noon',
    'wind-direction': 'wind_direction',
    'wind-speed': 'wind_speed',
    'sky-cover': 'sky_cover',
    'average-wind-speed-(period)': 'average_wind_speed',
    'average-pressure-(period)': 'average_pressure',
    'power-generated': 'power_generated'
}
data = data.rename(columns=new_column_names)

# Select feature columns (exclude 'power_generated')
features = data.drop(columns=['power_generated']).columns

# Page Navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Main Page"
    st.session_state["prediction_made"] = False
    st.session_state["input_df"] = pd.DataFrame()

# Main Page: for Prediction
if st.session_state["page"] == "Main Page":
    st.title("🌞 Solar Power Generation Prediction 🌞")

    # Create input fields for each feature
    st.write("### Enter the values for the features to predict the power generated:")

    user_input = {}
    for feature in features:
        if feature in ['temperature', 'humidity']:
            user_input[feature] = st.slider(f"Enter {feature}:", min_value=0, max_value=100, value=0)
        else:
            user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0, format="%.2f")

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Store user input in session state
    st.session_state["input_df"] = input_df

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)
            predicted_value = prediction[0]

            if np.isclose(input_df.values, 0).all():
                predicted_value = 0.0  # Set prediction to 0 if all inputs are zero

            # Color the prediction output based on the value
            if predicted_value > 1000:
                prediction_color = 'green'
            elif predicted_value < 500:
                prediction_color = 'red'
            else:
                prediction_color = 'black'

            st.markdown(f"<h3 style='color:{prediction_color};'>Predicted Power Generated: {predicted_value:.2f} Watts (W)</h3>", unsafe_allow_html=True)

            # Set flag to indicate that a prediction has been made
            st.session_state["prediction_made"] = True

            # Set the page to "More Information" after prediction
            st.session_state["page"] = "More Information"
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Show "More Information" button only if prediction has been made
if st.session_state["prediction_made"]:
    st.write("### Want to know more?")
    st.write("Click the button below to see feature importance and visual comparisons:")
    if st.button("More Information"):
        st.session_state["page"] = "More Information"

# More Information Page: for Feature Importance and User Input vs Dataset Plots
if st.session_state["page"] == "More Information":
    st.title("🔍 More Information - Insights and Visuals")

    # Feature Importance
    st.write("### Feature Importance")

    importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Create a bar plot for feature importance with a light background
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax, palette='Blues_d')

    # Set plot title and labels
    ax.set_title('Feature Importance', fontsize=16, color='black')
    ax.set_xlabel('Importance', fontsize=12, color='black')
    ax.set_ylabel('Feature', fontsize=12, color='black')

    st.pyplot(fig)

    # Display the User Input vs Dataset Distribution plots
    st.write("### User Input vs Dataset Distribution")

    # Retrieve user input DataFrame from session state
    input_df = st.session_state.get("input_df", pd.DataFrame())

    # Loop through each feature to compare dataset distribution with user input
    for feature in features:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the dataset distribution with KDE
        sns.histplot(data[feature], kde=True, label=f"Dataset {feature}", ax=ax, color='lightblue')

        # Plot the user input as a vertical line for visibility
        if not input_df.empty:
            ax.axvline(x=input_df[feature].values[0], color='darkorange', linewidth=2, label=f"User Input {feature}")

        # Set legend and title
        ax.legend(loc='upper right')
        ax.set_title(f"{feature} Distribution")

        # Display the plot in Streamlit
        st.pyplot(fig)

    # Back button to return to the Main Page
    if st.button("Back to Main Page"):
        st.session_state["page"] = "Main Page"
        st.session_state["prediction_made"] = True
