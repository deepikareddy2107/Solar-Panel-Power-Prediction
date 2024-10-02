import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv("solarpowergeneration.csv")

# Rename the columns
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

# Impute missing values (if any) with the mean of the column
for column in data.columns:
    if data[column].isnull().sum() > 0:
        mean_value = data[column].mean()
        data[column] = data[column].fillna(mean_value)

# Standardize the numerical features (if necessary)
#scaler = StandardScaler()
#numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
#scaled_data = scaler.fit_transform(data[numerical_columns])

# Create a DataFrame with the scaled data
#scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

# Splitting features and target variable
X = data.drop(columns=['power_generated'])
y = data['power_generated']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
xgb_model = XGBRegressor()

# Train the model
xgb_model.fit(X_train, y_train)

# Save the trained model to a file
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Load the model from file (for verification or future use)
# with open('xgb_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation Metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print evaluation results
print(f"XGBoost Evaluation:\nMSE: {mse_xgb}\nMAE: {mae_xgb}\nR²: {r2_xgb}\n")
