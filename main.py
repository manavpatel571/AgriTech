# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the trained models and encoders
catboost_model = joblib.load('catboost_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
state_encoder = joblib.load('State_encoder.pkl')
crop_type_encoder = joblib.load('Crop_Type_encoder.pkl')
soil_type_encoder = joblib.load('Soil_Type_encoder.pkl')

# Get the available options for categorical variables
states = state_encoder.classes_
crop_types = crop_type_encoder.classes_
soil_types = soil_type_encoder.classes_

def feature_engineering(df):
    processed = df.copy()
    
    numerical_cols = ['Year', 'Rainfall', 'Irrigation_Area']
    
    # 1. Basic interaction features
    processed['Year_Rainfall_interaction'] = processed['Year'] * processed['Rainfall']
    processed['Year_Irrigation_Area_interaction'] = processed['Year'] * processed['Irrigation_Area']
    processed['Rainfall_Irrigation_Area_interaction'] = processed['Rainfall'] * processed['Irrigation_Area']
    
    # 2. Ratio features
    processed['Year_ratio_to_rainfall'] = processed['Year'] / (processed['Rainfall'] + 1e-5)
    processed['Rainfall_ratio_to_rainfall'] = processed['Rainfall'] / (processed['Rainfall'] + 1e-5)
    
    # 3. Polynomial interaction terms
    processed['Year Rainfall'] = processed['Year'] * processed['Rainfall']
    processed['Year Irrigation_Area'] = processed['Year'] * processed['Irrigation_Area']
    processed['Rainfall Irrigation_Area'] = processed['Rainfall'] * processed['Irrigation_Area']
    processed['Year Rainfall Irrigation_Area'] = processed['Year'] * processed['Rainfall'] * processed['Irrigation_Area']
    
    # 4. Log transforms
    processed['Rainfall_log'] = np.log1p(processed['Rainfall'])
    processed['Irrigation_Area_log'] = np.log1p(processed['Irrigation_Area'])
    
    # 5. Aggregate features for State
    for col in numerical_cols:
        # Using mean values since we don't have access to the full dataset
        processed[f'State_{col}_mean'] = processed['State']  # placeholder
        processed[f'State_{col}_std'] = processed['State']   # placeholder
        processed[f'State_{col}_max'] = processed['State']   # placeholder
        processed[f'State_{col}_min'] = processed['State']   # placeholder
    
    # 6. Aggregate features for Crop_Type
    for col in numerical_cols:
        # Using mean values since we don't have access to the full dataset
        processed[f'Crop_Type_{col}_mean'] = processed['Crop_Type']  # placeholder
        processed[f'Crop_Type_{col}_std'] = processed['Crop_Type']   # placeholder
        processed[f'Crop_Type_{col}_max'] = processed['Crop_Type']   # placeholder
        processed[f'Crop_Type_{col}_min'] = processed['Crop_Type']   # placeholder
    
    # Ensure column order matches training data
    expected_columns = ['Year', 'State', 'Crop_Type', 'Rainfall', 'Soil_Type', 'Irrigation_Area',
                       'Year_Rainfall_interaction', 'Year_Irrigation_Area_interaction', 'Year_ratio_to_rainfall',
                       'Rainfall_Irrigation_Area_interaction', 'Rainfall_ratio_to_rainfall', 'Year Rainfall',
                       'Year Irrigation_Area', 'Rainfall Irrigation_Area', 'Year Rainfall Irrigation_Area',
                       'Rainfall_log', 'Irrigation_Area_log', 'State_Year_mean', 'State_Year_std',
                       'State_Year_max', 'State_Year_min', 'State_Rainfall_mean', 'State_Rainfall_std',
                       'State_Rainfall_max', 'State_Rainfall_min', 'State_Irrigation_Area_mean',
                       'State_Irrigation_Area_std', 'State_Irrigation_Area_max', 'State_Irrigation_Area_min',
                       'Crop_Type_Year_mean', 'Crop_Type_Year_std', 'Crop_Type_Year_max', 'Crop_Type_Year_min',
                       'Crop_Type_Rainfall_mean', 'Crop_Type_Rainfall_std', 'Crop_Type_Rainfall_max',
                       'Crop_Type_Rainfall_min', 'Crop_Type_Irrigation_Area_mean', 'Crop_Type_Irrigation_Area_std',
                       'Crop_Type_Irrigation_Area_max', 'Crop_Type_Irrigation_Area_min']
    
    return processed[expected_columns]

@app.route('/')
def home():
    return render_template('index.html', 
                         states=states,
                         crop_types=crop_types,
                         soil_types=soil_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        input_data = {
            'State': request.form['state'],
            'Crop_Type': request.form['crop_type'],
            'Soil_Type': request.form['soil_type'],
            'Year': float(request.form['year']),
            'Rainfall': float(request.form['rainfall']),
            'Irrigation_Area': float(request.form['irrigation_area'])
        }

        # Validate categorical inputs
        if input_data['State'] not in states:
            raise ValueError(f"Invalid state. Please select from available options.")
        if input_data['Crop_Type'] not in crop_types:
            raise ValueError(f"Invalid crop type. Please select from available options.")
        if input_data['Soil_Type'] not in soil_types:
            raise ValueError(f"Invalid soil type. Please select from available options.")

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        input_df['State'] = state_encoder.transform([input_data['State']])[0]
        input_df['Crop_Type'] = crop_type_encoder.transform([input_data['Crop_Type']])[0]
        input_df['Soil_Type'] = soil_type_encoder.transform([input_data['Soil_Type']])[0]

        # Apply feature engineering
        input_processed = feature_engineering(input_df)

        # Make predictions
        catboost_pred = catboost_model.predict(input_processed)[0]
        xgboost_pred = xgboost_model.predict(input_processed)[0]
        
        # Ensemble prediction
        final_prediction = 0.6 * catboost_pred + 0.4 * xgboost_pred

        return render_template('result.html', 
                             prediction=round(final_prediction, 2),
                             input_data=input_data)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)