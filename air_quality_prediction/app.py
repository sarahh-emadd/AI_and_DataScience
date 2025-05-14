from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model_path = 'models/best_model.pkl'  # Update path as needed
scaler_path = 'models/scaler.pkl'      # Update path as needed

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Check if model exists, if not we'll handle that in the routes
model_exists = os.path.exists(model_path)
scaler_exists = os.path.exists(scaler_path)

# Features that the model expects (update based on your actual model)
FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
           'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket']

# AQI bucket mapping (for reference)
AQI_BUCKETS = {
    'Good': 0,
    'Satisfactory': 1,
    'Moderate': 2,
    'Poor': 3,
    'Very Poor': 4,
    'Severe': 5
}

@app.route('/')
def home():
    return render_template('index.html', model_exists=model_exists)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_exists:
        return jsonify({'error': 'Model not found. Please upload a trained model first.'})
    
    try:
        # Get input data from form or JSON request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Create DataFrame from input
        input_data = {}
        for feature in FEATURES:
            if feature == 'AQI_Bucket':
                # Convert AQI_Bucket string to numeric value
                bucket_name = data.get(feature, 'Moderate')  # Default to Moderate if not provided
                input_data[feature] = AQI_BUCKETS.get(bucket_name, 2)  # Default to 2 (Moderate) if not found
            else:
                # Convert to float, default to 0 if not provided
                input_data[feature] = float(data.get(feature, 0))
        
        input_df = pd.DataFrame([input_data])
        
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Preprocess the input data
        X = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X)
        
        # Return prediction result
        return jsonify({
            'prediction': prediction.tolist(),
            'input_data': input_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files or 'scaler_file' not in request.files:
        return jsonify({'error': 'Model and scaler files are required'})
    
    model_file = request.files['model_file']
    scaler_file = request.files['scaler_file']
    
    if model_file.filename == '' or scaler_file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        model_file.save(model_path)
        scaler_file.save(scaler_path)
        
        # Update model_exists flag
        global model_exists, scaler_exists
        model_exists = True
        scaler_exists = True
        
        return jsonify({'message': 'Model and scaler uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)