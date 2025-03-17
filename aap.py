import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

 
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
 
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']
numerical_cols = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure_group']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {col: request.form[col] for col in categorical_cols + numerical_cols}

        # Convert numerical values to float
        for col in numerical_cols:
            data[col] = float(data[col])

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

         
        input_transformed = preprocessor.transform(input_df)

        
        prediction = model.predict(input_transformed)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
