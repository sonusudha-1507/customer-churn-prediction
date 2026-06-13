from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from src.explain import get_churn_explanation

app = Flask(__name__)

model = joblib.load('models/churn_xgb_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map web layout client state payload explicitly
        form_data = {
            'Age': [float(request.form['age'])],
            'Tenure': [float(request.form['tenure'])],
            'MonthlyCharges': [float(request.form['monthly_charges'])],
            'TotalCharges': [float(request.form['tenure']) * float(request.form['monthly_charges'])],
            'Contract': [request.form['contract']],
            'PaymentMethod': [request.form['payment_method']],
            'TechSupport': [request.form['tech_support']],
            'PaperlessBilling': [request.form['paperless_billing']]
        }
        
        df_input = pd.DataFrame(form_data)
        
        # Matrix Feature Transformations
        processed_input = preprocessor.transform(df_input)
        probability = model.predict_proba(processed_input)[0][1]
        prediction = int(model.predict(processed_input)[0])
        
        # Fetch analytical explanations
        top_drivers = get_churn_explanation(df_input)
        drivers_clean = [{"feature": f.replace('_Yes', '').replace('cat__', ''), "impact": round(v, 4)} for f, v in top_drivers]
        
        return jsonify({
            'success': True,
            'churn_risk': "High" if probability > 0.5 else "Low",
            'probability': round(float(probability) * 100, 2),
            'drivers': drivers_clean
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
