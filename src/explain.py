import shap
import joblib
import pandas as pd

def get_churn_explanation(raw_input_df):
    """Calculates granular SHAP values for single inference calls."""
    model = joblib.load('models/churn_xgb_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Process single sample
    transformed_data = preprocessor.transform(raw_input_df)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_data)
    
    # Map back to structural dataframe names
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Map top 3 dominant risk factors driving the specific prediction outcome
    contributions = shap_df.iloc[0].to_dict()
    sorted_drivers = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_drivers[:3]
