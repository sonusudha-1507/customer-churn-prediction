import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def generate_mock_data(num_samples=5000):
    """Generates a realistic customer behavior dataset for churn prediction."""
    np.random.seed(42)
    
    data = {
        'CustomerID': [f"CUST-{i:05d}" for i in range(num_samples)],
        'Age': np.random.randint(18, 70, size=num_samples),
        'Tenure': np.random.randint(1, 72, size=num_samples),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, size=num_samples),
        'TotalCharges': np.zeros(num_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=num_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], size=num_samples, p=[0.4, 0.6]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=num_samples)
    }
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges'] * np.random.uniform(0.9, 1.1, size=num_samples)
    
    # Injecting real-world internal logic so the models can actually learn patterns
    churn_prob = (
        (df['Contract'] == 'Month-to-month').astype(int) * 0.4 +
        (df['TechSupport'] == 'No').astype(int) * 0.2 +
        (df['Tenure'] < 12).astype(int) * 0.3 +
        (df['MonthlyCharges'] > 80).astype(int) * 0.1
    )
    df['Churn'] = (churn_prob + np.random.uniform(0, 0.3, size=num_samples) > 0.6).astype(int)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customer_churn_data.csv', index=False)
    return df

def preprocess_and_split():
    """Loads, cleans, encodes, scales, and balances the dataset."""
    if not os.path.exists('data/customer_churn_data.csv'):
        df = generate_mock_data()
    else:
        df = pd.read_csv('data/customer_churn_data.csv')
        
    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn']
    
    numeric_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'PaymentMethod', 'TechSupport', 'PaperlessBilling']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_acc = preprocessor.fit_transform(X_train)
    X_test_acc = preprocessor.transform(X_test)
    
    # Extract structural feature names post-encoding to pass seamlessly into SHAP models
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + encoded_cat_features
    
    # Handle class imbalance natively via SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_acc, y_train)
    
    return X_train_res, X_test_acc, y_train_res, y_test, preprocessor, feature_names
