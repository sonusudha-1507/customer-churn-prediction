import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pipeline import preprocess_and_split

def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    print(f"\n===== {name} Performance =====")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-Score:  {f1_score(y_test, preds):.4f}")

def main():
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_and_split()
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # 3. XGBoost Production Model
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    # Export artifacts for production runtime deployment
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb, 'models/churn_xgb_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("\n[INFO] Best models and transformers successfully persisted to disk.")

if __name__ == '__main__':
    import os
    main()
