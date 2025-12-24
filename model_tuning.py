import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier # CRITICAL CHANGE
from titanic_pipeline import load_data, preprocess_and_feature_engineer
from scalers import fit_scaler, transform_data 

import warnings
warnings.filterwarnings('ignore')

def tune_hyperparameters():
    """Performs GridSearchCV to find the best hyperparameters for XGBoost."""
    
    combined_df, y_train, _ = load_data()
    if combined_df is None:
        return

    # **FIRST PASS: FIT SCALER and get processed data**
    processed_df, fitted_scaler = preprocess_and_feature_engineer(combined_df.copy(), fit_mode=True)
    
    train_len = len(y_train)
    X = processed_df[:train_len]
    
    # Optimized XGBoost Parameter Grid
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [2, 3, 4], 
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9]
    }

    # IMPORTANT: Use XGBClassifier
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    print("Starting GridSearchCV for XGBoost Hyperparameter Tuning...")
    grid_search.fit(X, y_train)

    print("\n--- Tuning Results ---\n")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_

if __name__ == '__main__':
    tune_hyperparameters()