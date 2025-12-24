import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from scalers import fit_scaler, transform_data 
import warnings

warnings.filterwarnings('ignore') 

# NOTE: The circular import line has been removed.

def load_data(train_path='train.csv', test_path='test.csv'):
    """Loads and combines the Titanic train and test datasets."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Please ensure they are in the working directory.")
        return None, None, None
    
    test_passenger_ids = test_df['PassengerId']
    y_train = train_df['Survived']
    combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)
    
    return combined_df, y_train, test_passenger_ids

def preprocess_and_feature_engineer(df, scaler=None, fit_mode=False):
    """Performs cleaning, advanced feature engineering (binning, interactions), and prepares data."""
    
    # 1. Imputation
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # 2. Title Extraction & Age Imputation Proxy
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Age'].fillna(df['Age'].mean(), inplace=True) 
    
    # 3. CRITICAL: Age Binning
    df.loc[ df['Age'] <= 16, 'Age_Bin'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age_Bin'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age_Bin'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age_Bin'] = 3
    df.loc[ df['Age'] > 64, 'Age_Bin'] = 4
    df['Age_Bin'] = df['Age_Bin'].astype(int)

    # 4. CRITICAL: Fare Binning
    df.loc[ df['Fare'] <= 7.91, 'Fare_Bin'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare_Bin'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare_Bin'] = 2
    df.loc[ df['Fare'] > 31, 'Fare_Bin'] = 3
    df['Fare_Bin'] = df['Fare_Bin'].astype(int)

    # 5. Family Features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 6. Deck Feature
    df['Deck'] = df['Cabin'].str.extract(r'([A-Z])', expand=False).fillna('M')

    # 7. CRITICAL: Interaction Features
    df['Fare_Pclass_Inter'] = df['Fare_Bin'] * df['Pclass']
    df['Age_Class_Inter'] = df['Age_Bin'] * df['Pclass']
    
    # 8. STRUCTURAL INTERVENTION: High Risk Male Feature
    df['High_Risk_Male'] = 0
    df.loc[(df['Sex'] == 'male') & (df['Age_Bin'] >= 1) & (df['Pclass'].isin([2, 3])), 'High_Risk_Male'] = 1
    
    # 9. Drop and Clean up
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId', 'Age', 'Fare'], axis=1, inplace=True)

    # 10. One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Pclass', 'Deck', 'Age_Bin', 'Fare_Bin'], drop_first=True)
    
    return df, None 

def train_model(X_train, y_train, params=None):
    """Trains the Random Forest model."""
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_leaf': 4,
            'random_state': 42
        }
        
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    combined_df, y_train, test_ids = load_data()
    
    if combined_df is not None:
        processed_df, fitted_scaler = preprocess_and_feature_engineer(combined_df.copy(), fit_mode=True)
        
        train_len = len(y_train)
        X = processed_df[:train_len]
        X_test = processed_df[train_len:]
        
        # Validation
        X_train, X_val, y_train_split, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train_split)
        y_pred = model.predict(X_val)
        print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")

        # Final Training on full data
        final_model = train_model(X, y_train)
        predictions = final_model.predict(X_test)

        # Output for submission.py
        np.savetxt('final_predictions.txt', predictions, fmt='%d')
        pd.DataFrame({'PassengerId': test_ids}).to_csv('test_ids.csv', index=False)
        print("\nPipeline finished. Intermediate prediction files created.")