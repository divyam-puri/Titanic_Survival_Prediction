# scalers.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERICAL_COLS = ['Age', 'Fare', 'FamilySize'] 

def fit_scaler(df):
    return None

def transform_data(df, scaler):
    return df