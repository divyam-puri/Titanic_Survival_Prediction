import streamlit as st
import pandas as pd
import numpy as np
from titanic_pipeline import preprocess_and_feature_engineer, train_model, load_data
from scalers import fit_scaler, transform_data 

# --- Setup and Model Loading ---

@st.cache_resource
def load_and_train_model(): 
    """Loads data, preprocesses it, and trains the final Random Forest model."""
    
    combined_df, y_train, _ = load_data()
    
    if combined_df is None:
        raise FileNotFoundError("Could not load train.csv or test.csv. Please check file paths.")
        
    processed_df, fitted_scaler = preprocess_and_feature_engineer(combined_df.copy(), fit_mode=True)
    
    train_len = len(y_train)
    X = processed_df[:train_len]
    
    final_model = train_model(X, y_train)
    
    return final_model, processed_df.columns.tolist(), fitted_scaler

try:
    model, feature_columns, fitted_scaler = load_and_train_model() 
except Exception as e:
    st.error(f"Failed to initialize model. Error: {e}")
    st.stop()


# --- Feature Alignment Helper Function (CRITICAL FIX) ---
def align_features(processed_input_df, feature_columns):
    """Ensures the input DataFrame has the exact same columns as the model's training data."""
    X_input = processed_input_df.copy()

    clean_feature_columns = [col for col in feature_columns if col not in ['Survived', 'PassengerId']]
    
    missing_cols = set(clean_feature_columns) - set(X_input.columns)
    for c in missing_cols:
        X_input[c] = 0
    
    X_input = X_input.drop(columns=X_input.columns.difference(clean_feature_columns), errors='ignore')
    X_input = X_input[clean_feature_columns] 
    
    return X_input


# --- Streamlit Application Layout ---

st.set_page_config(page_title="Titanic Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor (Random Forest Model)") 
st.markdown("---")


# --- Input Form --- 
with st.form("prediction_form"):
    st.header("Passenger Details")
    
    col_pclass, col_sex, col_age = st.columns(3)
    pclass = col_pclass.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2, help="1st=Upper, 3rd=Lower")
    sex = col_sex.radio("Sex", ["male", "female"], horizontal=True)
    age = col_age.slider("Age", 0, 80, 35) 
    
    col_sibsp, col_parch, col_fare = st.columns(3)
    sibsp = col_sibsp.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = col_parch.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = col_fare.number_input("Fare Paid ($)", 0.0, 550.0, 8.00)
    
    col_emb, col_deck, col_name = st.columns(3)
    embarked = col_emb.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)
    deck_input = col_deck.selectbox("Cabin Deck (First Letter)", ['M', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0, help="'M' for Unknown/Missing")
    name_input = col_name.text_input("Full Name (Needed for Title)", "Smith, Mr. John")

    submitted = st.form_submit_button("Predict Survival")


# --- Prediction Logic ---

if submitted:
    
    # 1. Prepare Input DataFrame
    input_data = {
        'PassengerId': [9999], 'Pclass': [pclass], 'Name': [name_input], 'Sex': [sex],
        'Age': [float(age)], 'SibSp': [sibsp], 'Parch': [parch], 'Ticket': ['STUB'], 
        'Fare': [fare], 'Cabin': [deck_input + '1'], 'Embarked': [embarked]
    }
    input_df = pd.DataFrame(input_data)
    
    # 2. Preprocess the input data (Includes Feature Engineering)
    processed_input_df, _ = preprocess_and_feature_engineer(input_df.copy(), scaler=fitted_scaler) 
    
    # 3. ALIGN AND ORDER FEATURES
    X_input_aligned = align_features(processed_input_df, feature_columns)

    # 4. Make Prediction
    try:
        prediction_proba = model.predict_proba(X_input_aligned)[0]
    except Exception as e:
        st.error("Error during prediction. See exception details below.")
        st.exception(e)
        st.stop()
    
    # 5. Display Result with ADJUSTED THRESHOLD (>0.65)
    st.markdown("---")
    
    # CRUCIAL FIX: Survival threshold is now > 0.65 (65%)
    if prediction_proba[1] > 0.65:
        st.balloons()
        st.success(f"## ✅ Survival Predicted! (Likelihood: {prediction_proba[1]*100:.2f}%)")
    else:
        # Show Non-Survival (Likelihood of Non-Survival)
        st.error(f"## ❌ Survival Not Predicted. (Likelihood: {(1 - prediction_proba[1])*100:.2f}%)")