# 🚢 Titanic Survival Predictor (Machine Learning Project)

This project is an end-to-end Machine Learning application that predicts the survival probability of passengers aboard the Titanic using a **Random Forest classifier**. It demonstrates the complete ML workflow along with deployment via a **Streamlit web interface**.

---

## 📌 Features
- Data cleaning and preprocessing  
- Advanced feature engineering (age & fare binning, family size, interaction features)  
- Random Forest classification model  
- Probability-based survival prediction  
- Interactive Streamlit web application  
- Modular and scalable project structure  

---

## 🛠 Tech Stack
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure

```text
Titanic_Survival_Prediction/
│
├── train.csv               # Training dataset
├── test.csv                # Test dataset
│
├── titanic_pipeline.py     # Data preprocessing & model training
├── titanic_app.py          # Streamlit web application
├── model_tuning.py         # Hyperparameter tuning
├── scalers.py              # Scaling utilities
├── submission.py           # Prediction file generation
│
└── __pycache__/            # Cached Python files
