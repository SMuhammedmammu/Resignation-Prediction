import streamlit as st
import pickle
import numpy as np

# Load pickle files
with open('grid_search_rf.pkl', 'rb') as file:
    model = pickle.load(file)

with open('ordinal_encoder.pkl', 'rb') as file:
    ordinal_encoder = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('freq_encoding.pkl', 'rb') as file:
    freq_encoding = pickle.load(file)

with open('minmax_scaler.pkl', 'rb') as file:
    minmax_scaler = pickle.load(file)

# Dropdown options
gender_options = ['Male', 'Female', 'Others']
education_options = ordinal_encoder.categories_[0]
job_title_options = ordinal_encoder.categories_[1]
department_options = list(freq_encoding.keys())

# Streamlit app
st.title("Employee Resignation Prediction")

# Input form
st.header("Enter Employee Details")
with st.form("input_form"):
    # Dropdown inputs
    gender = st.selectbox("Gender", gender_options)
    education_level = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_title_options)
    department = st.selectbox("Department", department_options)

    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
    years_at_company = st.number_input("Years at Company", min_value=0, step=1, value=5)
    performance_score = st.number_input("Performance Score", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
    monthly_salary = st.number_input("Monthly Salary", min_value=0, step=500, value=5000)
    work_hours_per_week = st.number_input("Work Hours Per Week", min_value=0, max_value=168, step=1, value=40)
    projects_handled = st.number_input("Projects Handled", min_value=0, step=1, value=3)
    overtime_hours = st.number_input("Overtime Hours", min_value=0, step=1, value=5)
    sick_days = st.number_input("Sick Days", min_value=0, step=1, value=2)
    remote_work_frequency = st.number_input("Remote Work Frequency", min_value=0, max_value=7, step=1, value=2)
    team_size = st.number_input("Team Size", min_value=1, step=1, value=10)
    training_hours = st.number_input("Training Hours", min_value=0, step=1, value=20)
    promotions = st.number_input("Promotions", min_value=0, step=1, value=0)
    satisfaction_score = st.number_input("Employee Satisfaction Score", min_value=0.0, max_value=1.0, step=0.1, value=0.8)

    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Encode inputs
        gender_onehot = onehot_encoder.transform([[gender]])
        education_job_encoded = ordinal_encoder.transform([[education_level, job_title]])
        department_freq = freq_encoding.get(department, 0)

        # Combine all features into a single row
        row = [
            age,
            education_job_encoded[0][1],  # Job_Title
            years_at_company,
            education_job_encoded[0][0],  # Education_Level
            performance_score,
            monthly_salary,
            work_hours_per_week,
            projects_handled,
            overtime_hours,
            sick_days,
            remote_work_frequency,
            team_size,
            training_hours,
            promotions,
            satisfaction_score,
            *gender_onehot[0],  # Gender one-hot encoded
            department_freq  # Department frequency encoding
        ]

        # Scale numeric features
        row_scaled = minmax_scaler.transform([row])

        # Predict probabilities using the model
        prediction_prob = model.predict_proba(row_scaled)[0]

        # Adjust threshold for classification
        threshold = 0.5
        prediction = 1 if prediction_prob[1] >= threshold else 0

        # Format the prediction result
        prediction_text = "The employee is likely to resign." if prediction == 1 else "The employee is not likely to resign."

        # Display the result
        st.subheader("Prediction Result")
        st.write(prediction_text)
        st.write(f"Probability of Resignation: {prediction_prob[1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
