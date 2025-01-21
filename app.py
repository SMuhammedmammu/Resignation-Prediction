from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template(
        'index.html',
        gender_options=gender_options,
        education_options=education_options,
        job_title_options=job_title_options,
        department_options=department_options
    )

@app.route('/result', methods=['POST'])
def result():
    try:
        # Collect user inputs
        input_data = {
            'Gender': request.form.get('Gender', '').strip(),
            'Education_Level': request.form.get('Education_Level', '').strip(),
            'Job_Title': request.form.get('Job_Title', '').strip(),
            'Department': request.form.get('Department', '').strip(),
            'Age': float(request.form.get('Age', 0)),
            'Years_At_Company': float(request.form.get('Years_At_Company', 0)),
            'Performance_Score': float(request.form.get('Performance_Score', 0)),
            'Monthly_Salary': float(request.form.get('Monthly_Salary', 0)),
            'Work_Hours_Per_Week': float(request.form.get('Work_Hours_Per_Week', 0)),
            'Projects_Handled': float(request.form.get('Projects_Handled', 0)),
            'Overtime_Hours': float(request.form.get('Overtime_Hours', 0)),
            'Sick_Days': float(request.form.get('Sick_Days', 0)),
            'Remote_Work_Frequency': float(request.form.get('Remote_Work_Frequency', 0)),
            'Team_Size': float(request.form.get('Team_Size', 0)),
            'Training_Hours': float(request.form.get('Training_Hours', 0)),
            'Promotions': float(request.form.get('Promotions', 0)),
            'Employee_Satisfaction_Score': float(request.form.get('Employee_Satisfaction_Score', 0)),
        }

        # Check if required dropdown fields are empty
        if not input_data['Gender'] or not input_data['Education_Level'] or not input_data['Job_Title'] or not input_data['Department']:
            return "Error: Please ensure all dropdown fields are selected.", 400

        # Encoding inputs
        gender_onehot = onehot_encoder.transform([[input_data['Gender']]])
        education_job_encoded = ordinal_encoder.transform(
            [[input_data['Education_Level'], input_data['Job_Title']]]
        )
        department_freq = freq_encoding.get(input_data['Department'], 0)

        # Combine all features into a single row
        row = [
            input_data['Age'],
            education_job_encoded[0][1],  # Job_Title
            input_data['Years_At_Company'],
            education_job_encoded[0][0],  # Education_Level
            input_data['Performance_Score'],
            input_data['Monthly_Salary'],
            input_data['Work_Hours_Per_Week'],
            input_data['Projects_Handled'],
            input_data['Overtime_Hours'],
            input_data['Sick_Days'],
            input_data['Remote_Work_Frequency'],
            input_data['Team_Size'],
            input_data['Training_Hours'],
            input_data['Promotions'],
            input_data['Employee_Satisfaction_Score'],
            *gender_onehot[0],  # Gender one-hot encoded
            department_freq  # Department frequency encoding
        ]

        # Scale numeric features
        row_scaled = minmax_scaler.transform([row])

        # Predict using the loaded model
        prediction = model.predict(row_scaled)[0]

        # Format the prediction result for better readability
        prediction_text = "The employee is likely to resign." if prediction == 1 else "The employee is not likely to resign."

        return render_template('result.html', prediction=prediction_text)

    except ValueError as e:
        return f"Error: Invalid input data. {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)