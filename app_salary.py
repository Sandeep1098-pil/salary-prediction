import streamlit as st
import joblib
import pandas as pd
import datetime

# Background Styling
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://images.blocksurvey.io/templates/employee-salary-compensation-survey.png");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and columns
model = joblib.load('final_xgb_model.pkl')
model_columns = joblib.load('salary_model_columns.pkl')

st.title("Salary Prediction App")

job_titles = [col for col in model_columns if col not in ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Joined_year']]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    education = st.selectbox("Education Level", options=[0, 1, 2, 3],
                             format_func=lambda x: ["High School", "Bachelor's", "Master's", "PhD"][x])
    experience = st.number_input("Years of Experience (including pre joining experience)", min_value=0, max_value=50, value=0)
    joined_year = st.number_input("Joined Year", min_value=1980, max_value=2025, value=2020)
    job_title = st.selectbox("Job Title", job_titles)

    if st.button("Predict Salary"):
        current_year = datetime.datetime.now().year

        # Minimum graduation age based on education
        min_grad_age = {0: 17, 1: 21, 2: 23, 3: 27}
        grad_age = min_grad_age.get(education, 17)
        max_exp = age - grad_age

        # Calculate experience since joining
        company_exp = current_year - joined_year

        # Validations
        if age < grad_age:
            st.error(f"Age must be at least {grad_age} for the selected education level.")
        elif experience < 0 or experience > max_exp:
            st.error(f"Years of Experience must be between 0 and {max_exp} for your age & education.")
        elif joined_year < 1980 or joined_year > 2025:
            st.error("Joined Year must be between 1980 and 2025.")
        elif age < 18 or age > 60:
            st.error("Age must be between 18 and 60.")
        # 🚨 New strict rule: if experience < years since joining → ERROR
        elif experience < company_exp:
            st.error(
                f"Experience mismatch! Since you joined in {joined_year}, "
                f"you should have at least {company_exp} years of experience."
            )
        else:
            # Prepare input dataframe
            data = {
                'Age': age,
                'Gender': gender,
                'Education Level': education,
                'Years of Experience': experience,
                'Joined_year': joined_year
            }
            input_df = pd.DataFrame([data])
            for col in job_titles:
                input_df[col] = 1 if col == job_title else 0
            input_df = input_df[model_columns]

            # Predict salary
            pred = model.predict(input_df)[0]
            st.success(f"Predicted Salary:  ${pred:,.2f}")
