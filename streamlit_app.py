import streamlit as st
import pickle
import numpy as np

pickle_path = "E:\\Interview Assessment Project\\VIP_Detection_Project\\Pickle_files"
with open(f"{pickle_path}\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{pickle_path}\\best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
with open(f"{pickle_path}\\country_encoding.pkl", "rb") as f:
    country_encoding = pickle.load(f)
with open(f"{pickle_path}\\deposit_method_encoding.pkl", "rb") as f:
    deposit_method_encoding = pickle.load(f)


# Streamlit UI
st.set_page_config(page_title="VIP Detection", layout="wide")
st.title("🎯 VIP Detection Application")
st.markdown("""
This application predicts whether a user is a VIP based on their input details. 
Enter the required details below, and click on **Predict** to get the result.
""")

st.sidebar.header("About")
st.sidebar.markdown("""
- **Purpose**: To classify whether a user is VIP.
- **Model**: Evaluated using Log Loss metric.
- **Threshold**: 0.75 (for classification).
""")

# Input Fields
st.subheader("Input Data")
deposit_amount = st.number_input("Deposit Amount", min_value=0.0, value=100.0, step=10.0)
deposit_quantity = st.number_input("Deposit Quantity", min_value=0, value=1, step=1)
age = st.number_input("Age", min_value=18, value=25, step=1)
deposit_method = st.selectbox("Deposit Method", deposit_method_encoding.keys())
country = st.selectbox("Country", country_encoding.keys())
gender_f = st.radio("Gender", ["Female", "Male", "Non-Disclosed"], index=1)
alert_type = st.selectbox(
    "Alert Type", 
    [
        "All Deposits", "First Deposit", "Other Deposits", 
        "Re Deposits", "Sign Up", "Special"
    ]
)

# Transform Inputs
st.subheader("Transformed Inputs")
features = []

# Numerical Features
features.append(deposit_amount)
features.append(deposit_quantity)
features.append(age)

# Encoded Features
features.append(deposit_method_encoding[deposit_method])
features.append(country_encoding[country])

# Gender Features
if gender_f == "Female":
    features.extend([1, 0])
elif gender_f == "Non-Disclosed":
    features.extend([0, 1])
else:  # Male
    features.extend([0, 0])

# Alert Type One-Hot Encoding
alert_type_encoding = {
    "All Deposits": [1, 0, 0, 0, 0, 0],
    "First Deposit": [0, 1, 0, 0, 0, 0],
    "Other Deposits": [0, 0, 1, 0, 0, 0],
    "Re Deposits": [0, 0, 0, 1, 0, 0],
    "Sign Up": [0, 0, 0, 0, 1, 0],
    "Special": [0, 0, 0, 0, 0, 1],
}
features.extend(alert_type_encoding[alert_type])

# Scale Features
features = np.array(features).reshape(1, -1)
scaled_features = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prob = best_model.predict_proba(scaled_features)[0][1]  # Probability of being VIP

    # Manually calculate log loss for the predicted probability
    if prob > 0:
        log_loss_score = -np.log(prob)
    else:
        log_loss_score = float('inf')  # Edge case handling
    
    st.subheader("Prediction Result")
    if prob > 0.75:
        st.success(f"The user is predicted to be a VIP! 🎉\n\n**Probability**: {prob:.2f}")
    else:
        st.warning(f"The user is predicted **NOT** to be a VIP.\n\n**Probability**: {prob:.2f}")

    st.subheader("Evaluation Metric")
    st.info(f"**Log Loss**: {log_loss_score:.4f}")

