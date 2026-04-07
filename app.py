import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", page_icon="💳", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #f8f9fa, #e9ecef);
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #1f3c88;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-top: 15px;
}
.success-card {
    background-color: #d4edda;
    color: #155724;
}
.error-card {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
try:
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
except:
    st.error("❌ Model loading failed")
    st.stop()

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">💳 CreditCheck AI</div>', unsafe_allow_html=True)
st.markdown("### Smart Credit Card Approval System")

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Bulk Upload"])

# ---------------- FUNCTIONS ---------------- #
def get_sample_csv():
    sample = pd.DataFrame({
        "CODE_GENDER": ["M", "F"],
        "AMT_INCOME_TOTAL": [50000, 20000],
        "NAME_INCOME_TYPE": ["Working", "Commercial associate"],
        "NAME_EDUCATION_TYPE": ["Higher education", "Secondary"],
        "NAME_FAMILY_STATUS": ["Married", "Single"],
        "OCCUPATION_TYPE": ["Managers", "Sales staff"],
        "CNT_FAM_MEMBERS": [3, 2],
        "CREDIT_SCORE": [750, 600]
    })
    return sample.to_csv(index=False).encode('utf-8')

def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

# =========================================================
# 👤 SINGLE PREDICTION
# =========================================================
with tab1:

    st.markdown("### 👤 Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
        income = st.number_input("Annual Income (₹)", min_value=0)

    with col2:
        income_type = select_box("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
        education = select_box("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

    with col3:
        family_status = select_box("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
        occupation = select_box("Occupation", encoders['OCCUPATION_TYPE'].classes_)

    family_members = st.slider("Family Members", 1, 10, 2)
    credit_score_user = st.slider("Credit Score", 300, 900, 650)

    # ---------------- REAL-TIME VALIDATION ---------------- #
    errors = []

    if gender == "Select":
        errors.append("Gender")

    if income <= 0:
        errors.append("Income")

    if income_type == "Select":
        errors.append("Income Type")

    if education == "Select":
        errors.append("Education")

    if family_status == "Select":
        errors.append("Family Status")

    if occupation == "Select":
        errors.append("Occupation")

    # Show inline warning
    if errors:
        st.warning(f"⚠️ Please fill: {', '.join(errors)}")

    # ---------------- SMART BUTTON ---------------- #
    is_valid = len(errors) == 0

    if st.button("🔍 Analyze Credit Profile", disabled=not is_valid):

        try:
            credit_score = (900 - credit_score_user) / 100

            data = pd.DataFrame([[
                encoders['CODE_GENDER'].transform([gender])[0],
                income,
                encoders['NAME_INCOME_TYPE'].transform([income_type])[0],
                encoders['NAME_EDUCATION_TYPE'].transform([education])[0],
                encoders['NAME_FAMILY_STATUS'].transform([family_status])[0],
                encoders['OCCUPATION_TYPE'].transform([occupation])[0],
                family_members,
                credit_score
            ]], columns=[
                'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
            ])

            pred = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1] * 100

            # ---------------- RESULT UI ---------------- #
            st.markdown("## 📊 Result")

            if pred == 1:
                st.markdown(
                    f'<div class="card success-card">🎉 APPROVED<br>Confidence: {prob:.2f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="card error-card">⚠️ REJECTED<br>Risk: {100-prob:.2f}%</div>',
                    unsafe_allow_html=True
                )

            st.progress(int(prob))

        except Exception as e:
            st.error("❌ Error in prediction")
            st.write(e)
