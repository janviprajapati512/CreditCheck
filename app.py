import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ---------------- #
try:
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
except Exception as e:
    st.error("❌ Model loading failed. Please check files.")
    st.stop()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", page_icon="💳")

# ---------------- HEADER ---------------- #
st.title("💳 CreditCheck AI")
st.markdown("### Simple Credit Approval System")

st.info("""
✔ Enter your details  
✔ System checks your profile  
✔ Shows approval result with reason + suggestions  
""")

# ---------------- CREDIT LABEL ---------------- #
def get_score_label(score):
    if score < 500:
        return "🔴 Poor"
    elif score < 650:
        return "🟠 Average"
    elif score < 750:
        return "🟡 Good"
    else:
        return "🟢 Excellent"

# ---------------- DROPDOWN HELPER ---------------- #
def select_with_placeholder(label, options):
    return st.selectbox(label, ["-- Please Select --"] + list(options))

# ---------------- INPUT ---------------- #
gender = select_with_placeholder("👤 Gender *", encoders['CODE_GENDER'].classes_)
income = st.number_input("💰 Annual Income ($) *", min_value=0)

income_type = select_with_placeholder("💼 Income Type *", encoders['NAME_INCOME_TYPE'].classes_)
education = select_with_placeholder("🎓 Education *", encoders['NAME_EDUCATION_TYPE'].classes_)
family_status = select_with_placeholder("👨‍👩‍👧 Family Status *", encoders['NAME_FAMILY_STATUS'].classes_)
occupation = select_with_placeholder("🧑‍💼 Occupation *", encoders['OCCUPATION_TYPE'].classes_)

family_members = st.number_input("👨‍👩‍👧‍👦 Family Members *", min_value=1)

credit_score_user = st.slider(
    "📊 Credit Score (300 = poor, 900 = excellent) *",
    300, 900, 650
)

st.write(f"Score Category: {get_score_label(credit_score_user)}")

# ---------------- VALIDATION ---------------- #
errors = []

if gender == "-- Please Select --":
    errors.append("Select Gender")

if income <= 0:
    errors.append("Income must be greater than 0")

if income_type == "-- Please Select --":
    errors.append("Select Income Type")

if education == "-- Please Select --":
    errors.append("Select Education")

if family_status == "-- Please Select --":
    errors.append("Select Family Status")

if occupation == "-- Please Select --":
    errors.append("Select Occupation")

if family_members <= 0:
    errors.append("Family members must be at least 1")

# Show errors
if errors:
    for e in errors:
        st.error(f"❌ {e}")

# ---------------- BUTTON ---------------- #
is_valid = len(errors) == 0

if st.button("🔍 Check Approval", disabled=not is_valid):

    with st.spinner("🔄 Checking your credit profile..."):

        try:
            # ---------------- DATA PREP ---------------- #
            credit_score = (900 - credit_score_user) / 100

            user_data = [
                encoders['CODE_GENDER'].transform([gender])[0],
                income,
                encoders['NAME_INCOME_TYPE'].transform([income_type])[0],
                encoders['NAME_EDUCATION_TYPE'].transform([education])[0],
                encoders['NAME_FAMILY_STATUS'].transform([family_status])[0],
                encoders['OCCUPATION_TYPE'].transform([occupation])[0],
                family_members,
                credit_score
            ]

            columns = [
                'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
            ]

            df = pd.DataFrame([user_data], columns=columns)

            # ---------------- PREDICTION ---------------- #
            pred = model.predict(df)[0]

            try:
                prob = model.predict_proba(df)[0][1] * 100
            except:
                prob = None

            # ---------------- RESULT ---------------- #
            st.markdown("## 📊 Result")

            if pred == 1:
                if prob:
                    st.success(f"🎉 Approved (Confidence: {prob:.2f}%)")
                else:
                    st.success("🎉 Approved")
            else:
                if prob:
                    st.error(f"⚠️ Rejected (Risk: {100 - prob:.2f}%)")
                else:
                    st.error("⚠️ Rejected")

            # ---------------- CONFIDENCE BAR ---------------- #
            if prob:
                st.progress(int(prob))
                st.write(f"Confidence Score: {prob:.2f}%")

            # ---------------- WHY ---------------- #
            st.markdown("### 🧠 Why this decision?")

            reasons = []

            if credit_score_user < 600:
                reasons.append("Low credit score")

            if income < 30000:
                reasons.append("Low income")

            if family_members > 5:
                reasons.append("High financial responsibility")

            if credit_score_user > 750:
                reasons.append("Strong credit history")

            if income > 50000:
                reasons.append("Good income level")

            if not reasons:
                reasons.append("Based on overall profile")

            for r in reasons:
                st.write(f"• {r}")

            # ---------------- IMPROVEMENTS ---------------- #
            st.markdown("### 🚀 How to improve your chances")

            suggestions = []

            if credit_score_user < 750:
                suggestions.append("Improve credit score by paying bills on time")

            if income < 30000:
                suggestions.append("Increase income through better job or side hustle")

            if family_members > 5:
                suggestions.append("Reduce financial liabilities if possible")

            if credit_score_user < 500:
                suggestions.append("Avoid late payments and reduce loan defaults")

            if not suggestions:
                suggestions.append("Your profile is strong. Maintain good financial habits")

            for s in suggestions:
                st.write(f"✔ {s}")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.write(str(e))

