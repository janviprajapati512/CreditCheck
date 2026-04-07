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
st.markdown("### Credit Card Approval System")

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Bulk CSV Upload"])

# ---------------- COMMON FUNCTIONS ---------------- #
def get_score_label(score):
    if score < 500:
        return "🔴 Poor"
    elif score < 650:
        return "🟠 Average"
    elif score < 750:
        return "🟡 Good"
    else:
        return "🟢 Excellent"

def select_with_placeholder(label, options):
    return st.selectbox(label, ["-- Please Select --"] + list(options))

# =========================================================
# 👤 TAB 1 - SINGLE PREDICTION
# =========================================================
with tab1:

    st.subheader("Enter Applicant Details")

    gender = select_with_placeholder("👤 Gender *", encoders['CODE_GENDER'].classes_)
    income = st.number_input("💰 Annual Income (₹) *", min_value=0)

    income_type = select_with_placeholder("💼 Income Type *", encoders['NAME_INCOME_TYPE'].classes_)
    education = select_with_placeholder("🎓 Education *", encoders['NAME_EDUCATION_TYPE'].classes_)
    family_status = select_with_placeholder("👨‍👩‍👧 Family Status *", encoders['NAME_FAMILY_STATUS'].classes_)
    occupation = select_with_placeholder("🧑‍💼 Occupation *", encoders['OCCUPATION_TYPE'].classes_)

    family_members = st.number_input("👨‍👩‍👧‍👦 Family Members *", min_value=1)

    credit_score_user = st.slider(
        "📊 Credit Score (300 - 900) *",
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

    for e in errors:
        st.error(f"❌ {e}")

    # ---------------- BUTTON ---------------- #
    if st.button("🔍 Check Approval", disabled=len(errors) > 0):

        with st.spinner("🔄 Checking..."):

            try:
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

                pred = model.predict(df)[0]

                try:
                    prob = model.predict_proba(df)[0][1] * 100
                except:
                    prob = None

                st.markdown("## 📊 Result")

                if pred == 1:
                    st.success(f"🎉 Approved ({prob:.2f}% confidence)" if prob else "🎉 Approved")
                else:
                    st.error(f"⚠️ Rejected ({100 - prob:.2f}% risk)" if prob else "⚠️ Rejected")

            except Exception as e:
                st.error("❌ Prediction failed")
                st.write(str(e))

# =========================================================
# 📂 TAB 2 - BULK CSV UPLOAD
# =========================================================
with tab2:

    st.subheader("Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.write("### 📄 Preview")
            st.dataframe(df.head())

            required_columns = [
                'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
            ]

            missing = [col for col in required_columns if col not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()

            # Convert credit score (if user uploads 300–900)
            if df['CREDIT_SCORE'].max() > 10:
                df['CREDIT_SCORE'] = (900 - df['CREDIT_SCORE']) / 100

            # Encoding
            df['CODE_GENDER'] = encoders['CODE_GENDER'].transform(df['CODE_GENDER'])
            df['NAME_INCOME_TYPE'] = encoders['NAME_INCOME_TYPE'].transform(df['NAME_INCOME_TYPE'])
            df['NAME_EDUCATION_TYPE'] = encoders['NAME_EDUCATION_TYPE'].transform(df['NAME_EDUCATION_TYPE'])
            df['NAME_FAMILY_STATUS'] = encoders['NAME_FAMILY_STATUS'].transform(df['NAME_FAMILY_STATUS'])
            df['OCCUPATION_TYPE'] = encoders['OCCUPATION_TYPE'].transform(df['OCCUPATION_TYPE'])

            # Prediction
            preds = model.predict(df)

            try:
                probs = model.predict_proba(df)[:, 1] * 100
            except:
                probs = None

            df['Prediction'] = ["Approved" if p == 1 else "Rejected" for p in preds]

            if probs is not None:
                df['Confidence (%)'] = probs.round(2)

            # Show results
            st.write("### ✅ Results")
            st.dataframe(df)

            # Summary
            st.write("### 📊 Summary")
            st.write(df['Prediction'].value_counts())

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇ Download Results",
                data=csv,
                file_name="credit_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("❌ Error processing file")
            st.write(str(e))
