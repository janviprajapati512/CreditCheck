import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from difflib import get_close_matches

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- LOAD ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- LOAD TRAIN DATA (IMPORTANT FOR EDA) ---------------- #
try:
    train_df = pd.read_csv("application_record.csv")
except:
    train_df = None

# ---------------- INDIAN FORMAT ---------------- #
def format_inr(num):
    if num >= 10000000:
        return f"₹{num/10000000:.2f} Cr"
    elif num >= 100000:
        return f"₹{num/100000:.2f} L"
    else:
        return f"₹{num:,.0f}"

# Currency conversion (basic fallback)
def convert_to_inr(value, currency="INR"):
    rates = {"USD": 83, "EUR": 90, "INR": 1}
    return value * rates.get(currency, 1)

# ---------------- HELPERS ---------------- #
def safe_encode(col, value):
    classes = list(encoders[col].classes_)
    value = str(value).strip().title()

    if value in classes:
        return encoders[col].transform([value])[0]

    match = get_close_matches(value, classes, n=1, cutoff=0.6)
    if match:
        return encoders[col].transform([match[0]])[0]

    return -1

# Normalize
def normalize(df):
    df['AMT_INCOME_TOTAL'] /= 100000
    df['CNT_FAM_MEMBERS'] /= 10
    return df

# Reduce credit dominance
def transform_credit(score):
    return (900 - score) / 150

# ---------------- UI ---------------- #
st.title("💳 Credit Approval System (India)")

tab1, tab2 = st.tabs(["Individual Prediction", "Bulk Processing"])

# =====================================================
# 🔮 TAB 1: INDIVIDUAL
# =====================================================
with tab1:

    st.subheader("Applicant Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Select"] + list(encoders['CODE_GENDER'].classes_))
        income = st.number_input("Annual Income", min_value=0)

    with c2:
        income_type = st.selectbox("Income Type", ["Select"] + list(encoders['NAME_INCOME_TYPE'].classes_))
        education = st.selectbox("Education", ["Select"] + list(encoders['NAME_EDUCATION_TYPE'].classes_))

    with c3:
        family_status = st.selectbox("Family Status", ["Select"] + list(encoders['NAME_FAMILY_STATUS'].classes_))
        occupation = st.selectbox("Occupation", ["Select"] + list(encoders['OCCUPATION_TYPE'].classes_))

    family_members = st.slider("Family Members", 1, 10, 2)
    credit_score = st.slider("Credit Score", 300, 900, 650)

    # VALIDATION
    valid = all([
        gender != "Select",
        income > 0,
        income_type != "Select",
        education != "Select",
        family_status != "Select",
        occupation != "Select"
    ])

    # BUTTON (STRICT)
    if st.button("Predict", disabled=not valid):

        # Convert to INR if needed
        income_inr = convert_to_inr(income)

        df = pd.DataFrame([[ 
            safe_encode('CODE_GENDER', gender),
            income_inr,
            safe_encode('NAME_INCOME_TYPE', income_type),
            safe_encode('NAME_EDUCATION_TYPE', education),
            safe_encode('NAME_FAMILY_STATUS', family_status),
            safe_encode('OCCUPATION_TYPE', occupation),
            family_members,
            transform_credit(credit_score)
        ]], columns=[
            'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
            'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
        ])

        df = normalize(df)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        # Threshold tuning (bias mitigation)
        if prob > 65:
            decision = "Approved"
        elif prob < 45:
            decision = "Rejected"
        else:
            decision = "Borderline"

        st.success(f"{decision} ({prob:.2f}%)")
        st.write("Income:", format_inr(income_inr))

        # ---------------- INSTANT EDA ---------------- #
        if train_df is not None:
            st.markdown("### 📊 Input vs Dataset Comparison")

            fig, ax = plt.subplots()
            ax.hist(train_df['AMT_INCOME_TOTAL'], bins=30, alpha=0.5)
            ax.axvline(income_inr, linestyle="dashed")
            st.pyplot(fig)

# =====================================================
# 📂 TAB 2: BULK
# =====================================================
with tab2:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview")
        st.dataframe(df.head())

        # CLEAN
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).str.title()

        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].apply(transform_credit)
        df = normalize(df)

        preds = model.predict(df)
        probs = model.predict_proba(df)[:,1] * 100

        df['Decision'] = np.where(probs > 65, "Approved",
                          np.where(probs < 45, "Rejected", "Borderline"))

        df['Confidence'] = probs

        st.markdown("### 📊 Summary")
        st.metric("Total", len(df))
        st.metric("Approved", (df['Decision']=="Approved").sum())
        st.metric("Rejected", (df['Decision']=="Rejected").sum())

        st.dataframe(df)

        # ---------------- ADVANCED EDA ---------------- #
        st.markdown("## 📊 Data Insights")

        # Correlation
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        st.pyplot(fig)

        # Distribution
        fig, ax = plt.subplots()
        ax.hist(df['AMT_INCOME_TOTAL'], bins=20)
        st.pyplot(fig)

        # Feature importance (if tree model)
        if hasattr(model, "feature_importances_"):
            st.markdown("### Feature Importance")
            imp = pd.Series(model.feature_importances_, index=df.columns[:-2])
            st.bar_chart(imp.sort_values(ascending=False))
