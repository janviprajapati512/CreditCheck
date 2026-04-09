import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from difflib import get_close_matches

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", page_icon="💳", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
body {background:#0e1117; color:white;}
.title {font-size:36px;font-weight:bold;color:#4da6ff;text-align:center;}
.subtitle {text-align:center;color:#aaa;margin-bottom:25px;}
.stButton>button {background:#4da6ff;color:black;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CreditCheck AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Reliable • Explainable • Enterprise Ready</div>', unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- HELPERS ---------------- #
def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

def safe_encode(col, value):
    classes = list(encoders[col].classes_)
    value = str(value).strip().title()

    if value in classes:
        return encoders[col].transform([value])[0]

    match = get_close_matches(value, classes, n=1, cutoff=0.6)
    if match:
        return encoders[col].transform([match[0]])[0]

    return -1

# 🔥 NORMALIZATION (balance features)
def normalize(df):
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'] / 100000
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'] / 10
    return df

# 🔥 CREDIT TRANSFORM (less bias)
def transform_credit(score):
    return (900 - score) / 150

# 🔥 EXPLAINABILITY
def explain(df):
    reasons = []

    if df['AMT_INCOME_TOTAL'][0] > 0.5:
        reasons.append("High income improves repayment ability")

    if df['CNT_FAM_MEMBERS'][0] > 0.5:
        reasons.append("More dependents increase financial pressure")

    if df['CREDIT_SCORE'][0] < 2:
        reasons.append("Good credit history increases trust")

    if df['CREDIT_SCORE'][0] > 4:
        reasons.append("Poor credit history increases risk")

    return reasons

# ---------------- INPUT ---------------- #
st.subheader("Applicant Details")

c1, c2, c3 = st.columns(3)

with c1:
    gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
    income = st.number_input("Income", min_value=0)

with c2:
    income_type = select_box("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
    education = select_box("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

with c3:
    family_status = select_box("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
    occupation = select_box("Occupation", encoders['OCCUPATION_TYPE'].classes_)

family_members = st.slider("Family Members", 1, 10, 2)
credit_score = st.slider("Credit Score", 300, 900, 650)

# 🔴 STRICT VALIDATION
errors = []
if gender == "Select": errors.append("Gender")
if income <= 0: errors.append("Income")
if income_type == "Select": errors.append("Income Type")
if education == "Select": errors.append("Education")
if family_status == "Select": errors.append("Family Status")
if occupation == "Select": errors.append("Occupation")

if errors:
    st.warning("Please fill all fields: " + ", ".join(errors))

# ---------------- ANALYZE ---------------- #
if st.button("Analyze", disabled=len(errors) > 0):

    df = pd.DataFrame([[ 
        safe_encode('CODE_GENDER', gender),
        income,
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

    decision = "Approved" if pred == 1 else "Rejected"

    # 🔥 SAFETY CHECK (avoid overconfidence)
    if prob < 55:
        decision = "Borderline"

    # RESULT
    st.markdown("### Result")
    st.success(f"{decision} ({prob:.2f}%)")
    st.progress(int(prob))

    # EXPLANATION
    st.markdown("### Why this decision?")
    for r in explain(df):
        st.write("•", r)

    # =====================================================
    # 📊 EDA (BELOW ANALYZE BUTTON)
    # =====================================================
    st.markdown("## 📊 Quick Data Analysis")

    eda_file = st.file_uploader("Upload Dataset for Analysis", type=["csv"])

    if eda_file:
        edf = pd.read_csv(eda_file)

        st.write("Preview")
        st.dataframe(edf.head())

        # Stats
        st.write("### Statistics")
        st.write(edf.describe())

        # Missing
        st.write("### Missing Values")
        st.write(edf.isnull().sum())

        # Income Distribution
        st.write("### Income Distribution")
        fig, ax = plt.subplots()
        ax.hist(edf['AMT_INCOME_TOTAL'], bins=20)
        st.pyplot(fig)

        # Credit Score Distribution
        st.write("### Credit Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(edf['CREDIT_SCORE'], bins=20)
        st.pyplot(fig)

        # Correlation
        st.write("### Correlation Heatmap")
        corr = edf.corr(numeric_only=True)
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        st.pyplot(fig)
