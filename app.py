import streamlit as st
import pandas as pd
import joblib
from difflib import get_close_matches

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", page_icon="💳", layout="wide")

# ---------------- CLEAN UI ---------------- #
st.markdown("""
<style>
body {background:#f4f6f9;}
.title {font-size:38px;font-weight:bold;color:#1f3c88;text-align:center;}
.subtitle {text-align:center;color:#555;margin-bottom:25px;}

.card {
    padding:20px;
    border-radius:12px;
    background:white;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

.success-card {background:#e6f4ea;color:black;font-weight:600;}
.error-card {background:#fdecea;color:black;font-weight:600;}

.stButton>button {
    background:#1f3c88;
    color:white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">💳 CreditCheck AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise Credit Approval System</div>', unsafe_allow_html=True)

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

    return -1  # unseen safe

def get_sample_csv():
    df = pd.DataFrame({
        "CODE_GENDER":["M","F"],
        "AMT_INCOME_TOTAL":[50000,20000],
        "NAME_INCOME_TYPE":["Working","Commercial associate"],
        "NAME_EDUCATION_TYPE":["Higher education","Secondary"],
        "NAME_FAMILY_STATUS":["Married","Single"],
        "OCCUPATION_TYPE":["Managers","Sales staff"],
        "CNT_FAM_MEMBERS":[3,2],
        "CREDIT_SCORE":[750,600]
    })
    return df.to_csv(index=False).encode("utf-8")

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["👤 Single Prediction", "🏢 Bulk Upload"])

# =========================================================
# 👤 SINGLE PREDICTION
# =========================================================
with tab1:

    st.subheader("Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
        income = st.number_input("Income ($)", min_value=0)

    with col2:
        income_type = select_box("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
        education = select_box("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

    with col3:
        family_status = select_box("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
        occupation = select_box("Occupation", encoders['OCCUPATION_TYPE'].classes_)

    family_members = st.slider("Family Members",1,10,2)
    credit_score_user = st.slider("Credit Score",300,900,650)

    # VALIDATION
    errors = []
    if gender=="Select": errors.append("Gender")
    if income<=0: errors.append("Income")
    if income_type=="Select": errors.append("Income Type")
    if education=="Select": errors.append("Education")
    if family_status=="Select": errors.append("Family Status")
    if occupation=="Select": errors.append("Occupation")

    if errors:
        st.warning("⚠️ Fill: " + ", ".join(errors))

    if st.button("Analyze", disabled=len(errors)>0):

        credit_score_model = (900-credit_score_user)/100

        df = pd.DataFrame([[ 
            safe_encode('CODE_GENDER', gender),
            income,
            safe_encode('NAME_INCOME_TYPE', income_type),
            safe_encode('NAME_EDUCATION_TYPE', education),
            safe_encode('NAME_FAMILY_STATUS', family_status),
            safe_encode('OCCUPATION_TYPE', occupation),
            family_members,
            credit_score_model
        ]], columns=[
            'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
            'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
        ])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]*100

        st.markdown("## 📊 Result")

        if pred==1:
            st.markdown(f'<div class="card success-card">🎉 Approved<br>Confidence: {prob:.2f}%</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card error-card">⚠️ Rejected<br>Risk: {100-prob:.2f}%</div>',unsafe_allow_html=True)

        st.progress(int(prob))
        st.info(f"📊 Credit Score: {credit_score_user}")

# =========================================================
# 🏢 BULK UPLOAD
# =========================================================
with tab2:

    st.subheader("🏢 Bulk Credit Evaluation")

    st.download_button("📥 Sample CSV", get_sample_csv(), "sample.csv")

    file = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])

    if file:

        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

        st.write("### 📄 Preview")
        st.dataframe(df.head())

        # CLEAN
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).str.strip().str.title()

        # STORE ORIGINAL SCORE
        original_score = df['CREDIT_SCORE'].copy()

        # TRANSFORM FOR MODEL
        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].apply(lambda x: (900-x)/100 if x>10 else x)

        valid=[]
        errors=[]

        for idx,row in df.iterrows():
            try:
                temp = pd.DataFrame([[ 
                    safe_encode('CODE_GENDER',row['CODE_GENDER']),
                    row['AMT_INCOME_TOTAL'],
                    safe_encode('NAME_INCOME_TYPE',row['NAME_INCOME_TYPE']),
                    safe_encode('NAME_EDUCATION_TYPE',row['NAME_EDUCATION_TYPE']),
                    safe_encode('NAME_FAMILY_STATUS',row['NAME_FAMILY_STATUS']),
                    safe_encode('OCCUPATION_TYPE',row['OCCUPATION_TYPE']),
                    row['CNT_FAM_MEMBERS'],
                    row['CREDIT_SCORE']
                ]], columns=[
                    'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                    'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
                ])

                pred = model.predict(temp)[0]
                prob = model.predict_proba(temp)[0][1]*100

                r=row.to_dict()
                r["Prediction"]="Approved" if pred==1 else "Rejected"
                r["Confidence (%)"]=round(prob,2)
                r["Credit Score"]=original_score[idx]

                valid.append(r)

            except Exception as e:
                er=row.to_dict()
                er["Error"]=str(e)
                errors.append(er)

        valid_df=pd.DataFrame(valid)
        error_df=pd.DataFrame(errors)

        # REMOVE MODEL SCORE COLUMN
        valid_df.drop(columns=["CREDIT_SCORE"], inplace=True, errors="ignore")

        # METRICS
        total=len(df)
        ok=len(valid_df)
        bad=len(error_df)

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total",total)
        c2.metric("Valid",ok)
        c3.metric("Invalid",bad)
        c4.metric("Quality %",round((ok/total)*100,2) if total>0 else 0)

        # COLOR ROWS
        def color_rows(row):
            if row["Prediction"] == "Approved":
                return ["background-color:#e6f4ea; color:black"] * len(row)
            else:
                return ["background-color:#fdecea; color:black"] * len(row)

        st.markdown("## ✅ Results")

        if not valid_df.empty:
            st.dataframe(valid_df.style.apply(color_rows,axis=1))
            st.download_button("⬇ Download Results", valid_df.to_csv(index=False), "results.csv")

        st.markdown("## ❌ Errors")

        if not error_df.empty:
            st.dataframe(error_df)
            st.download_button("⬇ Download Errors", error_df.to_csv(index=False), "errors.csv")
