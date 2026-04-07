import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", page_icon="💳", layout="wide")

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.title {font-size:40px;font-weight:bold;color:#1f3c88;}
.card {padding:20px;border-radius:15px;background:white;
box-shadow:0 4px 10px rgba(0,0,0,0.1);margin-top:15px;}
.success-card {background-color:#d4edda;color:#155724;}
.error-card {background-color:#f8d7da;color:#721c24;}
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
st.markdown("### Enterprise Credit Approval System")

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["👤 Single Prediction", "🏢 Bulk Upload"])

# ---------------- HELPERS ---------------- #
def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

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

# =========================================================
# 👤 SINGLE PREDICTION
# =========================================================
with tab1:

    st.subheader("Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
        income = st.number_input("Income ₹", min_value=0)

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

        credit_score = (900-credit_score_user)/100

        df = pd.DataFrame([[
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

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]*100

        st.markdown("## Result")

        if pred==1:
            st.markdown(f'<div class="card success-card">Approved<br>{prob:.2f}%</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card error-card">Rejected<br>{100-prob:.2f}% risk</div>',unsafe_allow_html=True)

        st.progress(int(prob))

# =========================================================
# 🏢 ENTERPRISE BULK
# =========================================================
with tab2:

    st.subheader("Bulk Prediction")

    st.download_button("📥 Sample CSV", get_sample_csv(), "sample.csv")

    file = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])

    if file:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.dataframe(df.head())

        # CLEAN
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).str.strip().str.title()

        # AUTO FIX
        df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace({
            "Secondary":"Secondary / Secondary Special"
        })

        # CREDIT SCORE FIX
        if df['CREDIT_SCORE'].max()>10:
            df['CREDIT_SCORE'] = (900-df['CREDIT_SCORE'])/100

        from difflib import get_close_matches

        def fix(val, classes):
            if val in classes:
                return val
            match = get_close_matches(val, classes, n=1, cutoff=0.6)
            return match[0] if match else None

        valid=[]
        errors=[]

        for _,row in df.iterrows():
            try:

                def enc(col,val):
                    classes = list(encoders[col].classes_)
                    val = fix(val, classes)
                    if val is None:
                        raise ValueError(f"{col} invalid")
                    return encoders[col].transform([val])[0]

                row_data = [[
                    enc('CODE_GENDER',row['CODE_GENDER']),
                    row['AMT_INCOME_TOTAL'],
                    enc('NAME_INCOME_TYPE',row['NAME_INCOME_TYPE']),
                    enc('NAME_EDUCATION_TYPE',row['NAME_EDUCATION_TYPE']),
                    enc('NAME_FAMILY_STATUS',row['NAME_FAMILY_STATUS']),
                    enc('OCCUPATION_TYPE',row['OCCUPATION_TYPE']),
                    row['CNT_FAM_MEMBERS'],
                    row['CREDIT_SCORE']
                ]]

                temp = pd.DataFrame(row_data, columns=[
                    'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                    'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
                ])

                pred = model.predict(temp)[0]
                prob = model.predict_proba(temp)[0][1]*100

                r = row.to_dict()
                r["Prediction"]="Approved" if pred==1 else "Rejected"
                r["Confidence"]=round(prob,2)

                valid.append(r)

            except Exception as e:
                er = row.to_dict()
                er["Error"]=str(e)
                errors.append(er)

        valid_df = pd.DataFrame(valid)
        error_df = pd.DataFrame(errors)

        total=len(df)
        ok=len(valid_df)
        bad=len(error_df)

        st.metric("Total", total)
        st.metric("Valid", ok)
        st.metric("Invalid", bad)
        st.metric("Quality %", round((ok/total)*100,2) if total>0 else 0)

        st.markdown("### ✅ Valid Results")
        if not valid_df.empty:
            st.dataframe(valid_df)
            st.download_button("Download Valid", valid_df.to_csv(index=False), "valid.csv")

        st.markdown("### ❌ Errors")
        if not error_df.empty:
            st.dataframe(error_df)
            st.download_button("Download Errors", error_df.to_csv(index=False), "errors.csv")
