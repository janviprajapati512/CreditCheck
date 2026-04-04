import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------- LOAD DATA ---------------- #
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# ---------------- CREDIT MAPPING ---------------- #
credit['STATUS'] = credit['STATUS'].replace({
    'C': 0, 'X': 0,
    '0': 1, '1': 2, '2': 3,
    '3': 4, '4': 5, '5': 6
}).infer_objects(copy=False)

# Average credit behavior
credit_score = credit.groupby('ID')['STATUS'].mean().reset_index()
credit_score.rename(columns={'STATUS': 'CREDIT_SCORE'}, inplace=True)

# Merge
data = application.merge(credit_score, on='ID', how='inner')

# ---------------- ✅ IMPROVED TARGET ---------------- #
# Multi-condition approval (REALISTIC)
def approve_logic(row):
    if row['CREDIT_SCORE'] < 2 and row['AMT_INCOME_TOTAL'] > 30000:
        return 1
    elif row['CREDIT_SCORE'] < 1.5:
        return 1
    else:
        return 0

data['APPROVED'] = data.apply(approve_logic, axis=1)

# ---------------- FEATURES ---------------- #
features = [
    'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
    'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
]

data = data[features + ['APPROVED']]
data.dropna(inplace=True)

# ---------------- ENCODING ---------------- #
encoders = {}

for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ---------------- TRAIN ---------------- #
X = data.drop('APPROVED', axis=1)
y = data['APPROVED']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

pipeline.fit(X_train, y_train)

# ---------------- SAVE ---------------- #
pickle.dump(pipeline, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ Model trained & saved successfully")