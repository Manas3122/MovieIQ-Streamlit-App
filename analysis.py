"""
MovieIQ complete reproducible analysis.
Run:
    python analysis.py
"""
from pathlib import Path
import pandas as pd
import joblib
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "movies.csv")
df["genre_list"] = df["genres"].fillna("Unknown").apply(
    lambda x: [g for g in str(x).split("|") if g] or ["Unknown"]
)
df["success"] = (df["revenue"] > df["budget"]).astype(int)

print("Dataset shape:", df.shape)
print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())
print("Success rate:", round(df["success"].mean() * 100, 2), "%")

successful = df.loc[df["success"] == 1, "popularity"]
unsuccessful = df.loc[df["success"] == 0, "popularity"]
print("Welch T-Test:", ttest_ind(successful, unsuccessful, equal_var=False))

genre_df = df.assign(genre=df["genre_list"]).explode("genre")
print("Chi-Square:", chi2_contingency(pd.crosstab(genre_df["genre"], genre_df["success"]))[:3])

features = ["budget", "popularity", "runtime", "vote_average", "primary_genre"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["success"], test_size=0.20, random_state=42, stratify=df["success"]
)

numeric = ["budget", "popularity", "runtime", "vote_average"]
categorical = ["primary_genre"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
joblib.dump(model, BASE / "models" / "movieiq_random_forest.joblib")
print("Model saved.")
