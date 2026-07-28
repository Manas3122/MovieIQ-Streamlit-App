import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 MovieIQ")
st.write("Explore and search the movie dataset.")


@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")


try:
    movies = load_data()

except FileNotFoundError:
    st.error("movies.csv was not found.")
    st.stop()

except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Dataset", "Movie Search"]
)


if page == "Home":
    st.header("Welcome to MovieIQ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Movies", len(movies))

    with col2:
        st.metric("Columns", len(movies.columns))

    with col3:
        st.metric(
            "Missing Values",
            int(movies.isnull().sum().sum())
        )

    st.subheader("Dataset Preview")
    st.dataframe(
        movies.head(),
        use_container_width=True
    )


elif page == "Dataset":
    st.header("📊 Movie Dataset")

    rows = st.slider(
        "Number of rows",
        min_value=5,
        max_value=min(100, len(movies)),
        value=min(20, len(movies))
    )

    st.dataframe(
        movies.head(rows),
        use_container_width=True
    )

    st.write("Available columns:")
    st.write(movies.columns.tolist())


elif page == "Movie Search":
    st.header("🔍 Search for a Movie")

    if "title" not in movies.columns:
        st.error(
            "Your dataset does not contain a column named 'title'."
        )

        st.write("Available columns:")
        st.write(movies.columns.tolist())

    else:
        search_text = st.text_input(
            "Enter a movie title",
            placeholder="Example: Avatar"
        )

        if search_text:
            results = movies[
                movies["title"]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

            if results.empty:
                st.warning("No matching movie was found.")

            else:
                st.success(
                    f"{len(results)} movie(s) found."
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )
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
