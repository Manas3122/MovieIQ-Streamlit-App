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
