import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 MovieIQ")
st.subheader("Movie Recommendation and Analysis App")


@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")


try:
    movies = load_data()

    st.success("Dataset loaded successfully!")

    st.metric("Total Movies", len(movies))

    st.subheader("Movie Dataset")
    st.dataframe(
        movies.head(20),
        use_container_width=True
    )

except FileNotFoundError:
    st.error("movies.csv file was not found.")

except Exception as error:
    st.error(f"An error occurred: {error}")
