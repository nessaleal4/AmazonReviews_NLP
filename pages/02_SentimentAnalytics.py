import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

st.title("Sentiment Analytics by Category")
st.markdown("""
This page displays overall and category-specific sentiment analytics derived from our preprocessed Amazon reviews dataset.
""")

@st.cache_data
def load_data():
    # Adjust this path to match where your preprocessed data is stored.
    # It could be a CSV or Parquet file.
    try:
        # For example, if you saved data as CSV:
        df = pd.read_csv("data/processed/all_reviews.csv")
        # Alternatively, if using Parquet:
        # df = pd.read_parquet("data/processed/all_reviews.parquet")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()
    return df

df = load_data()

if not df.empty:
    st.markdown("### Overall Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    # Create a pie chart for overall sentiment distribution
    fig_overall = px.pie(sentiment_counts, values="Count", names="Sentiment",
                         title="Overall Sentiment Distribution")
    st.plotly_chart(fig_overall, use_container_width=True)

    st.markdown("### Sentiment Distribution by Category")
    # Group data by category and sentiment
    if "category" in df.columns:
        df_grouped = df.groupby(["category", "sentiment"]).size().reset_index(name="Count")
        fig_category = px.bar(df_grouped, x="category", y="Count", color="sentiment", barmode="group",
                              title="Sentiment Distribution by Category",
                              labels={"category": "Category", "Count": "Number of Reviews"})
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.warning("The data does not include category information.")

    st.markdown("### Detailed Data")
    st.dataframe(df)
else:
    st.write("No data available. Please ensure the preprocessed data file exists.")
