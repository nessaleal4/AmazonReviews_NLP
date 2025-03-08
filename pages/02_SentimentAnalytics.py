import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px

# Set up the page
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

st.title("Sentiment Analytics by Category")
st.markdown("""
This dashboard loads preprocessed CSV files (with sentiment columns) from Google Drive and 
visualizes sentiment distribution across review categories.
""")

# Dictionary of categories to their Google Drive File IDs
DRIVE_FILE_IDS = {
    "Beauty_and_Personal_Care": "1ZhGvPq2xvjljm530XOHxYlKN1Ssv8mkA",
    "Books": "1F-ypoWmen8wlJb8SZ9cO9quHSqY6Ji8J",
    "Electronics": "1i2qYCcWixYe-zFUyC73O9hOAA9Hwr2-z",
    "Home_and_Kitchen": "1A4pHlEbKL1PDrqMoSf2_aZb2a48H_t8w"
}

@st.cache_data
def load_csv_from_drive(file_id: str) -> pd.DataFrame:
    """
    Download a CSV file from Google Drive using its file ID and return a DataFrame.
    """
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()  # Raise an HTTPError if the download fails
    return pd.read_csv(BytesIO(response.content))

@st.cache_data
def load_all_categories() -> pd.DataFrame:
    """
    Load all category CSVs from Google Drive, merge them into one DataFrame, and 
    return the combined data with a 'category' column.
    """
    all_dfs = []
    for category, file_id in DRIVE_FILE_IDS.items():
        try:
            df = load_csv_from_drive(file_id)
            # Ensure we have a 'category' column in case the CSV doesn't have one
            if "category" not in df.columns:
                df["category"] = category
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading data for {category}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Load all category data
df = load_all_categories()

if df.empty:
    st.warning("No data available. Please ensure your Google Drive links are correct.")
else:
    # Check that 'sentiment' column exists
    if "sentiment" not in df.columns:
        st.error("No 'sentiment' column found in the merged data.")
    else:
        # Convert category column if it's missing or inconsistent
        if "category" not in df.columns:
            df["category"] = "Unknown"

        # Group data by Category and Sentiment
        df_grouped = df.groupby(["category", "sentiment"]).size().reset_index(name="Count")

        # Create a grouped bar chart using Plotly Express
        fig = px.bar(
            df_grouped,
            x="category",
            y="Count",
            color="sentiment",
            barmode="group",
            title="Sentiment Distribution by Category",
            labels={"category": "Category", "Count": "Number of Reviews"}
        )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

        # Display a sample of the raw data
        st.markdown("### Detailed Data (Sample)")
        st.dataframe(df.head(50), use_container_width=True)
