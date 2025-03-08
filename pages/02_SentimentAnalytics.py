import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

st.title("Sentiment Analytics by Category")
st.markdown("""
This dashboard displays sentiment distribution across review categories based on preprocessed CSV data stored on Google Drive.
""")

# Dictionary of categories to Google Drive File IDs
DRIVE_FILE_IDS = {
    "Beauty_and_Personal_Care": "1ZhGvPq2xvjljm530XOHxYlKN1Ssv8mkA",
    "Books": "1F-ypoWmen8wlJb8SZ9cO9quHSqY6Ji8J",
    "Electronics": "1i2qYCcWixYe-zFUyC73O9hOAA9Hwr2-z",
    "Home_and_Kitchen": "1A4pHlEbKL1PDrqMoSf2_aZb2a48H_t8w"
}

@st.cache_data
def load_csv_from_drive(file_id: str) -> pd.DataFrame:
    """
    Downloads a CSV file from Google Drive using its file ID and returns a DataFrame.
    Uses the Python engine with on_bad_lines="skip" to handle tokenizing errors.
    """
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    # Use the Python engine and skip bad lines
    df = pd.read_csv(BytesIO(response.content), engine="python", on_bad_lines="skip")
    return df

@st.cache_data
def load_all_categories() -> pd.DataFrame:
    """
    Loads CSV files for all categories from Google Drive,
    adds a 'category' column if missing, and concatenates them into one DataFrame.
    """
    all_dfs = []
    for category, file_id in DRIVE_FILE_IDS.items():
        try:
            df = load_csv_from_drive(file_id)
            # Ensure there's a category column
            if "category" not in df.columns:
                df["category"] = category
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading data for {category}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Load all category data from Drive
df_all = load_all_categories()

if df_all.empty:
    st.warning("No data available. Please ensure your Google Drive links are correct.")
else:
    st.markdown("### Overall Sentiment Distribution")
    if "sentiment" not in df_all.columns:
        st.error("The data does not include a 'sentiment' column.")
    else:
        # Group by category and sentiment
        df_grouped = df_all.groupby(["category", "sentiment"]).size().reset_index(name="Count")
        
        # Create a grouped bar chart with Plotly Express
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
        
        st.markdown("### Detailed Data (Sample)")
        st.dataframe(df_grouped, use_container_width=True)
