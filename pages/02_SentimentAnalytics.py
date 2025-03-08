import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
from bs4 import BeautifulSoup

st.set_page_config(page_title="Sentiment Analytics", layout="wide")

st.title("Sentiment Analytics by Category")
st.markdown("""
This dashboard displays sentiment distribution across review categories based on CSV data stored on Google Drive.
HTML tags in the review text are stripped for cleaner display.
""")

DRIVE_FILE_IDS = {
    "Beauty_and_Personal_Care": "YOUR_FILE_ID",
    "Books": "YOUR_FILE_ID",
    "Electronics": "YOUR_FILE_ID",
    "Home_and_Kitchen": "YOUR_FILE_ID"
}

def strip_html(raw_html: str) -> str:
    if not isinstance(raw_html, str):
        return str(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ")

@st.cache_data
def load_csv_from_drive(file_id: str) -> pd.DataFrame:
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), engine="python", on_bad_lines="skip")
    # Lowercase & strip columns
    df.columns = df.columns.str.lower().str.strip()
    # Strip HTML from text column if present
    if "text" in df.columns:
        df["text"] = df["text"].apply(strip_html)
    return df

@st.cache_data
def load_all_categories() -> pd.DataFrame:
    all_dfs = []
    for category, file_id in DRIVE_FILE_IDS.items():
        try:
            df = load_csv_from_drive(file_id)
            if "category" not in df.columns:
                df["category"] = category
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading data for {category}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

df_all = load_all_categories()

if df_all.empty:
    st.warning("No data available. Please ensure your Google Drive links are correct.")
else:
    # Check if 'sentiment' column is present
    if "sentiment" not in df_all.columns:
        st.error("The data does not include a 'sentiment' column.")
    else:
        # Group data by category and sentiment
        df_grouped = df_all.groupby(["category", "sentiment"]).size().reset_index(name="count")

        # Plot
        fig = px.bar(
            df_grouped,
            x="category",
            y="count",
            color="sentiment",
            barmode="group",
            title="Sentiment Distribution by Category",
            labels={"category": "Category", "count": "Number of Reviews"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Sample Data")
        # Optionally hide "text" column if it's huge
        columns_to_show = [col for col in df_all.columns if col not in ["text", "embedding"]]
        st.dataframe(df_all[columns_to_show].head(50))
