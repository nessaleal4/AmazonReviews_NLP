import streamlit as st
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient

# Set the page config (optional on sub-pages; best to have it in the main app)
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

# Load Qdrant secrets from Streamlit secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

q_client = get_qdrant_client()

@st.cache_data
def load_data_from_qdrant(sample_limit: int = 1000) -> pd.DataFrame:
    """
    Fetches a sample of points from the Qdrant collection and returns a DataFrame
    containing 'category' and 'sentiment' from the payload.
    """
    # Use the scroll method to fetch a batch of points
    scroll_response = q_client.scroll(
        collection_name="amazon_reviews",
        limit=sample_limit
    )
    points = scroll_response.result.points
    data = []
    for point in points:
        payload = point.payload
        data.append({
            "Category": payload.get("category", "Unknown"),
            "Sentiment": payload.get("sentiment", "Unknown")
        })
    return pd.DataFrame(data)

st.title("Sentiment Analytics by Category")
st.markdown("""
This dashboard displays sentiment distribution across review categories based on data stored in Qdrant.
""")

# Load a sample of the data from Qdrant
df = load_data_from_qdrant(1000)

if df.empty:
    st.write("No data available. Please ensure your Qdrant collection is populated.")
else:
    # Group by Category and Sentiment
    df_grouped = df.groupby(["Category", "Sentiment"]).size().reset_index(name="Count")
    
    # Create a grouped bar chart using Plotly Express
    fig = px.bar(
        df_grouped,
        x="Category",
        y="Count",
        color="Sentiment",
        barmode="group",
        title="Sentiment Distribution by Category",
        labels={"Count": "Number of Reviews"}
    )
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Optionally, show raw grouped data
    st.markdown("### Detailed Data")
    st.dataframe(df_grouped)
