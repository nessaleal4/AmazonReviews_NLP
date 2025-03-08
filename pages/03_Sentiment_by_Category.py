import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Set page config for this page (optional; set in app.py is sufficient)
st.set_page_config(page_title="Sentiment by Category", layout="wide")

# Load Qdrant secrets from Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

q_client = get_qdrant_client()
embedder = load_embedder()

st.title("Sentiment by Category")
st.markdown("""
This page shows a visualization of the sentiment distribution across review categories.
You can either enter a query to filter the reviews or view the overall distribution.
""")

query = st.text_input("Optional: Enter a query to filter reviews:", "")

# We'll fetch a larger number of results (e.g. top 100) from Qdrant.
if query.strip():
    query_vector = embedder.encode([query])[0].tolist()
    results = q_client.search(
        collection_name="amazon_reviews",
        query_vector=query_vector,
        limit=100,
        with_payload=True
    )
    data = []
    for hit in results:
        payload = hit.payload
        data.append({
            "Category": payload.get("category", "N/A"),
            "Sentiment": payload.get("sentiment", "N/A")
        })
    df = pd.DataFrame(data)
else:
    # If no query is provided, you could either:
    # a) Retrieve all reviews from Qdrant (if the collection is small), or
    # b) Load a pre-saved dataset.
    # Here, we'll assume you have a preprocessed CSV for demonstration.
    try:
        df = pd.read_csv("data/all_reviews.csv")
    except Exception:
        st.error("No query provided and no local data available.")
        df = pd.DataFrame()

if not df.empty:
    # Group by category and sentiment
    sentiment_by_category = (
        df.groupby(["Category", "Sentiment"])
          .size()
          .reset_index(name="Count")
    )
    fig = px.bar(
        sentiment_by_category,
        x="Category",
        y="Count",
        color="Sentiment",
        barmode="group",
        title="Sentiment Distribution by Category",
        labels={"Count": "Number of Reviews"}
    )
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data to display.")
