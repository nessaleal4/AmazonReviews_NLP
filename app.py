import streamlit as st
import pandas as pd
import re
import plotly.express as px
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest  # For reference

# Set the page configuration as the very first Streamlit command
st.set_page_config(page_title="Amazon Reviews Search", layout="wide")

# Load Qdrant secrets from Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

# Initialize embedder and Qdrant client
embedder = load_embedder()
q_client = get_qdrant_client()

# App header and description
st.title("Amazon Reviews Search")
st.markdown("""
Welcome to the Amazon Reviews Search app.
Enter a search query below to retrieve similar reviews and view their sentiment distribution.
""")

query = st.text_input("Enter your search query:", "Harry Potter")

if st.button("Search"):
    if query.strip():
        # Generate embedding for the query using MPNet
        query_vector = embedder.encode([query])[0].tolist()

        # Perform similarity search in Qdrant (retrieve top 10 results)
        results = q_client.search(
            collection_name="amazon_reviews",
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )

        # Extract search results into a DataFrame for display
        data = []
        for hit in results:
            payload = hit.payload
            data.append({
                "Review Text": payload.get("text", "N/A"),
                "Sentiment": payload.get("sentiment", "N/A"),
                "Category": payload.get("category", "N/A")
            })
        df_results = pd.DataFrame(data)

        st.markdown("### Search Results")
        st.dataframe(df_results, use_container_width=True)

        # If sentiment data is available, create a more polished bar chart using Plotly
        if not df_results["Sentiment"].empty:
            sentiment_counts = df_results["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.bar(sentiment_counts, 
                         x="Sentiment", 
                         y="Count",
                         color="Sentiment",
                         text="Count",
                         labels={"Sentiment": "Sentiment", "Count": "Review Count"},
                         title="Sentiment Distribution in Search Results")
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please enter a search query.")
