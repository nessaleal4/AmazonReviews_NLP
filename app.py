import streamlit as st
import pandas as pd
import re  # still imported if you need to extend functionality later
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest  # For reference, if needed

# Load Qdrant secrets from Streamlit secrets
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

st.set_page_config(page_title="Amazon Reviews Search", layout="wide")
st.title("Amazon Reviews Search")

st.markdown("""
This app performs a vector search on preprocessed Amazon reviews stored in Qdrant.
Enter a query below to find similar reviews and see their sentiment distribution.
""")

query = st.text_input("Enter your search query:", "great budget phone")

if st.button("Search"):
    if query.strip():
        # Generate query embedding using MPNet
        query_vector = embedder.encode([query])[0].tolist()

        # Perform similarity search in Qdrant (retrieve top 10 results)
        results = q_client.search(
            collection_name="amazon_reviews",
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )

        # Extract results into a DataFrame for display
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
        
        # Display a bar chart for sentiment distribution if available
        if not df_results["Sentiment"].empty:
            sentiment_counts = df_results["Sentiment"].value_counts()
            st.markdown("### Sentiment Distribution")
            st.bar_chart(sentiment_counts)
    else:
        st.warning("Please enter a search query.")
