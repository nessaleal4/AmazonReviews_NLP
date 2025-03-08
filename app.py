import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest

# Initialize the embedding model (make sure this matches the model used for offline processing)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

embedder = load_embedder()

# Connect to Qdrant Cloud
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url="https://8294e264-e739-44ca-ab59-0aff628d5f01.us-east-1-0.aws.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzcyNjY0MTY1fQ.D6P0Uh4asPS0CPFvl53iEVvxw9bBneRy-wI6yvwF2NY"
    )

q_client = get_qdrant_client()

st.title("Amazon Reviews Search")

# Input box for the query
query = st.text_input("Enter your search query:", value="great budget phone")

if st.button("Search"):
    if query:
        # Generate an embedding for the query
        query_vector = embedder.encode([query])[0].tolist()
        
        # Perform a similarity search in Qdrant (returns top 10 matches)
        results = q_client.search(
            collection_name="amazon_reviews",
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )
        
        st.write("### Results:")
        for idx, hit in enumerate(results):
            payload = hit.payload
            st.write(f"**Result {idx+1}:**")
            st.write(f"**Review:** {payload.get('text', 'N/A')}")
            st.write(f"**Sentiment:** {payload.get('sentiment', 'N/A')}")
            st.write(f"**Category:** {payload.get('category', 'N/A')}")
            st.markdown("---")
    else:
        st.warning("Please enter a query.")
