import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load secrets from Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

def main():
    st.title("Amazon Reviews Search")
    query = st.text_input("Enter your search query:", "great budget phone")
    
    if st.button("Search"):
        if query.strip():
            embedder = load_embedder()
            qdrant_client = get_qdrant_client()
            
            # Convert query to embedding
            query_vector = embedder.encode([query])[0].tolist()
            
            # Perform similarity search
            results = qdrant_client.search(
                collection_name="amazon_reviews",
                query_vector=query_vector,
                limit=10,
                with_payload=True
            )
            
            st.write("### Results:")
            for i, hit in enumerate(results):
                payload = hit.payload
                st.write(f"**Result {i+1}:**")
                st.write(f"**Review:** {payload.get('text', 'N/A')}")
                st.write(f"**Sentiment:** {payload.get('sentiment', 'N/A')}")
                st.write(f"**Category:** {payload.get('category', 'N/A')}")
                st.markdown("---")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
