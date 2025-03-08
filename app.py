import streamlit as st
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest  # not used directly, but available if needed

# Load Qdrant secrets from Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

# Initialize models
embedder = load_embedder()
nlp = load_spacy()
q_client = get_qdrant_client()

def extract_product_name(text):
    """
    Attempts to extract a product name from the review text using spaCy NER.
    Looks for entities labeled as 'PRODUCT' or 'ORG' (as a heuristic).
    """
    doc = nlp(text)
    candidates = [ent.text for ent in doc.ents if ent.label_ in ("PRODUCT", "ORG")]
    return candidates[0] if candidates else "Unknown Product"

st.title("Amazon Reviews Search")

query = st.text_input("Enter your search query:", "great budget phone")

if st.button("Search"):
    if query.strip():
        # Generate embedding for the query using MPNet
        query_vector = embedder.encode([query])[0].tolist()

        # Perform similarity search in Qdrant (top 10 results)
        results = q_client.search(
            collection_name="amazon_reviews",
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )

        # Extract search results into a DataFrame for visualization
        data = []
        for hit in results:
            payload = hit.payload
            text = payload.get("text", "N/A")
            sentiment = payload.get("sentiment", "N/A")
            category = payload.get("category", "N/A")
            product = extract_product_name(text)
            data.append({
                "text": text,
                "sentiment": sentiment,
                "category": category,
                "product": product
            })
        
        df_results = pd.DataFrame(data)
        
        st.write("### Search Results")
        st.dataframe(df_results)
        
        # Display a bar chart for sentiment distribution
        sentiment_counts = df_results["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Optionally, display the guessed product names with the associated text
        st.write("### Guessed Product Names")
        st.dataframe(df_results[["product", "text"]])
    else:
        st.warning("Please enter a query.")
