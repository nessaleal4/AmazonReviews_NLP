import streamlit as st
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
import random

# Set the page configuration
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

# Load Qdrant secrets from Streamlit
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Qdrant client
q_client = get_qdrant_client()

# App header and description
st.title("Amazon Reviews Sentiment Analytics")
st.markdown("""
This dashboard provides analytics on sentiment distribution across Amazon product reviews.
Use the filters below to explore different categories of products.
""")

# Define the categories from your preprocessing script
CATEGORIES = ["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"]

# Sidebar for category selection
st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox(
    "Product Category", 
    ["All"] + CATEGORIES
)

# Run analytics button
run_button = st.sidebar.button("Run Analytics")

# Main content
if run_button:
    with st.spinner("Fetching data from vector database..."):
        try:
            # Get all samples first without filters
            # Create a query using random vectors for diverse results
            random_vector = [random.uniform(-1, 1) for _ in range(768)]
            
            # Simple search with minimal parameters
            results = q_client.search(
                collection_name="amazon_reviews",
                query_vector=random_vector,
                limit=500,
                with_payload=True
            )
            
            # Process results
            data = []
            for hit in results:
                if not hasattr(hit, 'payload') or not hit.payload:
                    continue
                    
                payload = hit.payload
                category = payload.get("category", "Unknown")
                
                # Skip if not in selected category (filter client-side)
                if selected_category != "All" and category != selected_category:
                    continue
                    
                data.append({
                    "Review Text": payload.get("text", "N/A")[:100] + "..." if len(payload.get("text", "")) > 100 else payload.get("text", "N/A"),
                    "Sentiment": payload.get("sentiment", "N/A"),
                    "Category": category
                })
            
            if not data:
                st.warning(f"No reviews found for category: {selected_category}")
                st.stop()
                
            df_results = pd.DataFrame(data)
            
            # Display summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Reviews Analyzed", len(df_results))
            
            with col2:
                if "Sentiment" in df_results.columns:
                    # Check the actual values in the sentiment column
                    unique_sentiments = df_results["Sentiment"].unique()
                    st.text(f"Sentiment values found: {', '.join(unique_sentiments)}")
                    
                    # Try to calculate positive percentage
                    for positive_label in ["POSITIVE", "Positive", "positive"]:
                        if positive_label in unique_sentiments:
                            positive_pct = (df_results["Sentiment"] == positive_label).mean() * 100
                            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
                            break
            
            # Create visualizations
            st.markdown("### Sentiment Distribution")
            
            # Normalize sentiment values for consistency
            df_results["Sentiment"] = df_results["Sentiment"].apply(
                lambda x: "Positive" if x.upper() == "POSITIVE" else "Negative" if x.upper() == "NEGATIVE" else x
            )
            
            # 1. Sentiment breakdown
            if "Sentiment" in df_results.columns:
                sentiment_counts = df_results["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                
                fig1 = px.pie(sentiment_counts, 
                            values="Count", 
                            names="Sentiment",
                            title="Overall Sentiment Distribution",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#2ecc71",
                                "Negative": "#e74c3c"
                            })
                st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Sentiment by category if multiple categories are present and we're showing "All"
            if selected_category == "All" and "Category" in df_results.columns and df_results["Category"].nunique() > 1:
                category_sentiment = df_results.groupby(["Category", "Sentiment"]).size().reset_index()
                category_sentiment.columns = ["Category", "Sentiment", "Count"]
                
                fig2 = px.bar(category_sentiment,
                            x="Category",
                            y="Count",
                            color="Sentiment",
                            barmode="group",
                            title="Sentiment Distribution by Category",
                            color_discrete_map={
                                "Positive": "#2ecc71",
                                "Negative": "#e74c3c"
                            })
                st.plotly_chart(fig2, use_container_width=True)
            
            # Display sample reviews
            st.markdown("### Sample Reviews")
            st.dataframe(df_results, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            st.info("This might be due to connectivity issues or changes in the database structure.")
else:
    st.info("Select a category and click 'Run Analytics' to view sentiment analysis.")

# Add key insights at the bottom
st.markdown("---")
st.markdown("""
### Key Insights

- Sentiment analysis can help identify product categories with higher customer satisfaction
- Analyzing negative reviews can highlight areas for improvement
- Compare sentiment across different product categories to identify trends
""")
