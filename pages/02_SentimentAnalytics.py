import streamlit as st
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient

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
Use the filters below to explore different aspects of the data.
""")

# Sidebar for filters
st.sidebar.header("Filters")

# Category filter - Using scroll points from the vector DB instead of loading full dataset
@st.cache_data(ttl=3600)
def get_categories():
    # Use scroll to efficiently get a sample of unique categories
    # This avoids loading the entire dataset
    results = q_client.scroll(
        collection_name="amazon_reviews",
        limit=1000,  # Adjust based on your collection size
        with_payload=True,
        scroll_filter=None  # No filter to get diverse samples
    )
    
    if results and results[0]:
        # Extract unique categories from the sample
        categories = set()
        for hit in results[0]:
            if hit.payload and "category" in hit.payload:
                categories.add(hit.payload["category"])
        return sorted(list(categories))
    return []

categories = get_categories()
selected_category = st.sidebar.selectbox(
    "Product Category", 
    ["All"] + categories
)

# Rating filter (if available in your vector DB)
rating_options = ["All", "5 stars", "4 stars", "3 stars", "2 stars", "1 star"]
selected_rating = st.sidebar.selectbox("Rating", rating_options)

# Search for specific term in reviews
search_term = st.sidebar.text_input("Search for specific term in reviews")

# Run analytics button
run_button = st.sidebar.button("Run Analytics")

# Main content
if run_button:
    # Prepare filters for Qdrant query
    filters = []
    
    if selected_category != "All":
        filters.append({
            "field": "category",
            "match": {"value": selected_category}
        })
    
    if selected_rating != "All":
        # Extract numeric value from rating string
        rating_value = int(selected_rating.split()[0])
        filters.append({
            "field": "rating",
            "match": {"value": rating_value}
        })
    
    # Combine filters
    filter_query = None
    if filters:
        if len(filters) == 1:
            filter_query = filters[0]
        else:
            filter_query = {"must": filters}
    
    # If there's a search term, use hybrid search (vector + keyword)
    if search_term.strip():
        st.info(f"Searching for reviews containing: '{search_term}'")
        
        # For keyword search in vector DB, there are a few approaches:
        # 1. Use hybrid search if your Qdrant version supports it
        # 2. Add a filter for keyword match if payload is indexed
        # 3. Post-filter results from vector search
        
        # Here's a simple approach - we'll get more results and filter in Python
        # This avoids complex vector DB setup but still works with existing data
        
        results = q_client.scroll(
            collection_name="amazon_reviews",
            limit=1000,  # Get a good sample to filter from
            with_payload=True,
            scroll_filter=filter_query
        )
        
        # Post-filter for keyword match
        filtered_results = []
        if results and results[0]:
            for hit in results[0]:
                if hit.payload and "text" in hit.payload:
                    if search_term.lower() in hit.payload["text"].lower():
                        filtered_results.append(hit)
        
        data = []
        for hit in filtered_results[:500]:  # Limit to 500 to avoid processing too much
            payload = hit.payload
            data.append({
                "Review Text": payload.get("text", "N/A")[:100] + "...",  # Truncate for display
                "Sentiment": payload.get("sentiment", "N/A"),
                "Category": payload.get("category", "N/A"),
                "Rating": payload.get("rating", "N/A")
            })
    else:
        # Regular scroll query without keyword search
        results = q_client.scroll(
            collection_name="amazon_reviews",
            limit=500,  # Limit to avoid memory issues
            with_payload=True,
            scroll_filter=filter_query
        )
        
        data = []
        if results and results[0]:
            for hit in results[0]:
                payload = hit.payload
                data.append({
                    "Review Text": payload.get("text", "N/A")[:100] + "...",  # Truncate for display
                    "Sentiment": payload.get("sentiment", "N/A"),
                    "Category": payload.get("category", "N/A"),
                    "Rating": payload.get("rating", "N/A")
                })
    
    # Create DataFrame from results
    df_results = pd.DataFrame(data)
    
    if not df_results.empty:
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df_results))
        
        with col2:
            if "Sentiment" in df_results.columns:
                positive_pct = (df_results["Sentiment"] == "Positive").mean() * 100
                st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        
        with col3:
            if "Rating" in df_results.columns and df_results["Rating"].notna().any():
                avg_rating = df_results["Rating"].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}")
        
        # Create visualizations
        st.markdown("### Sentiment Distribution")
        
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
                             "Neutral": "#3498db",
                             "Negative": "#e74c3c"
                         })
            st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Sentiment by category if multiple categories are present
        if "Category" in df_results.columns and df_results["Category"].nunique() > 1:
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
                             "Neutral": "#3498db",
                             "Negative": "#e74c3c"
                         })
            st.plotly_chart(fig2, use_container_width=True)
        
        # 3. Top keywords by sentiment (if NLTK is installed)
        try:
            import nltk
            from nltk.corpus import stopwords
            
            # Download NLTK resources if not already present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            # Extract keywords
            st.markdown("### Top Keywords by Sentiment")
            
            def extract_keywords(text_series, sentiment_filter):
                # Filter texts by sentiment
                filtered_texts = text_series[df_results["Sentiment"] == sentiment_filter]
                
                if filtered_texts.empty:
                    return pd.DataFrame(columns=["Keyword", "Count"])
                
                # Combine all texts
                all_text = " ".join(filtered_texts.astype(str))
                
                # Tokenize and remove stopwords
                stop_words = set(stopwords.words('english'))
                words = [word.lower() for word in all_text.split() 
                         if word.isalpha() and word.lower() not in stop_words and len(word) > 3]
                
                # Count word frequencies
                word_counts = pd.Series(words).value_counts().reset_index()
                word_counts.columns = ["Keyword", "Count"]
                
                return word_counts.head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Positive Review Keywords")
                positive_keywords = extract_keywords(df_results["Review Text"], "Positive")
                if not positive_keywords.empty:
                    fig_pos = px.bar(positive_keywords, x="Count", y="Keyword", orientation='h',
                                     title="Top Keywords in Positive Reviews",
                                     color_discrete_sequence=["#2ecc71"])
                    st.plotly_chart(fig_pos, use_container_width=True)
            
            with col2:
                st.subheader("Negative Review Keywords")
                negative_keywords = extract_keywords(df_results["Review Text"], "Negative")
                if not negative_keywords.empty:
                    fig_neg = px.bar(negative_keywords, x="Count", y="Keyword", orientation='h',
                                     title="Top Keywords in Negative Reviews",
                                     color_discrete_sequence=["#e74c3c"])
                    st.plotly_chart(fig_neg, use_container_width=True)
        
        except ImportError:
            st.info("Install NLTK to enable keyword extraction functionality")
        
        # Display sample reviews
        st.markdown("### Sample Reviews")
        st.dataframe(df_results, use_container_width=True)
    
    else:
        st.warning("No reviews match the selected filters.")
else:
    st.info("Use the filters in the sidebar and click 'Run Analytics' to view sentiment analysis.")

# Add helpful tips at the bottom
st.markdown("---")
st.markdown("""
### Tips for Analysis
- Try filtering by different product categories to compare sentiment patterns
- Search for specific product features (e.g., "battery", "durability", "price") to see targeted feedback
- Analyze negative reviews to identify common customer pain points
""")
