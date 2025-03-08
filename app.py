import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import base64
from PIL import Image
import io

# Set the page configuration as the very first Streamlit command
st.set_page_config(
    page_title="Amazon Reviews Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main page styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header and title styling */
    h1 {
        color: #232F3E;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2, h3 {
        color: #232F3E;
        font-weight: 600 !important;
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #8B0000;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #6B0000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Card-like styling for sections */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        background-color: white;
        margin-bottom: 1.5rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-family: 'Arial', sans-serif;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sentiment colors */
    .positive {
        color: #2ecc71;
        font-weight: bold;
    }
    
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #666;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    
    /* For dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: #1E1E1E;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
    }
</style>
""", unsafe_allow_html=True)

# Customize the sidebar title and styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="color: #232F3E;">Amazon Reviews</h2>
    <p style="font-size: 1.1rem; color: #8B0000; font-weight: 600;">Product Search & Analysis</p>
</div>
""", unsafe_allow_html=True)

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

# Create a visually appealing header with icon and title
col1, col2 = st.columns([1, 6])
with col1:
    # Elegant icon
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <span style="font-size: 3.5rem; color: #8B0000;">📊</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.title("Product Sentiment Search - Amazon Reviews")

# More elegant app description
st.markdown("""
<div class="card">
    <p style="font-size: 1.1rem; line-height: 1.6;">
        Welcome to the <b>Product Sentiment Search - Amazon Reviews</b> platform. This tool leverages natural language processing and 
        semantic search to help you explore customer sentiments about products. Enter a search query 
        below to discover relevant reviews and analyze their sentiment distribution.
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced search input with example queries
st.markdown("<h3>Search for Product Reviews</h3>", unsafe_allow_html=True)

# Simple elegant search input
query = st.text_input("Enter your search query:", "Crock Pot", 
                      help="Try searching for product features, specific use cases, or quality descriptors")

search_col1, search_col2 = st.columns([1, 6])
with search_col1:
    search_button = st.button("Search", use_container_width=True)
    
# Results section
if search_button:
    if query.strip():
        with st.spinner("Searching for relevant reviews..."):
            # Generate embedding for the query using MPNet
            query_vector = embedder.encode([query])[0].tolist()
            
            # Perform similarity search in Qdrant
            results = q_client.search(
                collection_name="amazon_reviews",
                query_vector=query_vector,
                limit=10,
                with_payload=True
            )
            
            # Format the results with better styling
            if results:
                # Extract search results into a DataFrame for display
                data = []
                for hit in results:
                    payload = hit.payload
                    sentiment = payload.get("sentiment", "N/A")
                    sentiment_class = "positive" if sentiment == "POSITIVE" else "negative" if sentiment == "NEGATIVE" else ""
                    
                    data.append({
                        "Review Text": payload.get("text", "N/A")[:300] + "..." if len(payload.get("text", "")) > 300 else payload.get("text", "N/A"),
                        "Sentiment": sentiment,
                        "Category": payload.get("category", "N/A"),
                        "Sentiment_Class": sentiment_class  # For styling
                    })
                df_results = pd.DataFrame(data)
                
                # Results header with search query highlight
                st.markdown(f"""
                <div class="card">
                    <h3>Search Results for: <span style="color: #8B0000;">"{query}"</span></h3>
                    <p>Showing {len(results)} most relevant reviews based on semantic similarity.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations and results in tabs for better organization
                tab1, tab2 = st.tabs(["📊 Sentiment Analysis", "📝 Review Details"])
                
                with tab1:
                    # If sentiment data is available, create enhanced visualizations
                    if not df_results["Sentiment"].empty:
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # More visually appealing bar chart
                            sentiment_counts = df_results["Sentiment"].value_counts().reset_index()
                            sentiment_counts.columns = ["Sentiment", "Count"]
                            
                            # Elegant color palette
                            colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#8B0000", "NEUTRAL": "#3498db"}
                            
                            fig = px.bar(
                                sentiment_counts, 
                                x="Sentiment", 
                                y="Count",
                                color="Sentiment",
                                text="Count",
                                labels={"Sentiment": "Sentiment", "Count": "Review Count"},
                                title="Sentiment Distribution in Search Results",
                                color_discrete_map=colors
                            )
                            
                            fig.update_traces(
                                texttemplate='%{text}', 
                                textposition='outside',
                                marker_line_width=0,
                                width=0.6
                            )
                            
                            fig.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=14),
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Add a pie chart for better visualization
                            sentiment_pie = px.pie(
                                sentiment_counts, 
                                values="Count", 
                                names="Sentiment",
                                title="Sentiment Proportion",
                                color="Sentiment",
                                color_discrete_map=colors,
                                hole=0.4
                            )
                            
                            sentiment_pie.update_layout(
                                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                                margin=dict(l=20, r=20, t=40, b=0),
                                height=330
                            )
                            
                            sentiment_pie.update_traces(
                                textinfo="percent+label", 
                                pull=[0.05, 0, 0],
                                marker_line_width=2,
                                marker_line_color="white"
                            )
                            
                            st.plotly_chart(sentiment_pie, use_container_width=True)
                        
                        # Quick insights based on the data
                        st.markdown("""
                        <div class="card" style="background-color: #f8f9fa;">
                            <h4>📈 Quick Insights</h4>
                            <ul>
                                <li>The sentiment analysis shows the emotional tone of reviews matching your search.</li>
                                <li>Positive reviews often highlight product strengths and customer satisfaction.</li>
                                <li>Negative reviews can provide valuable feedback for potential concerns.</li>
                                <li>Consider reading reviews from both sentiment categories for a balanced perspective.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    # Enhance dataframe display with custom formatting for sentiment
                    def color_sentiment(val):
                        color = "#2ecc71" if val == "POSITIVE" else "#e74c3c" if val == "NEGATIVE" else "#7f8c8d"
                        weight = "bold"
                        return f'background-color: {color}25; color: {color}; font-weight: {weight}'
                    
                    # Apply the styling
                    styled_df = df_results[["Review Text", "Sentiment", "Category"]].style.applymap(
                        color_sentiment, subset=["Sentiment"]
                    )
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Download option for results
                    csv = df_results[["Review Text", "Sentiment", "Category"]].to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<div style="text-align: right;"><a href="data:file/csv;base64,{b64}" download="amazon_reviews_results.csv" style="color: #8B0000; text-decoration: none; font-weight: 600;">📥 Download Results</a></div>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No matching reviews found for your query. Try a different search term.")
    else:
        st.warning("Please enter a search query.")

# Add a footer with additional information
st.markdown("""
<div class="footer">
    <p>Powered by NLP and vector search technology. Data sourced from the Amazon Review Data (2023) dataset.</p>
</div>
""", unsafe_allow_html=True)

# Rename pages in sidebar
if "pages" not in st.session_state:
    st.session_state["pages"] = True
    import os
    import re
    
    # Function to rename a page in the sidebar
    def rename_page(from_name, to_name):
        pages_dir = "pages"
        if os.path.isdir(pages_dir):
            for filename in os.listdir(pages_dir):
                if filename.endswith(".py"):
                    original_path = os.path.join(pages_dir, filename)
                    with open(original_path, "r") as f:
                        content = f.read()
                    
                    # Check if this is the page we want to rename
                    if from_name.lower() in filename.lower():
                        # Modify the content to include the title configuration
                        page_title_pattern = r"st\.set_page_config\([^)]*title\s*=\s*\"[^\"]*\""
                        page_title_replacement = f"st.set_page_config(page_title=\"{to_name}\""
                        
                        if re.search(page_title_pattern, content):
                            content = re.sub(page_title_pattern, page_title_replacement, content)
                        else:
                            # If no existing set_page_config, add it at the beginning after imports
                            content = f'import streamlit as st\nst.set_page_config(page_title="{to_name}", page_icon="📈")\n' + \
                                     '\n'.join(content.split('\n')[1:])
                        
                        # Save the modified content
                        with open(original_path, "w") as f:
                            f.write(content)
    
    # Rename the sentiment analytics page
    rename_page("SentimentAnalytics", "Sentiment Analysis by Category")
