import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
from io import StringIO
import time

# Set the page configuration
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

# App header and description
st.title("Amazon Reviews Sentiment Analytics")
st.markdown("""
This dashboard provides analytics on sentiment distribution across Amazon product reviews.
Select a category and run the analysis to explore sentiment patterns.
""")

# Define the categories and their Dropbox links
CATEGORY_LINKS = {
    "Beauty_and_Personal_Care": "https://www.dropbox.com/s/is7gjqxtuu7pxct/Beauty_and_Personal_Care_sample.csv?st=v20gmuuf&dl=1",
    "Books": "https://www.dropbox.com/s/w35dhp5k7r27lkr/Books_sample.csv?st=iuq9zczq&dl=1",
    "Electronics": "https://www.dropbox.com/s/bopycg75dfxbyba/Electronics_sample.csv?st=zcs64jv3&dl=1",
    "Home_and_Kitchen": "https://www.dropbox.com/s/fzt9u5t0gkx3dfh/Home_and_Kitchen_sample.csv?st=7u6ufnk1&dl=1"
}

# Sidebar for controls
st.sidebar.header("Analysis Settings")

# Category selection
selected_category = st.sidebar.selectbox(
    "Select Category to Analyze", 
    ["All"] + list(CATEGORY_LINKS.keys())
)

# Sample size slider
sample_size = st.sidebar.slider(
    "Sample Size per Category", 
    min_value=100, 
    max_value=5000,
    value=1000,
    step=100
)

# Cache directory for downloaded files
CACHE_DIR = ".st_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Helper function to download and cache files
@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_csv_from_dropbox(category, url, sample_size=1000):
    """Download a CSV file from Dropbox, cache it, and return as DataFrame"""
    cache_path = os.path.join(CACHE_DIR, f"{category}_{sample_size}.csv")
    
    # Check if we have a cached version
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, nrows=sample_size)
        except Exception:
            # If cached file is corrupted, delete it and download again
            os.remove(cache_path)
    
    try:
        with st.spinner(f"Downloading {category} data..."):
            # Download the file from Dropbox
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                st.error(f"Failed to download {category} data. Status code: {response.status_code}")
                return pd.DataFrame()
            
            # Parse CSV directly from the response content
            df = pd.read_csv(StringIO(response.content.decode('utf-8')), nrows=sample_size)
            
            # Add category column if not present
            if 'category' not in df.columns:
                df['category'] = category
                
            return df
            
    except Exception as e:
        st.error(f"Error downloading {category} data: {e}")
        return pd.DataFrame()

# Run analysis button
run_button = st.sidebar.button("Run Analysis")

# Main analysis function
def run_sentiment_analysis(category, sample_size):
    """Run sentiment analysis for the selected category"""
    start_time = time.time()
    
    try:
        # Get data based on category selection
        if category == "All":
            all_data = []
            # Use progress bar for multiple categories
            progress_bar = st.progress(0)
            
            for i, (cat, url) in enumerate(CATEGORY_LINKS.items()):
                df = download_csv_from_dropbox(cat, url, sample_size=sample_size // len(CATEGORY_LINKS))
                if not df.empty:
                    all_data.append(df)
                progress_bar.progress((i + 1) / len(CATEGORY_LINKS))
            
            progress_bar.empty()
            
            if all_data:
                df_results = pd.concat(all_data, ignore_index=True)
                st.success(f"Successfully loaded data from {len(all_data)} categories")
            else:
                st.warning("No data was loaded. Please check your internet connection or try again later.")
                return
        else:
            # Process single category
            url = CATEGORY_LINKS.get(category)
            if not url:
                st.error(f"No download link found for {category}")
                return
                
            df_results = download_csv_from_dropbox(category, url, sample_size)
            if df_results.empty:
                st.warning(f"No data was loaded for {category}. Please try again later.")
                return
        
        # Display timing information
        loading_time = time.time() - start_time
        st.info(f"Data loaded in {loading_time:.2f} seconds. Analyzing {len(df_results)} reviews.")
            
        # Make sure we have the necessary columns
        required_cols = ['text', 'sentiment']
        if not all(col in df_results.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_results.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.write("Available columns:", df_results.columns.tolist())
            return
        
        # Analysis starts here
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews Analyzed", len(df_results))
        
        with col2:
            # Normalize sentiment values to handle different formats
            df_results['sentiment'] = df_results['sentiment'].astype(str).str.upper()
            positive_pct = (df_results['sentiment'].str.contains('POSITIVE')).mean() * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        
        with col3:
            if 'rating' in df_results.columns:
                try:
                    avg_rating = df_results['rating'].astype(float).mean()
                    st.metric("Average Rating", f"{avg_rating:.1f}")
                except:
                    st.metric("Average Rating", "N/A")
        
        # Create visualizations
        st.markdown("### Sentiment Distribution")
        
        # Normalize sentiment for visualization
        df_results['sentiment_normalized'] = df_results['sentiment'].apply(
            lambda x: "Positive" if "POSITIVE" in x.upper() else "Negative" if "NEGATIVE" in x.upper() else "Neutral"
        )
        
        # 1. Sentiment breakdown
        sentiment_counts = df_results['sentiment_normalized'].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        fig1 = px.pie(sentiment_counts, 
                    values="Count", 
                    names="Sentiment",
                    title="Overall Sentiment Distribution",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral": "#3498db"
                    })
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Sentiment by category if multiple categories
        if 'category' in df_results.columns and df_results['category'].nunique() > 1:
            category_sentiment = df_results.groupby(['category', 'sentiment_normalized']).size().reset_index()
            category_sentiment.columns = ["Category", "Sentiment", "Count"]
            
            fig2 = px.bar(category_sentiment,
                        x="Category",
                        y="Count",
                        color="Sentiment",
                        barmode="group",
                        title="Sentiment Distribution by Category",
                        color_discrete_map={
                            "Positive": "#2ecc71",
                            "Negative": "#e74c3c",
                            "Neutral": "#3498db"
                        })
            st.plotly_chart(fig2, use_container_width=True)
        
        # 3. If rating column exists, show rating distribution
        if 'rating' in df_results.columns:
            try:
                df_results['rating'] = pd.to_numeric(df_results['rating'], errors='coerce')
                rating_counts = df_results['rating'].value_counts().sort_index().reset_index()
                rating_counts.columns = ["Rating", "Count"]
                
                fig3 = px.bar(rating_counts,
                            x="Rating",
                            y="Count",
                            title="Rating Distribution",
                            color="Rating",
                            color_continuous_scale="Viridis")
                st.plotly_chart(fig3, use_container_width=True)
                
                # 4. Rating vs Sentiment
                rating_sentiment = df_results.groupby(['rating', 'sentiment_normalized']).size().reset_index()
                rating_sentiment.columns = ["Rating", "Sentiment", "Count"]
                
                fig4 = px.bar(rating_sentiment,
                            x="Rating",
                            y="Count",
                            color="Sentiment",
                            barmode="stack",
                            title="Sentiment Distribution by Rating",
                            color_discrete_map={
                                "Positive": "#2ecc71",
                                "Negative": "#e74c3c",
                                "Neutral": "#3498db"
                            })
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not process rating data for visualization: {e}")
        
        # Display sample reviews
        st.markdown("### Sample Reviews")
        
        # Create tabs for positive and negative reviews
        tab1, tab2 = st.tabs(["Positive Reviews", "Negative Reviews"])
        
        with tab1:
            positive_df = df_results[df_results['sentiment_normalized'] == "Positive"].sample(
                min(20, (df_results['sentiment_normalized'] == "Positive").sum())
            ).reset_index(drop=True)
            if not positive_df.empty:
                display_cols = ['text', 'sentiment']
                if 'rating' in positive_df.columns:
                    display_cols.append('rating')
                if 'category' in positive_df.columns and positive_df['category'].nunique() > 1:
                    display_cols.append('category')
                    
                st.dataframe(positive_df[display_cols], use_container_width=True)
            else:
                st.info("No positive reviews found.")
        
        with tab2:
            negative_df = df_results[df_results['sentiment_normalized'] == "Negative"].sample(
                min(20, (df_results['sentiment_normalized'] == "Negative").sum())
            ).reset_index(drop=True)
            if not negative_df.empty:
                display_cols = ['text', 'sentiment']
                if 'rating' in negative_df.columns:
                    display_cols.append('rating')
                if 'category' in negative_df.columns and negative_df['category'].nunique() > 1:
                    display_cols.append('category')
                    
                st.dataframe(negative_df[display_cols], use_container_width=True)
            else:
                st.info("No negative reviews found.")
                
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.info("This might be due to issues with data download or processing.")

# Main app logic
if run_button:
    with st.spinner("Running sentiment analysis..."):
        run_sentiment_analysis(selected_category, sample_size)
else:
    # Default welcome screen
    st.info("👈 Select a category and click 'Run Analysis' to start exploring sentiment patterns.")
    
    # Show preview of what's available
    categories_preview = px.bar(
        x=list(CATEGORY_LINKS.keys()),
        y=[1, 1, 1, 1],  # Placeholder values
        title="Available Categories for Analysis",
        labels={"x": "Category", "y": ""},
        color=list(CATEGORY_LINKS.keys()),
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    categories_preview.update_layout(showlegend=False)
    st.plotly_chart(categories_preview, use_container_width=True)
    
    # Tips for best experience
    st.markdown("""
    ### Tips for Best Results
    
    - Start with a single category to analyze specific product types
    - Use "All" to compare sentiment across different categories
    - Adjust the sample size based on your needs (larger samples take longer to process)
    - The analysis is cached for faster repeated runs
    """)

# Add key insights at the bottom
st.markdown("---")
st.markdown("""
### Key Insights from Sentiment Analysis

- **Positive reviews** often highlight product strengths and customer satisfaction
- **Negative reviews** provide valuable feedback for improvements
- **Rating distribution** shows the overall customer satisfaction level
- **Category comparisons** reveal which product types perform better in customer sentiment
""")
