# AmazonReviews_NLP

A sophisticated natural language processing application that analyzes Amazon product reviews, providing semantic search capabilities and elegant sentiment analysis visualization.

## Overview

This project leverages advanced semantic search and sentiment analysis to help users explore Amazon product reviews. It allows users to search for reviews using natural language queries and visualizes the sentiment distribution with elegant, interactive charts.

The application:
- Embeds reviews using sentence transformers (all-mpnet-base-v2)
- Stores and retrieves vectors using Qdrant vector database
- Provides a polished, user-friendly interface built with Streamlit
- Visualizes sentiment distribution with professional Plotly charts

![AmazonReviews_NLP Interface](/assets/app_preview.png)

## Project Structure

```
AmazonReviews_NLP/
├── .devcontainer/
│   └── devcontainer.json     # Development container configuration
├── pages/
│   └── 02_SentimentAnalytics.py  # Analytics dashboard for sentiment analysis by category
├── app.py                    # Main application file for the product search interface
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Data Source

This project uses the Amazon Review Data (2023) dataset available at [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/).

The dataset includes:

### Review Data
Each category is stored as a compressed JSONL file:
* Books.jsonl.gz
* Electronics.jsonl.gz
* Beauty_and_Personal_Care.jsonl.gz
* Home_and_Kitchen.jsonl.gz

### Metadata
Additional metadata files are available for each category:
* meta_Books.jsonl.gz
* meta_Electronics.jsonl.gz
* meta_Beauty_and_Personal_Care.jsonl.gz
* meta_Home_and_Kitchen.jsonl.gz

These files contain review text, ratings, and product information that are processed, embedded, and stored in the Qdrant vector database for efficient semantic search.

## Features

- **Elegant Interface**: Sophisticated, modern UI with thoughtful design elements
- **Semantic Search**: Search for reviews using natural language queries instead of exact keyword matching
- **Sentiment Analysis**: Reviews are classified by sentiment to understand customer opinions
- **Interactive Visualization**: Visualize sentiment distribution with professional interactive charts
- **Category Analytics**: Explore sentiment patterns across product categories in the analytics dashboard
- **Responsive Design**: Optimized for both desktop and mobile experiences
- **Download Capability**: Export search results for further analysis

## Setup

### Prerequisites

- Python 3.8+
- A Qdrant Cloud account (or self-hosted Qdrant instance)
- Streamlit account (for secrets management and deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nessaleal4/AmazonReviews_NLP.git
   cd AmazonReviews_NLP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Streamlit secrets:
   Create a `.streamlit/secrets.toml` file with:
   ```toml
   QDRANT_URL = "your-qdrant-cloud-url"
   QDRANT_API_KEY = "your-qdrant-api-key"
   ```

### Running the Application

```bash
streamlit run app.py
```

## Usage

### Main Search Interface

1. Enter a natural language query like "Harry Potter" in the search box
2. Click the "Search" button
3. Review the results table showing matching reviews
4. Examine the sentiment distribution visualizations in the "Sentiment Analysis" tab
5. Download results for further analysis

### Sentiment Analytics Dashboard

1. Navigate to the "Sentiment Analysis by Category" page from the sidebar
2. Select a product category to analyze or use "All" to compare across categories
3. Adjust the sample size using the slider for performance optimization
4. Click "Run Analysis" to generate interactive visualizations
5. Explore sentiment distribution across different product categories

## Technical Details

### Data Processing

The project processes Amazon review data from the following categories:
- Books
- Electronics
- Beauty and Personal Care
- Home and Kitchen

Each review is:
1. Extracted from the JSONL files
2. Preprocessed and cleaned
3. Analyzed for sentiment
4. Embedded using the sentence transformer model
5. Stored in the vector database with its metadata

### Vector Database

The application uses Qdrant for storing and querying vector embeddings of reviews:
- Collection name: "amazon_reviews"
- Vector dimension: 768 (from all-mpnet-base-v2)
- Payload includes: text, sentiment, and category

### Embeddings

- Model: all-mpnet-base-v2 from Sentence Transformers
- This model provides high-quality embeddings for semantic search

### Design Principles

The interface follows modern design principles:
- Card-based layout for clear content separation
- Strategic use of color for sentiment indicators (green for positive, dark red for negative)
- Tabbed interface for organizing complex content
- Responsive design that adapts to different screen sizes
- Elegant typography and spacing for improved readability

### Development Container

A development container configuration is provided for consistent development environments. This is particularly useful for GitHub Codespaces or local development with VSCode's Remote Containers extension.

## Future Improvements

- Add filtering by product categories or star ratings in the main interface
- Implement time-based analysis of sentiment trends
- Add support for multi-language reviews
- Extend with aspect-based sentiment analysis
- Incorporate keyword extraction for deeper review insights
- Enable comparison of products based on sentiment analysis

## Acknowledgments

- This project uses Sentence Transformers by UKPLab
- Qdrant vector database for efficient vector search
- Streamlit for the interactive web interface
- Plotly for professional data visualization
- Data from the Amazon Review Data (2023) dataset [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)
