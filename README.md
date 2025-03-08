# AmazonReviews_NLP

A natural language processing application that analyzes Amazon product reviews, providing semantic search capabilities and sentiment analysis visualization.

## Overview

This project leverages semantic search and sentiment analysis to help users explore Amazon product reviews. It allows users to search for reviews using natural language queries and visualizes the sentiment distribution of the search results.

The application:
- Embeds reviews using sentence transformers (all-mpnet-base-v2)
- Stores and retrieves vectors using Qdrant vector database
- Provides a user-friendly interface built with Streamlit
- Visualizes sentiment distribution with Plotly

## Project Structure

```
AmazonReviews_NLP/
├── .devcontainer/
│   └── devcontainer.json     # Development container configuration
├── pages/
│   └── 02_SentimentAnalytics.py  # Analytics dashboard for sentiment analysis
├── app.py                    # Main application file for the search interface
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

- **Semantic Search**: Search for reviews using natural language queries instead of exact keyword matching
- **Sentiment Analysis**: Reviews are classified by sentiment to understand customer opinions
- **Interactive Visualization**: Visualize sentiment distribution with interactive charts
- **Category Filters**: Filter reviews by product categories (available in the analytics dashboard)

## Setup

### Prerequisites

- Python 3.8+
- A Qdrant Cloud account (or self-hosted Qdrant instance)
- Streamlit account (for secrets management)

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

1. Enter a natural language query like "Olay body wash" in the search box
2. Click the "Search" button
3. Review the results table showing matching reviews
4. Examine the sentiment distribution chart

### Sentiment Analytics Dashboard

1. Navigate to the Sentiment Analytics page from the sidebar
2. Explore more detailed sentiment analysis by categories
3. Filter and analyze trends in customer sentiment

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

### Development Container

A development container configuration is provided for consistent development environments. This is particularly useful for GitHub Codespaces or local development with VSCode's Remote Containers extension.

## Future Improvements

- Add filtering by product categories or star ratings
- Implement time-based analysis of sentiment trends
- Add support for multi-language reviews
- Extend with aspect-based sentiment analysis

## License

[Add your license information here]

## Acknowledgments

- This project uses Sentence Transformers by UKPLab
- Qdrant vector database for efficient vector search
- Streamlit for the interactive web interface
- Data from the Amazon Review Data (2023) dataset [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)
