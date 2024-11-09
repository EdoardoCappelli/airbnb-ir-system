# Real Estate Information Retrieval System

## Project Overview
This project aims to create a simple Information Retrieval (IR) system to help users find the most relevant real estate listings based on their query. Using sentence embeddings, the system compares user queries with property descriptions and other details, ranking the best matches.

The core functionalities include:
- Preprocessing real estate data.
- Generating embeddings using the **SentenceTransformer** model.
- Retrieving top-matching listings based on a user's query using cosine similarity.

## Dataset
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/alessiocrisafulli/airbnb-italy) and contains Airbnb listings in Italy. The CSV file should include the following columns:
- **id**: Unique identifier for the property.
- **listing_url**: URL to the property listing.
- **description**: Text description of the property.
- **beds**: Number of beds.
- **price**: Price of the listing in the format `$X,XXX`.
- **bathrooms_text**: Description of bathrooms (e.g., "1 bath").
- **number_of_reviews**: Number of reviews the property has received.
- **review_scores_rating**: Average rating score of the property.
- **latitude**: Latitude coordinate of the property.
- **longitude**: Longitude coordinate of the property.

You can download the dataset directly from the link above and save it in the `data/` directory.

## Prerequisites
Ensure you have the following packages installed:
```bash
pip install pandas numpy torch sentence-transformers scikit-learn
```

Make sure you have **CUDA** installed if you plan to run the model on a GPU for faster performance.

## Project Structure
```
project/
├── data/
│   └── simple_listings.csv      # The dataset
├── retrieval.py                 # Script for IR functionalities
├── main.py                      # Main entry point for running the system
└── README.md                    # Project documentation
```

## Explanation of Files

### 1. `retrieval.py`
This script contains the `SimpleRealEstateIR` class, which handles:
- **Data Preprocessing**: Cleans the dataset, particularly the `price` column.
- **Creating Embeddings**: Uses the **all-MiniLM-L6-v2** model to generate embeddings for property descriptions.
- **Search Functionality**: Computes cosine similarity between the query embedding and the embeddings of all listings to return the top matches.

### 2. `main.py`
This script runs the Information Retrieval system:
- Loads the dataset.
- Initializes the IR system.
- Preprocesses data (currently commented out but available for customization).
- Generates embeddings for all listings.
- Prompts the user for a query.
- Retrieves and displays the top 3 most relevant listings.

## How to Run
Make sure your dataset is in the `data/` folder and named `simple_listings.csv`. To run the system:

```bash
python main.py
```

You will be prompted to enter a query. For example:
```
What kind of apartment are you looking for?
> 1 bathroom, $50 max per night, at least 100 reviews, at least 4.5 rating
```

### Sample Output
```
------------------ TOP SOLUTIONS FOUND -----------------
ID: 12345
Url: https://example.com/listing/12345
Beds: 2
Price: 45.0
Bathrooms: 1 bath
Rating: 4.8
Number of reviews: 150
Similarity Score: 0.923
--------------------------------------------------------
ID: 67890
Url: https://example.com/listing/67890
Beds: 1
Price: 50.0
Bathrooms: 1 bath
Rating: 4.7
Number of reviews: 200
Similarity Score: 0.910
--------------------------------------------------------
...
```

## Future Improvements
This project was primarily developed with an educational intent, to understand how Information Retrieval systems based on embeddings and text similarity work. **The goal was not to create a fully usable product**, but rather to explore techniques in natural language processing and search capabilities using pre-trained models.

That said, there are several improvements that could be made to make it more robust and useful in real-world scenarios:

- **Geographic Filtering**: Add location-based filtering using latitude and longitude to refine results based on proximity.
- **Advanced Query Parsing**: Improve query understanding to handle more complex and intuitive requests.
- **Web Interface**: Develop a web interface using Flask or Django to make it easier for users to interact with the system.
- **Scalability**: Optimize the system to handle larger datasets and improve performance.
  
