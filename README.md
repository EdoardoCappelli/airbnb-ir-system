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
 
You can download the dataset directly from the link above and save it in the `data/` directory.

 
  
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
Enter the number of bedrooms: 2
Enter the number of bathrooms:
Enter the price of the apartment: 50
Enter the number of reviews:
Enter the average rating:
```

### Sample Output
```
------------------ TOP 3 SOLUTIONS FOUND -----------------
ID: 24011469
Url: https://www.airbnb.com/rooms/24011469
Beds: 2.0
Beds: 2.0
Price: 50.0
Price: 50.0
Bathrooms: 1.5 baths
Rating: 4.84
Bathrooms: 1.5 baths
Rating: 4.84
Number of reviews: 51
Similarity Score: 0.859
--------------------------------------------------------
ID: 20292780
Url: https://www.airbnb.com/rooms/20292780
Beds: 2.0
Price: 50.0
Bathrooms: 1.5 baths
Rating: 4.75
Number of reviews: 36
Similarity Score: 0.852
--------------------------------------------------------
ID: 52270967
Url: https://www.airbnb.com/rooms/52270967
Beds: 2.0
Price: 50.0
Bathrooms: 1.5 baths
Rating: 4.9
Number of reviews: 21
Similarity Score: 0.848
--------------------------------------------------------
```
## Training
### Data Generation:
Randomly generate examples of apartment listings with various attributes (e.g., number of bedrooms, bathrooms, price, reviews, and rating).

For each generated listing, create:

* Anchor: A partial query that includes some of the listing attributes.
* Positive: A full listing with all attributes matching the anchor.
* Negative: A listing with at least one attribute different from the anchor.

The generated data is stored as triplet examples: (Anchor, Positive, Negative).

### Triplet Loss Function
I used the Triplet Loss objective, which encourages the model to embed the anchor closer to the positive example than to the negative example. The model is trained for 10 epochs, with a warmup step to gradually adjust the learning rate

## Future Improvements
This project was primarily developed with an educational intent, to understand how Information Retrieval systems based on embeddings and text similarity work. **The goal was not to create a fully usable product**, but rather to explore techniques in natural language processing and search capabilities using pre-trained models.

That said, there are several improvements that could be made to make it more robust and useful in real-world scenarios:

- **Geographic Filtering**: Add location-based filtering using latitude and longitude to refine results based on proximity.
- **Advanced Query Parsing**: Improve query understanding to handle more complex and intuitive requests.
- **Web Interface**: Develop a web interface using Flask or Django to make it easier for users to interact with the system.
- **Scalability**: Optimize the system to handle larger datasets and improve performance.
  
