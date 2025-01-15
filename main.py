import pandas as pd 
from retrieval import ApartmentRetriever

# Load the data
df = pd.read_csv('data/simple_listings.csv')

# Initialize the retriever
retriever = ApartmentRetriever()
retriever.prepare_data(df)
retriever.compute_embeddings()

# User query
beds = input("Enter the number of bedrooms: ")
bathrooms_text = input("Enter the number of bathrooms (e.g., '1 bath' or '2.5 baths'): ")
price = input("Enter the price of the apartment: ")
number_of_reviews = input("Enter the number of reviews: ")
review_scores_rating = input("Enter the average rating: ")
query_parts = []

if beds:
    query_parts.append(f"{beds} bedrooms")

if bathrooms_text:
    query_parts.append(f"{bathrooms_text} bathrooms")

if price:
    try:
        query_parts.append(f"${float(price):.2f}")
    except ValueError:
        print("Prezzo non valido, ignorato.")

if number_of_reviews:
    query_parts.append(f"{number_of_reviews} reviews")

if review_scores_rating:
    query_parts.append(f"{review_scores_rating} rating")

# Concatenazione della query
query = ", ".join(query_parts)

# Search for apartments
results = retriever.search(query)

print("\n------------------ TOP 3 SOLUTIONS FOUND -----------------")
for result in results:
    print(f"ID: {result['id']}")
    print(f"Url: {result['url']}")
    print(f"Beds: {result['beds']}")
    print(f"Price: {result['price']}")
    print(f"Bathrooms: {result['bathrooms']}")
    print(f"Rating: {result['rating']}")
    print(f"Number of reviews: {result['reviews']}")
    print(f"Similarity Score: {result['similarity']:.3f}")
    print("--------------------------------------------------------")
