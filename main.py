from retrieval import SimpleRealEstateIR
import pandas as pd 

def run_ir_system(data_path, query):
    df = pd.read_csv(data_path)
    
    ir_system = SimpleRealEstateIR()
    
    # Preprocessing minimo
    # df = ir_system.preprocess_data(df)
    
    embeddings = ir_system.create_embeddings(df)

    results = ir_system.search(query, embeddings, df)
    
    return ir_system, embeddings, df, results


query = input("What kind of apartment are you looking for?\n")

# 1 bathroom, 50$ max per night, at least 100 reviews, at least 4.5 rating

ir_system, embeddings, df, results = run_ir_system(
    r'C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab2\retrieval\data\simple_listings.csv', 
    query
)

print("\n------------------ TOP 3 SOLUTIONS FOUND -----------------")
for result in results:
    print(f"ID: {result['id']}")
    print(f"Url: {result['url']}")
    print(f"Beds: {result['beds']}")
    print(f"Price: {result['price']}")
    print(f"Bathrooms: {result['bathrooms']}")
    print(f"Rating: {result['rating']}")
    print(f"Number of reviews: {result['number_of_reviews']}")
    print(f"Similarity Score: {result['similarity_score']:.3f}")
    print("--------------------------------------------------------")
    
