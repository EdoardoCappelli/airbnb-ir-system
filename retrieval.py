import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SimpleRealEstateIR:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
    def preprocess_data(self, df):
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        return df
        
    def create_embeddings(self, df):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text_descriptions = []
        for _, row in df.iterrows():
            description = f"Property with {row['beds']} beds and {row['bathrooms_text']}. "
            description += f"Price: ${row['price']}. "
            description += f"Rating: {row['review_scores_rating']} from {row['number_of_reviews']} reviews."
            description += f"{row['description']}"
            
            text_descriptions.append(description)
            
        print("=> Create the embeddings")
        embeddings = self.text_model.encode(text_descriptions, device=device, show_progress_bar=True)

        return embeddings
    
    def search(self, query, embeddings, df, top_k=3):
        print("=> Finding the best solutions for your query...")

        query_embedding = self.text_model.encode([query])
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            result = {
                'id': df.iloc[idx]['id'],
                'url': df.iloc[idx]['listing_url'],
                'beds': df.iloc[idx]['beds'],
                'price': df.iloc[idx]['price'],
                'bathrooms': df.iloc[idx]['bathrooms_text'],
                'rating': df.iloc[idx]['review_scores_rating'],
                'number_of_reviews': df.iloc[idx]['number_of_reviews'],
                'similarity_score': similarities[idx],
                # 'coordinates': (df.iloc[idx]['latitude'], df.iloc[idx]['longitude'])
            }
            results.append(result)
            
        return results
