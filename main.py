import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class ApartmentRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.apartments_df = None
        self.embeddings = None
        self.embeddings_file = "embeddings/apartments_embeddings.npy"   
        
    def prepare_data(self, df):
        """Process and clean the apartment data"""
        self.apartments_df = df.copy()
        
        self.apartments_df['price'] = self.apartments_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
   
        self.apartments_df['structured_desc'] = self.apartments_df.apply(
            lambda x: 
                    f"{x['beds']} bedrooms, "
                    f"{x['bathrooms_text']} bathrooms, "
                    f"${x['price']:.2f}, "
                    f"{x['number_of_reviews']} reviews, "
                    f"{x['review_scores_rating']} rating, ", axis=1
        )
        
    def encode_text(self, text):
        """Encode text using the transformer model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # Using [CLS] token
        
    def save_embeddings(self, embeddings, filename="embeddings/apartments_embeddings.npy"):
        """Save the embeddings to disk"""
        np.save(filename, embeddings)
        print(f"Embeddings saved to {filename}")
        
    def load_embeddings(self, filename="embeddings/apartments_embeddings.npy"):
        """Load the embeddings from disk if they exist"""
        if os.path.exists(filename):
            return np.load(filename)
        else:
            return None
        
    def compute_embeddings(self):
        """Compute embeddings for all apartments"""
        print("Checking if embeddings already exist...")
        self.embeddings = self.load_embeddings(self.embeddings_file)
        
        if self.embeddings is None:
            print("Computing the embeddings...")
            self.embeddings = np.vstack([
                self.encode_text(desc)[0] 
                for desc in self.apartments_df['structured_desc']
            ])
            self.save_embeddings(self.embeddings, self.embeddings_file)   
        else:
            print("Embeddings loaded from disk.")
        
    def search(self, query, top_k=3):
        """Search for apartments based on a natural language query"""
        # Encode the query
        query_embedding = self.encode_text(query)
        
        # Compute similarity scores
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with similarity scores
        results = []
        for idx in top_indices:
            apartment = self.apartments_df.iloc[idx]
            results.append({
                'similarity': similarities[idx],
                'id': apartment['id'],
                'url': apartment['listing_url'],
                'description': apartment['description'],
                'beds': apartment['beds'],
                'price': apartment['price'],
                'bathrooms': apartment['bathrooms_text'],
                'reviews': apartment['number_of_reviews'],
                'rating': apartment['review_scores_rating']
            })
        
        return results
    
