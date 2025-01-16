import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class ApartmentRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.apartments_df = None
        self.embeddings = None
        self.embeddings_file = r"embeddings/apartments_embeddings.npy"   
        
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
                    f"{x['review_scores_rating']} rating", axis=1
        )
        desc1 = self.apartments_df['structured_desc'][0]
        print(desc1)
        
    def encode_text(self, text):
        """Encode text using the transformer model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # Move inputs to GPU if available
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Move output back to CPU for numpy conversion
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Using [CLS] token
        
    def save_embeddings(self, embeddings, filename=r"embeddings/apartments_embeddings.npy"):
        """Save the embeddings to disk"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, embeddings)
        print(f"Embeddings saved to {filename}")
        
    def load_embeddings(self, filename=("embeddings/apartments_embeddings.npy")):
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
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(self.apartments_df), batch_size):
                batch = self.apartments_df['structured_desc'][i:i+batch_size].tolist()
                batch_embeddings = np.vstack([
                    self.encode_text(desc)[0] 
                    for desc in batch
                ])
                all_embeddings.append(batch_embeddings)
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(self.apartments_df)} descriptions...")
            
            self.embeddings = np.vstack(all_embeddings)
            self.save_embeddings(self.embeddings, self.embeddings_file)   
        else:
            print("Embeddings loaded from disk.")
        
    def search(self, query, top_k=3):
        """Search for apartments based on a natural language query"""
        # Encode the query
        query_embedding = self.encode_text(query)
        
        # Compute similarity scores using GPU if available
        if torch.cuda.is_available():
            # Convert to PyTorch tensors and ensure correct shape
            query_tensor = torch.from_numpy(query_embedding).to(self.device)
            embeddings_tensor = torch.from_numpy(self.embeddings).to(self.device)
            
            # Reshape tensors to 2D: (1, features) and (n_samples, features)
            query_tensor = query_tensor.squeeze(0)  # Remove extra dimension if present
            if len(query_tensor.shape) == 1:
                query_tensor = query_tensor.unsqueeze(0)
            
            # Compute similarities on GPU using matrix multiplication and normalization
            # Normalize the vectors
            query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
            embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
            
            # Compute cosine similarity
            similarities = torch.mm(query_norm, embeddings_norm.t()).squeeze().cpu().numpy()
        else:
            # Fall back to sklearn's CPU implementation
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with similarity scores
        results = []
        for idx in top_indices:
            apartment = self.apartments_df.iloc[idx]
            results.append({
                'similarity': float(similarities[idx]),
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
