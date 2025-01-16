from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import json
import torch

# Funzione per generare una descrizione completa di un appartamento
def generate_full_listing(beds, baths, price, reviews, rating):
    return f"{beds} bedrooms, {baths} bathrooms, ${price}, {reviews} reviews, {rating} rating"

# Funzione per creare una query incompleta (Anchor)
def generate_partial_query(beds, baths, price, reviews, rating):
    components = []
    if random.choice([True, False]):
        components.append(f"{beds} bedrooms")
    if random.choice([True, False]):
        components.append(f"{baths} bathrooms")
    if random.choice([True, False]):
        components.append(f"${price}")
    if random.choice([True, False]):
        components.append(f"{reviews} reviews")
    if random.choice([True, False]):
        components.append(f"{rating} rating")
    
    # Se nessuna info viene scelta, forziamo almeno il prezzo
    if not components:
        components.append(f"${price}")
    
    return ", ".join(components)

# Generazione di 100 esempi
train_examples = []

for _ in range(10000):
    # Generazione casuale di tutti i dettagli
    beds = random.choice([1, 2, 3])
    baths = random.choice([1, 2])
    price = round(random.uniform(40, 200), 2)
    reviews = random.randint(10, 300)
    rating = round(random.uniform(3.0, 5.0), 2)
    
    # Anchor: query parziale dell'utente
    anchor = generate_partial_query(beds, baths, price, reviews, rating)
    
    # Positive: descrizione completa che include tutte le info dell'Anchor
    positive = generate_full_listing(beds, baths, price, reviews, rating)
    
    # Negative: descrizione diversa in almeno un campo specificato nell'Anchor
    while True:
        neg_beds = random.choice([1, 2, 3, 4, 5])
        neg_baths = random.choice([1, 2, 3])
        neg_price = round(random.uniform(40, 400), 2)
        neg_reviews = random.randint(0, 500)
        neg_rating = round(random.uniform(2.0, 5.0), 2)
        
        negative = generate_full_listing(neg_beds, neg_baths, neg_price, neg_reviews, neg_rating)
        
        # Condizione: il Negative deve differire in almeno uno degli aspetti specificati nell'Anchor
        conflict = False
        if "bedrooms" in anchor and f"{neg_beds} bedrooms" == anchor:
            conflict = True
        if "bathrooms" in anchor and f"{neg_baths} bathrooms" == anchor:
            conflict = True
        if "$" in anchor and f"${neg_price}" == anchor:
            conflict = True
        if "reviews" in anchor and f"{neg_reviews} reviews" == anchor:
            conflict = True
        if "rating" in anchor and f"{neg_rating} rating" == anchor:
            conflict = True
        
        if not conflict:
            break  # Il Negative è valido
    
    # Creazione dell'esempio triplo
    train_examples.append(InputExample(texts=[anchor, positive, negative]))

def save_examples_to_file(examples, filename="train_data.jsonl"):
    with open(filename, "w") as f:
        for example in examples:
            json.dump({
                "anchor": example.texts[0],
                "positive": example.texts[1],
                "negative": example.texts[2]
            }, f)
            f.write("\n")

save_examples_to_file(train_examples, filename="train_data.jsonl")

# Caricamento del modello pre-addestrato
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Ricarica dei dati dal file salvato
train_examples = []
with open("train_data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        train_examples.append(InputExample(texts=[data["anchor"], data["positive"], data["negative"]]))

# Creazione del DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Definizione della Triplet Loss
train_loss = losses.TripletLoss(model=model)

# Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    show_progress_bar=True
)

# Salvataggio del modello fine-tuned
model.save("fine_tuned_model")
print("Training completato e modello salvato in 'fine_tuned_model'.")
