from pymongo import MongoClient
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Connexion à MongoDB
client = MongoClient("mongodb://mongo:27017/")
db = client["searchengine"]
collection = db["documents"]

# Charger le modèle Word2Vec entraîné
model = Word2Vec.load("models/word2vec.model")

def vectorize_text(text):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def search_engine(query):
    query_vector = vectorize_text(query)
    
    results = []
    for doc in collection.find():
        doc_vector = np.array(doc["vector"])
        similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
        
        if similarity > 0.3:  # Seuil de pertinence
            results.append({"text": doc["text"], "similarity": similarity})
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results