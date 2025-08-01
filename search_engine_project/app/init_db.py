import os
import json
import numpy as np
from pymongo import MongoClient
from gensim.models import Word2Vec

# Connexion à MongoDB
client = MongoClient("mongodb://mongo:27017/")
db = client["searchengine"]
collection = db["documents"]

# Chemin absolu pour accéder au fichier JSON
json_path = os.path.join(os.path.dirname(__file__), "..", "data", "workplace-documents.json")

# Charger le corpus de documents
with open(json_path, "r", encoding="utf-8") as file:
    documents = json.load(file)

# Entraîner Word2Vec
sentences = [doc["content"].split() for doc in documents if "content" in doc]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Chemin absolu correct pour sauvegarder le modèle Word2Vec
model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)  # Assurer que le dossier 'models/' existe

model_path = os.path.join(model_dir, "word2vec.model")
model.save(model_path)

# Insérer les documents vectorisés dans MongoDB
for doc in documents:
    if "content" in doc:  # Vérifier que le champ "content" existe
        words = doc["content"].split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:  # Vérifier que la liste n'est pas vide avant de calculer la moyenne
            vector = np.mean(vectors, axis=0)
            doc["vector"] = vector.tolist()
            collection.insert_one(doc)

print("Base de données initialisée avec succès !")
