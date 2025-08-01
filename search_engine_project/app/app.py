import os
import json
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from gensim.models import Word2Vec

app = Flask(__name__)

# ✅ Configuration MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = MongoClient(MONGO_URI)
db = client["searchengine"]
collection = db["documents"]

# ✅ Attente que MongoDB soit prêt
def wait_for_mongo():
    print("⏳ Attente de MongoDB...")
    retries = 5
    while retries > 0:
        try:
            client.admin.command("ping")
            print("✅ MongoDB est prêt !")
            return
        except Exception as e:
            print(f"MongoDB non disponible, nouvel essai... ({retries})")
            time.sleep(5)
            retries -= 1
    print("❌ Échec de connexion à MongoDB")
    exit(1)

wait_for_mongo()

# ✅ Initialisation de la base de données et entraînement du modèle Word2Vec
def initialize_db():
    global model

    print("🗑️ Suppression des anciens documents...")
    collection.delete_many({})  # Supprime tous les documents
    print("✅ Base de données nettoyée.")

    print("📥 Insertion des documents...")

    # Chemin du fichier JSON
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "workplace-documents.json"))
    
    if not os.path.exists(json_path):
        print(f"❌ Le fichier JSON {json_path} n'existe pas.")
        exit(1)

    with open(json_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    # 🔥 RECONSTRUCTION DES DOCUMENTS SANS _id
    clean_documents = [{"content": doc["content"]} for doc in documents if "content" in doc]

    collection.insert_many(clean_documents)
    print("✅ Données insérées.")

    # 🧠 Entraînement du modèle Word2Vec...
    print("🧠 Génération du modèle Word2Vec...")
    sentences = [doc["content"].split() for doc in documents if "content" in doc]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # ✅ Sauvegarde du modèle Word2Vec
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "word2vec.model")
    model.save(model_path)
    print("✅ Modèle Word2Vec généré et sauvegardé.")

    # ✅ Insérer les documents vectorisés dans MongoDB
    for doc in documents:
        if "content" in doc:
            words = doc["content"].split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            if vectors:
                vector = np.mean(vectors, axis=0)
                doc["vector"] = vector.tolist()
                collection.insert_one(doc)
    
    print("✅ Base de données initialisée avec succès !")

initialize_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").lower()
    results = collection.find({"content": {"$regex": query, "$options": "i"}})

    response = [{"content": doc["content"]} for doc in results]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
