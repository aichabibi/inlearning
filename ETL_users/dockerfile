# Étape 1 : Utiliser une image Python comme base
FROM python:3.9

# Étape 2 : Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libpq-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Étape 3 : Définir le répertoire de travail
WORKDIR /app

# Étape 4 : Copier les fichiers nécessaires dans l'image
COPY . /app

# Étape 5 : Mettre à jour pip et installer les bibliothèques Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ajouter /app au PYTHONPATH
ENV PYTHONPATH="/app"

# Étape 7 : Commande pour exécuter le script principal
CMD ["python", "pipeline.py"]