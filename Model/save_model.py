import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Charge et prépare les données depuis le fichier JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    texts = []
    levels = []
    titles = []
    original_data = []  # Stocker les données originales
    
    for item in data:
        course = item['cours']
        
        # Combine all text content
        text_content = [
            course.get('titre', ''),
            course.get('description', '')
        ]
        
        # Add content from 'contenus'
        if 'contenus' in course:
            contenus = course['contenus']
            text_content.extend(contenus.get('paragraphs', []))
            
            # Flatten and add lists
            for list_items in contenus.get('lists', []):
                if isinstance(list_items, list):
                    text_content.extend(list_items)
            
            # Add examples
            text_content.extend(contenus.get('examples', []))
        
        # Join all text content
        combined_text = ' '.join(filter(None, text_content))
        
        # Get the level (default to 1 if not specified)
        level = course.get('niveau', 1)
        
        texts.append(combined_text)
        levels.append(level)
        titles.append(course.get('titre', ''))
        original_data.append(item)  # Sauvegarder les données originales
    
    return pd.DataFrame({
        'title': titles,
        'text': texts,
        'level': levels
    }), original_data

# Charger les données
input_file = 'augmented_merged_courses.json'
df, original_data = load_data(input_file)
print(f"Nombre total de cours : {len(df)}")
print("\nDistribution des niveaux :")
print(df['level'].value_counts().sort_index())

# Visualisation de la distribution des niveaux
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='level')
plt.title('Distribution des niveaux de cours')
plt.xlabel('Niveau')
plt.ylabel('Nombre de cours')
plt.savefig('niveau_distribution.png')
plt.close()

# Vectorisation du texte avec TF-IDF
print("\nVectorisation des textes avec TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['level'].values

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nDimensions des données d'entraînement:", X_train.shape)
print("Dimensions des données de test:", X_test.shape)

# Création et entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle et le vectorizer
joblib.dump(model, 'level_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("\nModèle et vectorizer sauvegardés avec succès!")

# Afficher les features les plus importantes
feature_importance = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 20 features les plus importantes :")
print(feature_importance.head(20))

# Visualisation des features importantes
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 features les plus importantes')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Prédire les niveaux pour tous les cours
all_predictions = model.predict(X)

# Mettre à jour le fichier JSON avec les prédictions
output_file = 'cours_avec_niveaux.json'
for i, item in enumerate(original_data):
    if 'cours' in item:
        item['cours']['niveau'] = int(all_predictions[i])

# Sauvegarder les données mises à jour
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(original_data, f, indent=2, ensure_ascii=False)

print(f"\nFichier JSON mis à jour avec les prédictions et sauvegardé dans {output_file}") 