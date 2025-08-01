import json
import sys
from pathlib import Path

# Ajouter le chemin racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, Tuple, List
import joblib
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoursePipeline:
    def __init__(self, categories_classifier, level_model_path: str, vectorizer_path: str):
        """
        Initialise la pipeline avec les modèles nécessaires.
        
        Args:
            categories_classifier: Instance de CourseClassifier pour la classification par catégorie
            level_model_path: Chemin vers le modèle RandomForest sauvegardé
            vectorizer_path: Chemin vers le vectorizer TF-IDF sauvegardé
        """
        self.categories_classifier = categories_classifier
        self.level_model = joblib.load(level_model_path)
        self.level_vectorizer = joblib.load(vectorizer_path)
        logger.info("Modèles chargés avec succès")
    
    def process_single_course(self, course_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un cours unique pour prédire sa catégorie et son niveau.
        
        Args:
            course_data: Dictionnaire contenant les données du cours
        
        Returns:
            Dict contenant les données du cours enrichies
        """
        # Extraire le texte du cours
        title = course_data.get("titre", "")
        description = course_data.get("description", "")
        contents = course_data.get("contenus", {})
        paragraphs = contents.get("paragraphs", []) if isinstance(contents, dict) else []
        
        # Concaténer le texte pour l'analyse
        full_text = f"{title} {description} " + " ".join(paragraphs)
        
        # Classifier la catégorie
        category, category_score = self.categories_classifier.classify_text(full_text)
        
        # Vectoriser le texte avec TF-IDF
        level_vector = self.level_vectorizer.transform([full_text])
        
        # Prédire le niveau
        predicted_level = self.level_model.predict(level_vector)[0]
        level_probabilities = self.level_model.predict_proba(level_vector)[0]
        
        # Enrichir les données du cours
        enriched_data = course_data.copy()
        enriched_data.update({
            "categorie": category,
            "score_categorie": round(float(category_score), 3),
            "niveau": int(predicted_level),
            "probabilites_niveau": {
                f"niveau_{j+1}": round(float(prob), 3)
                for j, prob in enumerate(level_probabilities)
            }
        })
        
        return enriched_data
        
    def process_courses_batch(self, courses_data: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Traite un lot de cours pour prédire leurs catégories et niveaux.
        
        Args:
            courses_data: Liste de dictionnaires contenant les données des cours
            batch_size: Taille du lot pour le traitement
            
        Returns:
            Liste de dictionnaires contenant les données des cours enrichies
        """
        # Extraire les textes des cours
        texts = []
        for course in courses_data:
            title = course.get("titre", "")
            description = course.get("description", "")
            contents = course.get("contenus", {})
            paragraphs = contents.get("paragraphs", []) if isinstance(contents, dict) else []
            
            # Concaténer le texte pour l'analyse
            full_text = f"{title} {description} " + " ".join(paragraphs)
            texts.append(full_text)
        
        # Vectoriser les textes avec TF-IDF
        logger.info("Vectorisation des textes avec TF-IDF...")
        level_vectors = self.level_vectorizer.transform(texts)
        
        # Prédire les niveaux pour tous les cours
        logger.info("Prédiction des niveaux...")
        predicted_levels = self.level_model.predict(level_vectors)
        level_probabilities = self.level_model.predict_proba(level_vectors)
        
        # Enrichir les données des cours
        logger.info("Enrichissement des données des cours...")
        enriched_courses = []
        for i, course in enumerate(tqdm(courses_data, desc="Traitement des cours")):
            # Classifier la catégorie
            category, category_score = self.categories_classifier.classify_text(texts[i])
            
            # Enrichir les données du cours
            enriched_data = course.copy()
            enriched_data.update({
                "categorie": category,
                "score_categorie": round(float(category_score), 3),
                "niveau": int(predicted_levels[i]),
                "probabilites_niveau": {
                    f"niveau_{j+1}": round(float(prob), 3)
                    for j, prob in enumerate(level_probabilities[i])
                }
            })
            enriched_courses.append(enriched_data)
        
        return enriched_courses

    def process_file(self, input_path: Path, output_path: Path, batch_size: int = 32) -> None:
        """
        Traite un fichier JSON contenant des cours.
        
        Args:
            input_path: Chemin vers le fichier JSON d'entrée
            output_path: Chemin où sauvegarder les résultats
            batch_size: Taille du lot pour le traitement
        """
        try:
            # Charger les données
            logger.info(f"Chargement des données depuis {input_path}...")
            with open(input_path, "r", encoding="utf-8") as f:
                courses = json.load(f)
            
            # Préparer les données des cours
            courses_data = []
            for course in courses:
                if isinstance(course, dict):
                    course_data = course.get("cours", {}) if "cours" in course else course
                    courses_data.append(course_data)
            
            # Traiter les cours par lots
            logger.info(f"Traitement de {len(courses_data)} cours...")
            processed_courses = self.process_courses_batch(courses_data, batch_size)
            
            # Sauvegarder les résultats
            logger.info(f"Sauvegarde des résultats dans {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_courses, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Traitement terminé. {len(processed_courses)} cours traités et sauvegardés dans {output_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
            raise

def main():
    # Chemins des modèles (à ajuster selon vos besoins)
    MODEL_PATH = "./course_pipeline/level_model.joblib"
    VECTORIZER_PATH = "./course_pipeline/vectorizer.joblib"
    
    # Initialiser le classifier de catégories
    from learning_platform.models.course_pipeline.course_classifier import CourseClassifier, CATEGORIES
    categories_classifier = CourseClassifier(CATEGORIES)
    
    # Créer la pipeline
    pipeline = CoursePipeline(
        categories_classifier=categories_classifier,
        level_model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH
    )
    
    # Exemple d'utilisation pour un fichier
    input_file = Path("augmented_merged_courses.json")
    output_file = Path("cours_enrichis.json")
    
    # Traiter les cours par lots de 32
    pipeline.process_file(input_file, output_file, batch_size=32)
    
    # Exemple d'utilisation pour un cours unique
    # Créer un cours de test
    test_course = {
        "titre": "Introduction à Python",
        "description": "Un cours pour débutants qui couvre les bases de Python",
        "contenus": {
            "paragraphs": [
                "Python est un langage de programmation facile à apprendre.",
                "Ce cours couvre les variables, les boucles et les fonctions."
            ],
            "lists": [
                ["Variables", "Types de données", "Opérateurs"],
                ["Boucles for et while", "Conditions if/else"]
            ],
            "examples": [
                "x = 10",
                "for i in range(5):",
                "    print(i)"
            ]
        }
    }
    
    # Traiter le cours unique
    processed_course = pipeline.process_single_course(test_course)
    print("\nRésultat pour le cours unique:")
    print(f"Titre: {processed_course['titre']}")
    print(f"Niveau prédit: {processed_course['niveau']}")
    print(f"Catégorie: {processed_course['categorie']}")
    print(f"Score de catégorie: {processed_course['score_categorie']}")
    print("Probabilités de niveau:")
    for niveau, prob in processed_course['probabilites_niveau'].items():
        print(f"  {niveau}: {prob}")

if __name__ == "__main__":
    main()