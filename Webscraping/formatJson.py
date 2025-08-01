import json
import os
from typing import Dict, List

def format_course_json(input_file: str, output_file: str) -> None:
    """
    Lit un fichier JSON existant et ajoute les champs manquants selon le schéma défini.
    Les champs id et vecteur_embedding seront laissés vides pour être remplis plus tard.
    
    Args:
        input_file (str): Chemin vers le fichier JSON d'entrée
        output_file (str): Chemin vers le fichier JSON de sortie
    """
    # Lecture du JSON existant
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ajout des champs manquants pour chaque cours
    formatted_courses = []
    for course in data:
        # Garder les données existantes
        formatted_course = course.copy()
        
        # S'assurer que tous les champs requis existent
        if "cours" not in formatted_course:
            formatted_course["cours"] = {}
            
        cours = formatted_course["cours"]
        
        # Ajouter les champs manquants avec des valeurs par défaut
        if "id" not in cours:
            cours["id"] = ""
        if "contenus" not in cours:
            cours["contenus"] = {
                "texte": cours.get("texte", ""),
                "lienVideo": cours.get("lienVideo", "")
            }
        if "categories" not in cours:
            cours["categories"] = []
        if "niveau" not in cours:
            cours["niveau"] = "Débutant"
        if "duree" not in cours:
            cours["duree"] = "45 minutes"
        if "vecteur_embedding" not in cours:
            cours["vecteur_embedding"] = []
            
        formatted_courses.append(formatted_course)
    
    # Sauvegarde dans un nouveau fichier JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_courses, f, ensure_ascii=False, indent=2)

# Exemple d'utilisation
if __name__ == "__main__":
    input_folder = "output"
    formatted_folder = "formatted"
    
    if not os.path.exists(formatted_folder):
        os.makedirs(formatted_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith("course_content.json"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.join(formatted_folder, filename.replace("course_content.json", "formatted_content.json"))
            format_course_json(input_path, output_filename)