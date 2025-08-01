import os
import json
import anthropic
import uuid
from time import sleep

# Initialiser le client Anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_advanced_course(level):
    """
    Génère un nouveau cours de niveau avancé en utilisant Claude
    """
    # Exemples de cours avancés pour guider Claude
    examples = """
    Exemple de cours niveau 4:
    {
        "titre": "Advanced System Architecture and Microservices",
        "description": "Deep dive into distributed systems architecture, focusing on scalability patterns and microservices implementation",
        "contenus": {
            "paragraphs": [
                "Understanding distributed system architectures and their implications for modern applications",
                "Implementation of advanced microservices patterns including Circuit Breaker, Bulkhead, and Sidecar",
                "Performance optimization techniques for distributed systems",
                "Advanced monitoring and observability strategies"
            ]
        }
    }

    Exemple de cours niveau 5:
    {
        "titre": "Machine Learning Systems Design",
        "description": "Expert-level course on designing and implementing production-grade machine learning systems",
        "contenus": {
            "paragraphs": [
                "Architecture patterns for large-scale ML systems",
                "Advanced model deployment strategies and MLOps practices",
                "Optimization of distributed training pipelines",
                "System design for real-time inference at scale"
            ]
        }
    }
    """

    prompt = f"""
    En tant qu'expert en informatique, génère un nouveau cours de niveau {level} (où 5 est expert).
    Le cours doit être très technique et avancé, adapté à des développeurs expérimentés.

    Exemples de référence:
    {examples}

    Génère un nouveau cours différent des exemples mais de même niveau technique.
    Utilise un format JSON similaire aux exemples.
    Inclus des concepts avancés et des techniques de pointe.
    Le cours doit être détaillé et approfondi.

    Réponds uniquement avec le JSON du cours, sans autre texte.
    """

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extraire et parser le JSON
        course_json = json.loads(response.content[0].text)
        
        # Ajouter un ID unique et le niveau
        course_json["id"] = str(uuid.uuid4())
        course_json["niveau"] = level
        course_json["duree"] = "120 minutes"  # Durée par défaut pour cours avancés
        
        return course_json
    
    except Exception as e:
        print(f"Erreur lors de la génération du cours: {e}")
        return None

# Charger le JSON existant
with open('merged_courses.json', 'r', encoding='utf-8') as f:
    courses = json.load(f)

# Générer de nouveaux cours
new_courses = []
num_courses_to_generate = 100  # Nombre de cours à générer pour chaque niveau

for level in [4, 5]:
    print(f"\nGénération de {num_courses_to_generate} cours de niveau {level}...")
    
    for i in range(num_courses_to_generate):
        print(f"Génération du cours {i+1}/{num_courses_to_generate}")
        
        new_course = generate_advanced_course(level)
        if new_course:
            # Adapter au format existant
            formatted_course = {
                "url": "https://generated-course.com",
                "cours": new_course
            }
            new_courses.append(formatted_course)
        
        # Pause pour respecter les limites d'API
        sleep(2)

# Ajouter les nouveaux cours
courses.extend(new_courses)

# Sauvegarder le dataset enrichi
output_file = 'augmented_merged_courses.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(courses, f, ensure_ascii=False, indent=2)

# Afficher les statistiques
def print_level_stats(courses_list):
    levels = {}
    for course in courses_list:
        level = course["cours"]["niveau"]
        levels[level] = levels.get(level, 0) + 1
    
    print("\nDistribution des niveaux:")
    for level in sorted(levels.keys()):
        print(f"Niveau {level}: {levels[level]} cours")

print("\nStatistiques finales:")
print_level_stats(courses)