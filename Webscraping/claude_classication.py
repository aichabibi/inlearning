import os
import json
import anthropic

# Initialiser le client Anthropic avec la clé API depuis les variables d'environnement
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Définition des mots-clés avancés (sans surpondérer)
advanced_keywords = ["multithreading", "concurrency", "security", "memory management",
                     "optimization", "architecture", "performance tuning", "machine learning",
                     "deep learning", "big data", "scalability", "networking", "cloud computing"]

# Fonction pour appeler Claude et obtenir le niveau principal du cours
def get_course_level(course_text):
    print("🤖 Analyse du cours avec Claude...")
    prompt = f"""
    Voici le texte d'un cours de programmation :
    ---
    {course_text}
    ---
    Analyse la difficulté du cours et attribue un niveau **entre 1 et 5** :
    - **1 = Très basique** (Introduction, syntaxe simple)
    - **2 = Débutant** (Bases complètes du langage)
    - **3 = Intermédiaire** (POO, API, structures avancées)
    - **4 = Avancé** (Multithreading, optimisation, sécurité)
    - **5 = Expert** (Architecture, machine learning, systèmes distribués)

    **Prends en compte la notion principale** : si le cours est majoritairement simple, il ne doit pas être classé haut, même s'il contient quelques notions avancées.

    Réponds uniquement avec un chiffre entre **1 et 5**.
    """

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        level = int(message.content[0].text.strip())  # Convertir la réponse en entier
        print(f"✅ Niveau principal attribué par Claude : {level}")
        return level
    
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à Claude : {e}")
        return 1  # Valeur par défaut si erreur

# Fonction pour ajuster le niveau en fonction des notions avancées
def adjust_level_with_keywords(course_text, base_level):
    text_lower = course_text.lower()
    
    # Compter les occurrences de notions avancées
    advanced_count = sum(text_lower.count(keyword) for keyword in advanced_keywords)

    # Si plusieurs notions avancées sont détectées, on ajuste le niveau de +1 max
    if advanced_count > 2 and base_level < 5:
        adjusted_level = base_level + 1
        print(f"🔼 Notions avancées détectées ({advanced_count} occurrences), passage de {base_level} → {adjusted_level}")
        return adjusted_level
    return base_level

# Créer le dossier level s'il n'existe pas
if not os.path.exists('level'):
    os.makedirs('level')
    print("📁 Création du dossier 'level'")

# Traiter tous les fichiers JSON du dossier formatted
formatted_dir = 'formatted'
print(f"\n🔍 Analyse des fichiers dans le dossier '{formatted_dir}'...")

for filename in os.listdir(formatted_dir):
    if filename.endswith('.json'):
        input_file = os.path.join(formatted_dir, filename)
        output_file = os.path.join('level', filename)
        
        print(f"\n📄 Traitement du fichier : {filename}")

        # Charger le fichier JSON des cours
        with open(input_file, "r", encoding="utf-8") as f:
            courses_data = json.load(f)
            print(f"📚 Nombre de cours à analyser : {len(courses_data)}")

        # Appliquer l'analyse Claude à chaque cours
        for i, course_data in enumerate(courses_data, 1):
            print(f"\n🔄 Analyse du cours {i}/{len(courses_data)}")
            course = course_data["cours"]
            text_content = f"{course.get('titre', '')} {course.get('description', '')} " \
                        f"{' '.join(course.get('contenus', {}).get('paragraphs', []))}"
            
            # Étape 1 : Claude attribue le niveau principal
            base_level = get_course_level(text_content)
            
            # Étape 2 : Ajustement en fonction des notions avancées
            course["niveau"] = adjust_level_with_keywords(text_content, base_level)

        # Sauvegarder le fichier mis à jour
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(courses_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Sauvegarde du fichier : {output_file}")

        print(f"✨ Classification terminée pour {filename} !")
