import json
from pathlib import Path

# Chemin vers ton fichier
input_path = Path("learning_platform/data/course_pipeline/courses.json")
output_path = Path("learning_platform/data/course_pipeline/courses_fixed.json")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_data = []
for item in data:
    cours = item.get("cours", {})
    fixed_data.append({
        "title": cours.get("titre", ""),
        "description": cours.get("description", ""),
        "url": item.get("url", cours.get("lien", "")),
        "level": cours.get("niveau", 1),
        "duration": cours.get("duree", ""),
        "category": cours.get("categories", ""),
        "contents": cours.get("contenus", {})
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, indent=2, ensure_ascii=False)

print(f"✅ Nouveau fichier généré : {output_path}")
