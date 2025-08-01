import json

with open('merged_courses.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    if 'cours' in item and 'vecteur_embedding' in item['cours']:
        del item['cours']['vecteur_embedding']

with open('merged_courses_cleaned.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
    