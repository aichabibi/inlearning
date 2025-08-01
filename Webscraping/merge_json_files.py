import json
import os
import uuid

def merge_json_files():
    # Chemin vers le dossier contenant les fichiers JSON
    input_dir = "level"
    # Chemin pour le fichier de sortie
    output_file = "merged_courses.json"
    
    # Liste pour stocker tout le contenu
    all_content = []
    
    # Parcourir tous les fichiers JSON dans le dossier
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                    # Si le contenu est une liste
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'cours' in item:
                                # Générer un ID unique pour chaque cours
                                item['cours']['id'] = str(uuid.uuid4())
                                all_content.append(item)
                    # Si le contenu est un dictionnaire
                    elif isinstance(content, dict):
                        if 'cours' in content:
                            content['cours']['id'] = str(uuid.uuid4())
                            all_content.append(content)
                            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Sauvegarder le contenu fusionné
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully merged {len(all_content)} items into {output_file}")
        
        # Afficher quelques statistiques
        print(f"\nStatistics:")
        print(f"Total number of courses: {len(all_content)}")
        print(f"Number of files processed: {len([f for f in os.listdir(input_dir) if f.endswith('.json')])}")
        
    except Exception as e:
        print(f"Error saving merged file: {e}")

if __name__ == "__main__":
    merge_json_files()