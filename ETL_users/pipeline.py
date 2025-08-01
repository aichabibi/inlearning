from etl import extract, transform, load
#from etl import load
import yaml
import os

def load_config(filepath: str):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def run_pipeline():
    print("Démarrage du pipeline ETL...")
    config = {
        "host": os.environ["POSTGRES_HOST"],
        "port": int(os.environ["POSTGRES_PORT"]),
        "username": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database_name": os.environ["POSTGRES_DB"],
        "filepath": "data/data.xlsx",
    }
    print("Configuration chargée avec succès.")

    # Étape 1 : Extraction des données
    data = extract.extract_data(config["filepath"])
    if data is not None:
        print(f"Données extraites avec succès. Nombre de lignes : {len(data)}")

        # Étape 2 : Transformation des données
        transformed_data = transform.transform_data(data)
        if transformed_data is not None:
            print("Données transformées avec succès.")

            # Étape 3 : Création de la table si nécessaire
            load.create_table_if_not_exists(config)

            # Étape 4 : Chargement des données
            load.load_data_to_postgres(transformed_data, config)
            print("Pipeline ETL terminé avec succès.")

if __name__ == "__main__":
    run_pipeline()
