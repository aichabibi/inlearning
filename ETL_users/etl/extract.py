import pandas as pd

def extract_data(filepath: str) -> pd.DataFrame:
    """
    Fonction pour extraire les données depuis un fichier Excel.
    :param filepath: Chemin du fichier à lire.
    :return: Un DataFrame Pandas contenant les données ou None en cas d'erreur.
    """
    df = None  # Initialisation par défaut
    try:
        # Lire le fichier Excel
        df = pd.read_excel(filepath, engine='openpyxl')
        print(f"Données extraites avec succès. Nombre de lignes : {len(df)}")
    except FileNotFoundError as e:
        print(f"Erreur : Fichier non trouvé - {e}")
    except pd.errors.ParserError as e:
        print(f"Erreur de parsing des données : {e}")
    except Exception as e:
        print(f"Erreur inattendue lors de l'extraction des données : {e}")
    
    return df
