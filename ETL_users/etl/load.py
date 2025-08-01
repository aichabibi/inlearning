import psycopg2
from psycopg2.extras import execute_values

def create_table_if_not_exists(config):
    """
    Crée la table 'users' si elle n'existe pas déjà.
    """
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS users (
        nom TEXT,
        age INT,
        sexe TEXT,
        email TEXT,
        telephone TEXT,
        langue_preferee TEXT,
        mode_apprentissage TEXT,
        centres_interet TEXT,
        niveau_academique TEXT,
        experience_annees INT
    );
    '''
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["username"],
            password=config["password"],
            database=config["database_name"]
        )
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()
        print("Table 'users' créée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création de la table 'users' : {e}")

def load_data_to_postgres(df, config):
    """
    Chargement des données dans PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["username"],
            password=config["password"],
            database=config["database_name"]
        )
        cur = conn.cursor()
        #execute_values(cur, "INSERT INTO users (nom, age, sexe, email, telephone, langue_preferee, mode_apprentissage, centres_interet, niveau_academique, experience_annees) VALUES %s", df.values)
        execute_values(
            cur,
            "INSERT INTO users (nom, age, sexe, email, telephone, langue_preferee, mode_apprentissage, centres_interet, niveau_academique, experience_annees) VALUES %s",
            df.values
        )

        conn.commit()
        cur.close()
        conn.close()
        print("Données chargées avec succès dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors du chargement dans PostgreSQL : {e}")
