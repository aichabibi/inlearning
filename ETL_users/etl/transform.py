# Dans transform.py
def transform_data(df):
    """
    Transformation des données pour qu'elles correspondent aux colonnes de PostgreSQL.
    """
    df = df.rename(columns={
        "Nom": "nom",
        "Âge": "age",
        "Sexe": "sexe",
        "Email": "email",
        "Téléphone": "telephone",
        "Langue préférée": "langue_preferee",
        "Mode d'apprentissage": "mode_apprentissage",
        "Centres d'intérêt": "centres_interet",
        "Niveau académique": "niveau_academique",
        "Expérience (années)": "experience_annees"
    })

    # Sélectionner les colonnes dans le bon ordre
    df = df[[
        "nom", "age", "sexe", "email", "telephone",
        "langue_preferee", "mode_apprentissage", "centres_interet",
        "niveau_academique", "experience_annees"
    ]]

    return df
