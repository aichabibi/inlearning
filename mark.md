1. README.md (racine)
# InLearning – Plateforme d’apprentissage personnalisée

InLearning est une plateforme innovante de personnalisation de l’apprentissage basée sur l’IA.
Elle permet :
- La génération automatique de parcours pédagogiques grâce à l’API **Anthropic**.
- Un moteur de recherche intelligent basé sur **ElasticSearch** et **SBERT**.
- Un suivi de la progression des étudiants.
- Une interface simple et interactive avec **Streamlit**.

## Documentation
- [Installation](docs/INSTALLATION.md)
- [Guide d’utilisation](docs/USAGE.md)
- [Architecture technique](docs/ARCHITECTURE.md)
- [Contribuer au projet](docs/CONTRIBUTING.md)

## Technologies principales
- Python 3.12
- Flask (API)
- PostgreSQL (base de données)
- ElasticSearch (moteur de recherche)
- Docker & Docker Compose
- Streamlit (interface utilisateur)
- Anthropic API (génération de cours)

---

2. docs/INSTALLATION.md
# Installation et Configuration

## Prérequis
- Python 3.12
- Docker & Docker Compose
- Git

## Cloner le dépôt
```bash
git clone https://github.com/tonrepo/inlearning.git
cd inlearning

Variables d’environnement

Créer un fichier .env à la racine du projet :

POSTGRES_USER=inlearning
POSTGRES_PASSWORD=secret
POSTGRES_DB=inlearning_db
FLASK_ENV=development
ANTHROPIC_API_KEY=sk-xxxxxxxx

▶Lancer le projet
docker-compose up -d

Accéder aux services

Flask API : http://localhost:5000

Streamlit UI : http://localhost:8501

PostgreSQL : localhost:5432

ElasticSearch : http://localhost:9200


---

## 3. docs/USAGE.md

```markdown
# Guide d’utilisation – InLearning

## 1. Inscription
- Remplissez le formulaire utilisateur (âge, niveau, objectifs).
- L’IA **Anthropic** prédit votre niveau et génère un plan de cours personnalisé.

## 2. Navigation
- **Accueil** : recommandations principales.
- **Cours** : catalogue disponible.
- **Mon parcours** : cours en cours.
- **Progression** : suivi des résultats.
- **Profil** : mise à jour des préférences.

## 3. Exemple d’utilisation
```bash
curl -X POST http://localhost:5000/generate_course \
-H "Content-Type: application/json" \
-d '{"age":24,"objectif":"Apprendre Python pour la Data Science","niveau":"débutant"}'


Résultat → un plan de cours + estimation du niveau + recommandations.


---

## 4. docs/CONTRIBUTING.md

```markdown
# Contribuer au projet InLearning

Merci de votre intérêt ! Voici comment contribuer :

## Étapes
1. Forker le projet
2. Créer une branche de fonctionnalité :
   ```bash
   git checkout -b feature/nouvelle-fonctionnalite


Faire vos modifications et tester :

pytest


Commit et push :

git commit -m "Ajout : nouvelle fonctionnalité"
git push origin feature/nouvelle-fonctionnalite


Ouvrir une Pull Request

Règles de code

Respecter la PEP8 pour Python.

Documenter les fonctions et classes.

Ajouter des tests unitaires.


---

## 5. docs/ARCHITECTURE.md

```markdown
# Architecture technique – InLearning

## Composants principaux
- **Pipelines batch** → ingestion et transformation des données (Python + Pandas).
- **Docker** → orchestration des services.
- **Flask API** → endpoints pour la recherche, l’ajout de cours et la génération IA.
- **PostgreSQL** → stockage des données utilisateurs et cours.
- **ElasticSearch** → indexation et recherche vectorielle.
- **Anthropic API** → génération de cours et prédiction de niveau.
- **Streamlit** → interface utilisateur interactive.

## Diagramme simplifié
```text
[Utilisateur] → [Streamlit UI] → [Flask API] → [PostgreSQL / ElasticSearch]
                                  ↓
                          [Anthropic API]
