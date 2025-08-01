import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
import json
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Système Avancé de Classification des Cours",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Télécharger les ressources NLTK nécessaires
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk()

# Chargement des données
@st.cache_data
def load_data(json_file):
    data = json.load(json_file)
    df = pd.DataFrame(data)
    
    # Construction du texte complet et prétraitement
    def build_full_text(row):
        title = row.get('title', '')
        description = row.get('description', '')
        content = row.get('content', {})
        paragraphs = content.get('paragraphs', []) if isinstance(content, dict) else []
        return f"{title} {description} {' '.join(paragraphs)}"
    
    if 'title' in df.columns:
        df["texte_complet"] = df.apply(build_full_text, axis=1)
    
    # Prétraitement des dates si elles existent
    if 'date_creation' in df.columns:
        df['date_creation'] = pd.to_datetime(df['date_creation'], errors='coerce')
    
    # Nettoyer les données
    df = df.dropna(subset=["title", "texte_complet"])
    df = df[df["texte_complet"].str.lower().str.strip() != "nan nan"]
    
    # Ajouter le nombre de mots pour analyse
    df["nombre_mots"] = df["texte_complet"].apply(lambda x: len(str(x).split()))
    
    return df

# Classification avancée des cours
@st.cache_data
def classer_cours_avance(df):
    # Catégories principales avec mots-clés associés
    categories = {
        "Développement Web": ["react", "angular", "vue", "html", "css", "jsx", "frontend", "web", "javascript", "dom", "responsive", "sass", "bootstrap", "webpack", "spa"],
        "Développement Backend": ["node", "express", "api", "backend", "rest", "serveur", "microservices", "graphql", "django", "flask", "fastapi", "spring", "php", "laravel"],
        "Programmation Générale": ["typescript", "javascript", "python", "java", "c++", "c#", "algorithmes", "structures de données", "fonctionnel", "rust", "go", "swift"],
        "Programmation Orientée Objet": ["oop", "class", "inheritance", "encapsulation", "polymorphisme", "objets", "classes", "méthodes", "interfaces", "abstraction"],
        "Bases de Données": ["sql", "database", "bdd", "schema", "postgresql", "nosql", "mongodb", "mysql", "oracle", "sqlite", "redis", "cassandra", "elasticsearch"],
        "Analyse de Données": ["pandas", "numpy", "dataset", "analyse", "visualisation", "matplotlib", "tableau", "power bi", "excel", "dataframe", "statistiques"],
        "Intelligence Artificielle": ["machine learning", "deep learning", "model", "neural network", "classification", "regression", "clustering", "scikit-learn", "tensorflow", "pytorch", "keras"],
        "Big Data & Cloud": ["spark", "aws", "azure", "gcp", "hadoop", "databricks", "emr", "s3", "ec2", "lambda", "big data", "cloud computing", "dataproc"],
        "Systèmes & Réseaux": ["tcp", "linux", "réseau", "protocole", "unix", "shell", "bash", "administration système", "serveur", "routing", "firewall", "sécurité"],
        "DevOps": ["docker", "kubernetes", "jenkins", "ci/cd", "pipeline", "ansible", "terraform", "github actions", "gitlab ci", "déploiement continu", "intégration continue"],
        "Gestion de Projet": ["agile", "scrum", "kanban", "jira", "product owner", "sprint", "backlog", "burndown", "itératif", "planification", "management"],
        "IA Générative & NLP": ["nlp", "bert", "gpt", "traitement du langage", "large language model", "transformers", "text mining", "hugging face", "spacy", "sentiment analysis", "génération de texte"]
    }
    
    def classification_avancee(text):
        text = text.lower()
        scores = {}
        
        # Calculer un score pour chaque catégorie basé sur la fréquence des mots-clés
        for categorie, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[categorie] = score
        
        # Retourner la catégorie avec le score le plus élevé, ou "Autres" si tous les scores sont à 0
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "Autres"
    
    df["category"] = df["texte_complet"].apply(classification_avancee)
    
    # Analyse des mots-clés pour chaque catégorie (pour la sous-catégorisation)
    def extraire_mots_cles(text, min_length=3):
        text = text.lower()
        stop_words = set(stopwords.words('french') + stopwords.words('english'))
        tokens = word_tokenize(text)
        return [word for word in tokens if word.isalpha() and len(word) >= min_length and word not in stop_words]
    
    df["mots_cles"] = df["texte_complet"].apply(extraire_mots_cles)
    
    # Technologies/frameworks spécifiques pour la sous-catégorisation
    technologies = {
        "Python": ["python", "django", "flask", "fastapi"],
        "JavaScript": ["javascript", "js", "ecmascript", "node"],
        "React": ["react", "jsx", "hooks", "redux"],
        "Angular": ["angular", "typescript", "component", "directive"],
        "Vue": ["vue", "vuex", "nuxt"],
        "SQL": ["sql", "postgresql", "mysql", "oracle"], 
        "NoSQL": ["mongodb", "nosql", "dynamodb", "couchdb"],
        "Docker": ["docker", "conteneur", "dockerfile"],
        "Kubernetes": ["kubernetes", "k8s", "orchestration"],
        "TensorFlow": ["tensorflow", "keras", "deep learning"],
        "PyTorch": ["pytorch", "torch", "neural"],
        "Pandas": ["pandas", "dataframe", "numpy"],
        "AWS": ["aws", "amazon", "s3", "ec2", "lambda"],
        "Azure": ["azure", "microsoft", "cloud"],
        "GCP": ["gcp", "google cloud", "bigquery"],
    }
    
    def sous_categorisation(mots_cles):
        mots_cles_text = " ".join(mots_cles)
        scores = {}
        
        for tech, kw_list in technologies.items():
            score = sum(1 for kw in kw_list if kw in mots_cles_text)
            scores[tech] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "Autre"
    
    df["sous_categorie"] = df["mots_cles"].apply(sous_categorisation)
    
    # Estimation du niveau de difficulté basée sur une analyse lexicale
    def estimer_niveau_difficulte(texte):
        if pd.isna(texte):
            return "Intermédiaire"  # Valeur par défaut
            
        texte = texte.lower()
        mots_debutant = ["introduction", "basique", "débutant", "fondamentaux", "bases", "initiation"]
        mots_avance = ["avancé", "expert", "professionnel", "optimisation", "architecture", "complexe", "approfondi"]
        
        score_debutant = sum(1 for mot in mots_debutant if mot in texte)
        score_avance = sum(1 for mot in mots_avance if mot in texte)
        
        if score_avance > score_debutant:
            return "Avancé"
        elif score_debutant > score_avance:
            return "Débutant"
        else:
            return "Intermédiaire"
    
    # Appliquer l'estimation du niveau si la colonne 'level' n'existe pas
    if 'level' not in df.columns or df['level'].isna().all():
        df["level"] = df["texte_complet"].apply(estimer_niveau_difficulte)
    
    return df

# Clustering des cours avec KMeans
@st.cache_data
def clustering_cours(df, n_clusters=5):
    # Vectorisation avec TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=list(stopwords.words('french')) + list(stopwords.words('english')),
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["texte_complet"])
    
    # Réduction de dimensionnalité pour la visualisation
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    
    # Extraire les mots les plus importants par cluster pour les nommer
    cluster_names = {}
    
    for i in range(n_clusters):
        # Obtenir les indices des documents dans ce cluster
        cluster_indices = [idx for idx, label in enumerate(clusters) if label == i]
        if cluster_indices:
            # Agréger tous les mots-clés de ce cluster
            all_keywords = []
            for idx in cluster_indices:
                all_keywords.extend(df.iloc[idx].get("mots_cles", []))
            
            # Compter et obtenir les plus fréquents
            if all_keywords:
                keyword_counter = Counter(all_keywords)
                top_keywords = [kw for kw, _ in keyword_counter.most_common(3)]
                cluster_names[i] = " & ".join(top_keywords) if top_keywords else f"Cluster {i+1}"
            else:
                cluster_names[i] = f"Cluster {i+1}"
        else:
            cluster_names[i] = f"Cluster {i+1}"
    
    # Appliquer t-SNE pour la visualisation 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1) if len(df) > 1 else 1)
    X_tsne = tsne.fit_transform(X_reduced)
    
    # Ajouter les résultats au DataFrame
    df["cluster"] = clusters
    df["cluster_name"] = df["cluster"].map(cluster_names)
    df["x_tsne"] = X_tsne[:, 0]
    df["y_tsne"] = X_tsne[:, 1]
    
    return df, vectorizer, cluster_names

# Encoder les textes pour la recherche sémantique
@st.cache_resource
def encoder_textes(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["texte_complet"].tolist(), show_progress_bar=True)
    return model, embeddings

# Visualisation des clusters avec Plotly
def visualiser_clusters(df):
    fig = px.scatter(
        df, 
        x="x_tsne", 
        y="y_tsne", 
        color="cluster_name",
        hover_data=["title", "category", "sous_categorie", "level"],
        labels={"cluster_name": "Cluster"},
        title="Clustering des cours basé sur leur contenu (t-SNE)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

# Visualisation par catégorie et sous-catégorie
def visualiser_categories(df):
    fig = px.scatter(
        df, 
        x="x_tsne", 
        y="y_tsne", 
        color="category",
        symbol="sous_categorie",
        hover_data=["title", "level"],
        title="Classification des cours par catégorie et sous-catégorie",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

# Générer un nuage de mots
def generer_nuage_mots(df, colonne="texte_complet"):
    all_text = " ".join(df[colonne].dropna().astype(str).tolist())
    stop_words = set(stopwords.words('french') + stopwords.words('english'))
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=stop_words,
        max_words=100
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Recherche sémantique avancée
def recherche_semantique(df, embeddings, model, requete, n_results=10, min_score=0.3):
    # Encoder la requête
    query_embed = model.encode(requete)
    
    # Calculer les similarités
    scores = util.cos_sim(query_embed, embeddings)[0]
    
    # Ajouter les scores au DataFrame
    df_temp = df.copy()
    df_temp["similarite"] = scores.numpy()
    
    # Filtrer les résultats avec un score minimum
    resultats = df_temp[df_temp["similarite"] >= min_score].sort_values("similarite", ascending=False).head(n_results)
    
    return resultats

# Analyse de tendances temporelles (si les dates sont disponibles)
def analyse_tendances(df):
    if 'date_creation' not in df.columns or df['date_creation'].isna().all():
        return None, None
    
    # Grouper par année et mois
    df['annee_mois'] = df['date_creation'].dt.strftime('%Y-%m')
    cours_par_mois = df.groupby(['annee_mois', 'category']).size().reset_index(name='count')
    
    # Créer un graphique de tendances
    fig = px.line(
        cours_par_mois, 
        x='annee_mois', 
        y='count', 
        color='category',
        title='Évolution du nombre de cours par catégorie',
        labels={'count': 'Nombre de cours', 'annee_mois': 'Période'}
    )
    
    # Distribution par niveau
    niveau_fig = px.histogram(
        df, 
        x='level', 
        color='category',
        title='Distribution des cours par niveau et catégorie',
        barmode='group'
    )
    
    return fig, niveau_fig

# Fonction pour analyser la distribution des longueurs de cours
def analyser_longueur_cours(df):
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df['nombre_mots'],
        x=df['category'],
        name='Distribution',
        boxmean=True,
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title_text='Distribution de la longueur des cours par catégorie',
        yaxis_title='Nombre de mots',
        xaxis_title='Catégorie',
        height=600
    )
    
    return fig

# Analyser la corrélation entre niveau et longueur
def analyser_correlation_niveau_longueur(df):
    fig = px.box(
        df, 
        x='level', 
        y='nombre_mots',
        color='level',
        title='Relation entre le niveau et la longueur du cours',
        labels={'nombre_mots': 'Nombre de mots', 'level': 'Niveau du cours'}
    )
    
    return fig

# Interface principale
def main():
    st.markdown('<h1 class="main-header">Système Avancé de Classification et d\'Analyse des Cours</h1>', unsafe_allow_html=True)

    # Barre latérale
    with st.sidebar:
        st.header("Configuration")
        
        # Upload de fichier JSON
        json_file = st.file_uploader("Importer un fichier JSON de cours", type="json")
        
        if not json_file:
            st.info("Veuillez importer un fichier JSON pour commencer.")
            st.markdown("""
            ### Format attendu:
            ```json
            [
                {
                    "title": "Titre du cours",
                    "description": "Description du cours",
                    "content": {
                        "paragraphs": ["paragraphe 1", "paragraphe 2"]
                    },
                    "level": "Débutant/Intermédiaire/Avancé",
                    "url": "https://...",
                    "date_creation": "2023-01-15"
                },
                ...
            ]
            ```
            """)
            return
    
    # Si fichier téléchargé
    if json_file:
        # Afficher un spinner pendant le chargement
        with st.spinner("Chargement et analyse des données en cours..."):
            # Chargement et prétraitement
            df = load_data(json_file)
            
            # Classification avancée
            df = classer_cours_avance(df)
            
            # Clustering
            n_clusters = min(5, len(df)) if len(df) > 0 else 1
            df, vectorizer, cluster_names = clustering_cours(df, n_clusters=n_clusters)
            
            # Encodage pour recherche sémantique
            model, embeddings = encoder_textes(df)
        
        # Onglets pour organiser l'interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Visualisation",
            "Recherche Sémantique",
            "Statistiques & Analyse",
            "Nuage de mots",
            "Explorer les données"
        ])
        
        # Onglet 1: Visualisation
        with tab1:
            st.markdown('<h2 class="sub-header">Visualisation interactive des cours</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="">', unsafe_allow_html=True)
                # Filtres
                st.subheader("Filtres")
                
                # Filtres de catégorie et niveau
                col_cat, col_niv = st.columns(2)
                with col_cat:
                    selected_categories = st.multiselect(
                        "Catégories",
                        df["category"].unique().tolist(),
                        default=[]
                    )
                
                with col_niv:
                    selected_levels = st.multiselect(
                        "Niveaux",
                        df["level"].unique().tolist(),
                        default=[]
                    )
                
                # Filtre par sous-catégorie
                selected_sous_cat = st.multiselect(
                    "Sous-catégories",
                    df["sous_categorie"].unique().tolist(),
                    default=[]
                )
                
                # Filtre par cluster
                selected_clusters = st.multiselect(
                    "Clusters",
                    df["cluster_name"].unique().tolist(),
                    default=[]
                )
                
                # Appliquer les filtres
                df_filtered = df.copy()
                
                if selected_categories:
                    df_filtered = df_filtered[df_filtered["category"].isin(selected_categories)]
                if selected_levels:
                    df_filtered = df_filtered[df_filtered["level"].isin(selected_levels)]
                if selected_sous_cat:
                    df_filtered = df_filtered[df_filtered["sous_categorie"].isin(selected_sous_cat)]
                if selected_clusters:
                    df_filtered = df_filtered[df_filtered["cluster_name"].isin(selected_clusters)]
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="">', unsafe_allow_html=True)
                # Affichage des métriques
                st.subheader("Métriques")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown(f'<div class="metric-value">{len(df_filtered)}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Cours sélectionnés</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f'<div class="metric-value">{df_filtered["category"].nunique()}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Catégories</div>', unsafe_allow_html=True)
                
                with metric_col3:
                    if "nombre_mots" in df_filtered.columns:
                        avg_words = int(df_filtered["nombre_mots"].mean())
                        st.markdown(f'<div class="metric-value">{avg_words}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Mots en moyenne</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-value">N/A</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Mots en moyenne</div>', unsafe_allow_html=True)
                        
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Choisir la visualisation
            viz_type = st.radio(
                "Type de visualisation",
                ["Par cluster", "Par catégorie"],
                horizontal=True
            )
            
            if viz_type == "Par cluster":
                st.plotly_chart(visualiser_clusters(df_filtered), use_container_width=True)
            else:
                st.plotly_chart(visualiser_categories(df_filtered), use_container_width=True)
        
        # Onglet 2: Recherche sémantique
        with tab2:
            st.markdown('<h2 class="sub-header">Recherche sémantique avancée</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                requete = st.text_input(
                    "Entrez votre recherche ici",
                    placeholder="Ex: programmation orientée objet avec Python, bases de données relationnelles..."
                )
            
            with col2:
                n_results = st.slider("Nombre de résultats", min_value=1, max_value=20, value=5)
                min_score = st.slider("Score minimum", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            
            # Filtres pour la recherche
            with st.expander("Filtres de recherche avancés"):
                col_cat, col_niv = st.columns(2)
                
                with col_cat:
                    search_categories = st.multiselect(
                        "Filtrer par catégorie",
                        df["category"].unique().tolist(),
                        default=[]
                    )
                
                with col_niv:
                    search_levels = st.multiselect(
                        "Filtrer par niveau",
                        df["level"].unique().tolist(),
                        default=[]
                    )
            
            if requete:
                with st.spinner("Recherche en cours..."):
                    # Filtrer d'abord selon les critères choisis
                    df_search = df.copy()
                    if search_categories:
                        df_search = df_search[df_search["category"].isin(search_categories)]
                    if search_levels:
                        df_search = df_search[df_search["level"].isin(search_levels)]
                    
                    # Si des filtres ont été appliqués, recalculer les embeddings
                    if len(df_search) != len(df):
                        search_model, search_embeddings = encoder_textes(df_search)
                        resultats = recherche_semantique(df_search, search_embeddings, search_model, requete, n_results, min_score)
                    else:
                        resultats = recherche_semantique(df, embeddings, model, requete, n_results, min_score)
                
                if len(resultats) > 0:
                    st.markdown('<h3 class="sub-header">Résultats de la recherche</h3>', unsafe_allow_html=True)
                    
                    # Afficher un score d'ensemble
                    avg_score = resultats["similarite"].mean()
                    st.markdown(f"<div class='highlight'>Score moyen de pertinence: {avg_score:.2f}</div>", unsafe_allow_html=True)
                    
                    # Trier les résultats similaires par catégorie
                    categories_similaires = resultats["category"].value_counts().reset_index()
                    categories_similaires.columns = ["category", "count"]
                    
                    if len(categories_similaires) > 1:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("### Distribution des résultats par catégorie")
                            cat_fig = px.pie(
                                categories_similaires, 
                                values="count", 
                                names="category",
                                hole=0.4
                            )
                            st.plotly_chart(cat_fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Résultats par niveau")
                            level_counts = resultats["level"].value_counts().reset_index()
                            level_counts.columns = ["level", "count"]
                            niv_fig = px.bar(
                                level_counts, 
                                x="level", 
                                y="count",
                                color="level"
                            )
                            st.plotly_chart(niv_fig, use_container_width=True)
                    
                    # Afficher les résultats
                    for i, (_, row) in enumerate(resultats.iterrows()):
                        with st.container():
                            st.markdown('<div class="">', unsafe_allow_html=True)
                            
                            # En-tête avec score
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### {row['title']}")
                            with col2:
                                st.markdown(f"<div class='highlight'>Score: {row['similarite']:.2f}</div>", unsafe_allow_html=True)
                            
                            # Métadonnées
                            st.markdown(f"""
                            **Catégorie:** {row['category']} | **Sous-catégorie:** {row['sous_categorie']}  
                            **Niveau:** {row.get('level', 'Non spécifié')} | **Cluster:** {row.get('cluster_name', 'Non classifié')}
                            """)
                            
                            # Description
                            description = str(row.get("description", "") or "").strip()
                            if description:
                                st.markdown(f"**Description:** {description}")
                            
                            # Lien vers le cours
                            url = str(row.get("url", "") or "").strip()
                            if url:
                                st.markdown(f"[Accéder au cours]({url})", unsafe_allow_html=True)
                            
                            # Extrait du contenu avec mise en évidence des mots-clés de la requête
                            if "texte_complet" in row:
                                with st.expander("Voir extrait du contenu"):
                                    texte = row["texte_complet"]
                                    # Mettre en surbrillance les mots de la requête
                                    for mot in requete.lower().split():
                                        if len(mot) > 3:  # Ignorer les mots courts
                                            texte = re.sub(
                                                f"(?i)({re.escape(mot)})",
                                                r"**\1**",
                                                texte
                                            )
                                    st.markdown(texte[:500] + "..." if len(texte) > 500 else texte)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Option pour exporter les résultats
                    if st.button("Exporter les résultats en CSV"):
                        csv = resultats[["title", "category", "sous_categorie", "level", "similarite"]].to_csv(index=False)
                        st.download_button(
                            label="Télécharger le CSV",
                            data=csv,
                            file_name=f"resultats_recherche_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Aucun résultat ne correspond à votre recherche. Essayez de modifier votre requête ou de réduire le score minimum.")
        
        # Onglet 3: Statistiques et Analyse
        with tab3:
            st.markdown('<h2 class="sub-header">Statistiques et Analyse approfondie</h2>', unsafe_allow_html=True)
            
            # Distribution des catégories
            st.markdown("### Distribution des cours par catégorie")
            cat_dist = df["category"].value_counts().reset_index()
            cat_dist.columns = ["category", "count"]
            
            fig_cat = px.bar(
                cat_dist, 
                x="category", 
                y="count",
                color="category",
                labels={"count": "Nombre de cours", "category": "Catégorie"},
                title="Répartition des cours par catégorie"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Analyse de la longueur des cours
            st.markdown("### Analyse de la longueur des cours")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(analyser_longueur_cours(df), use_container_width=True)
            
            with col2:
                st.plotly_chart(analyser_correlation_niveau_longueur(df), use_container_width=True)
            
            # Analyse de la complexité des cours
            st.markdown("### Analyse de la complexité des cours")
            
            # Calculer des métriques de complexité
            if "texte_complet" in df.columns:
                df["complexite_lexicale"] = df["texte_complet"].apply(
                    lambda x: len(set(word_tokenize(str(x).lower()))) / len(word_tokenize(str(x).lower())) if len(word_tokenize(str(x).lower())) > 0 else 0
                )
                
                fig_complexite = px.box(
                    df, 
                    x="category", 
                    y="complexite_lexicale",
                    color="level",
                    labels={"complexite_lexicale": "Complexité lexicale", "category": "Catégorie"},
                    title="Complexité lexicale par catégorie et niveau"
                )
                st.plotly_chart(fig_complexite, use_container_width=True)
            
            # Heat map de corrélation entre catégories et sous-catégories
            st.markdown("### Corrélation entre catégories et sous-catégories")
            
            # Créer une matrice pivot
            pivot = pd.crosstab(df["category"], df["sous_categorie"])
            
            # Normaliser les valeurs par ligne
            pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
            
            # Créer la heatmap
            fig_heatmap = px.imshow(
                pivot_norm,
                labels=dict(x="Sous-catégorie", y="Catégorie", color="Proportion"),
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig_heatmap.update_layout(title="Corrélation entre catégories et sous-catégories")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Onglet 4: Nuage de mots
        with tab4:
            st.markdown('<h2 class="sub-header">Analyse du vocabulaire</h2>', unsafe_allow_html=True)
            
            # Sélectionner la catégorie pour le nuage de mots
            word_cloud_cat = st.selectbox(
                "Sélectionnez une catégorie pour le nuage de mots",
                ["Toutes les catégories"] + df["category"].unique().tolist()
            )
            
            # Filtrer selon la catégorie choisie
            if word_cloud_cat == "Toutes les catégories":
                df_wordcloud = df.copy()
            else:
                df_wordcloud = df[df["category"] == word_cloud_cat]
            
            # Générer le nuage de mots
            with st.spinner("Génération du nuage de mots en cours..."):
                wordcloud_fig = generer_nuage_mots(df_wordcloud)
                st.pyplot(wordcloud_fig)
            
            # Analyse des mots-clés les plus fréquents
            st.markdown("### Mots-clés les plus fréquents")
            
            # Extraire et compter tous les mots-clés
            all_keywords = []
            for keywords in df_wordcloud["mots_cles"]:
                all_keywords.extend(keywords)
            
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(20)
            
            # Créer un DataFrame pour l'affichage
            df_keywords = pd.DataFrame(top_keywords, columns=["mot", "fréquence"])
            
            # Afficher un graphique à barres
            fig_keywords = px.bar(
                df_keywords,
                x="mot",
                y="fréquence",
                color="fréquence",
                labels={"mot": "Mot-clé", "fréquence": "Fréquence"},
                title=f"Top 20 des mots-clés les plus fréquents {'' if word_cloud_cat == 'Toutes les catégories' else 'pour ' + word_cloud_cat}"
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        # Onglet 5: Explorer les données
        with tab5:
            st.markdown('<h2 class="sub-header">Explorer les données brutes</h2>', unsafe_allow_html=True)
            
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                view_categories = st.multiselect(
                    "Filtrer par catégorie",
                    df["category"].unique().tolist(),
                    default=[],
                    key="view_categories"
                )

            with col2:
                view_levels = st.multiselect(
                    "Filtrer par niveau",
                    df["level"].unique().tolist(),
                    default=[],
                    key="view_levels"
                )

            with col3:
                view_sous_cat = st.multiselect(
                    "Filtrer par sous-catégorie",
                    df["sous_categorie"].unique().tolist(),
                    default=[],
                    key="view_sous_cat"
                )
                # Afficher le nombre total de cours
            
            # Appliquer les filtres
            df_view = df.copy()
            
            if view_categories:
                df_view = df_view[df_view["category"].isin(view_categories)]
            if view_levels:
                df_view = df_view[df_view["level"].isin(view_levels)]
            if view_sous_cat:
                df_view = df_view[df_view["sous_categorie"].isin(view_sous_cat)]
            
            # Sélectionner les colonnes à afficher
            colonnes_affichage = ["title", "category", "sous_categorie", "level", "cluster_name"]
            if "date_creation" in df_view.columns:
                colonnes_affichage.append("date_creation")
            
            # Option de recherche textuelle
            search_text = st.text_input("Rechercher par titre", "")
            if search_text:
                df_view = df_view[df_view["title"].str.contains(search_text, case=False, na=False)]
            
            # Afficher les données
            st.dataframe(
                df_view[colonnes_affichage],
                hide_index=True,
                use_container_width=True
            )
            
            # Option d'export
            if st.button("Exporter les données filtrées"):
                # Sélectionner toutes les colonnes sauf texte_complet et mots_cles pour réduire la taille
                export_cols = [col for col in df_view.columns if col not in ["texte_complet", "mots_cles"]]
                export_df = df_view[export_cols]
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger en CSV",
                    data=csv,
                    file_name=f"cours_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Exécution de l'application
if __name__ == "__main__":
    main()