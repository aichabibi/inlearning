import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import re
import numpy as np
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Définition des catégories et des mots-clés associés
CATEGORIES: Dict[str, List[str]] = {
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "tensorflow", "supervised", "unsupervised", "neural network", "deep learning"],
    "Deep Learning": ["deep learning", "cnn", "rnn", "lstm", "transformer", "pytorch", "autoencoder", "gan", "artificial intelligence"],
    "Data Science": ["data analysis", "pandas", "numpy", "big data", "statistics", "matplotlib", "seaborn", "data mining"],
    "Développement Web": ["html", "css", "javascript", "flask", "django", "react", "vue", "angular", "node.js", "web app", "frontend", "backend"],
    "Programmation Générale": ["variables", "functions", "loops", "syntax", "object-oriented", "C++", "Python", "Java", "Rust", "programming", "debugging"],
    "Bases de Données": ["sql", "mysql", "mongodb", "postgresql", "database", "oracle", "indexing", "query optimization", "transactions", "SQL Server"],
    "Cybersécurité": ["cybersecurity", "hacking", "pentest", "malware", "cryptography", "firewall", "phishing", "ransomware", "SOC", "forensics"],
    "Cloud Computing": ["aws", "azure", "gcp", "cloud computing", "virtual machines", "kubernetes", "docker", "terraform", "cloud storage", "serverless"],
    "Systèmes embarqués & IoT": ["iot", "arduino", "raspberry pi", "microcontroller", "firmware", "embedded systems", "sensors", "automation"],
    "Blockchain & Cryptomonnaies": ["bitcoin", "ethereum", "blockchain", "smart contract", "decentralized", "cryptocurrency", "nft", "web3"],
    "DevOps & CI/CD": ["devops", "continuous integration", "jenkins", "terraform", "docker", "kubernetes", "ci/cd", "automation", "ansible", "infrastructure as code"],
    "Systèmes d'exploitation": ["linux", "windows", "unix", "shell scripting", "bash", "powershell", "kernel", "sysadmin", "process management", "server administration"],
    "Algorithmique & Complexité": ["algorithms", "data structures", "big o", "graph theory", "dynamic programming", "sorting", "recursion", "graph", "computational complexity"],
    "Réseaux Informatiques": ["networking", "TCP/IP", "routing", "switching", "firewall", "VPN", "network security", "wireless", "ethernet"],
    "Intelligence Artificielle": ["artificial intelligence", "ai", "neural networks", "nlp", "computer vision", "reinforcement learning", "robotics", "chatbot"],
    "Autres Cours": []  # Catégorie par défaut
}

# Configuration
class Config:
    # Utilisation de Path pour la portabilité entre OS
    DOSSIER = Path(r"C:\Users\hp\Downloads\Inlearning-main (1)\Inlearning-main\Webscraping\output")
    FICHIERS_JSON = [
        "cpp_course_content.json", "python_course_content.json", "cours_classes.json", "cs_course_content.json", 
        "css_course_content.json", "git_course_content.json", "html_course_content.json", "java_course_content.json", 
        "jquery_course_content.json", "js_course_content.json", "mysql_course_content.json", "nodejs_course_content.json", 
        "numpy_course_content.json", "php_course_content.json", "react_course_content.json", "sql_course_content.json"
    ]
    OUTPUT_JSON = DOSSIER / "cours_classes_v2.json"
    OUTPUT_EXCEL = DOSSIER / "cours_classes_v2.xlsx"
    MAX_WORKERS = 4  # Nombre de workers pour le traitement parallèle
    MIN_SIMILARITY_THRESHOLD = 0.1  # Seuil minimal de similarité pour accepter une classification


class CourseClassifier:
    def __init__(self, categories: Dict[str, List[str]]):
        self.categories = categories
        # Créer une base de référence avec des textes par catégorie
        self.categories_texts = {cat: " ".join(mots) for cat, mots in categories.items() if mots}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Pré-traitement des vecteurs de catégories
        self._prepare_vectorizer()
    
    def _prepare_vectorizer(self) -> None:
        """Prépare le vectoriseur et les vecteurs de référence pour les catégories."""
        all_texts = list(self.categories_texts.values())
        if all_texts:
            self.vectorizer.fit(all_texts)
            self.category_vectors = self.vectorizer.transform(all_texts)
            self.category_names = list(self.categories_texts.keys())
        else:
            logger.warning("Aucune catégorie avec mots-clés trouvée.")
            self.category_vectors = None
            self.category_names = []
    
    def classify_text(self, text: str) -> Tuple[str, float]:
        """Classifie un texte et retourne la catégorie la plus proche avec le score de similarité."""
        if not text or not self.category_vectors.shape[0]:
            return "Autres Cours", 0.0
        
        # Nettoyer le texte (enlever les caractères spéciaux, normaliser)
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Transformer le texte en vecteur
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Calculer les similarités avec toutes les catégories
        similarities = cosine_similarity(text_vector, self.category_vectors)[0]
        
        # Trouver la meilleure correspondance
        best_match_idx = similarities.argmax()
        best_score = similarities[best_match_idx]
        
        if best_score < Config.MIN_SIMILARITY_THRESHOLD:
            return "Autres Cours", best_score
        
        return self.category_names[best_match_idx], best_score


class CourseProcessor:
    def __init__(self, classifier: CourseClassifier):
        self.classifier = classifier
    
    def process_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Traite un fichier JSON et classifie chaque cours."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning(f"Format inattendu dans {file_path} - attendu: liste, obtenu: {type(data)}")
                return []
            
            results = []
            for course in data:
                course_data = course.get("cours", {}) if isinstance(course, dict) else {}
                
                # Extraire les données du cours
                title = course_data.get("titre", "Titre inconnu")
                description = course_data.get("description", "")
                contents = course_data.get("contenus", {})
                paragraphs = contents.get("paragraphs", []) if isinstance(contents, dict) else []
                
                # Concaténer le texte pour la classification
                full_text = f"{title} {description} " + " ".join(paragraphs)
                
                # Classifier le cours
                category, score = self.classifier.classify_text(full_text)
                
                results.append({
                    "titre": title,
                    "categorie": category,
                    "score_similarite": round(score, 3),
                    "url": course_data.get("lien", "URL inconnue"),
                    "texte": full_text[:500] + "..." if len(full_text) > 500 else full_text  # Tronquer pour économiser de l'espace
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur en traitant {file_path}: {str(e)}")
            return []
    
    def process_all_files(self, files_list: List[str], base_dir: Path) -> List[Dict[str, Any]]:
        """Traite tous les fichiers JSON de manière parallèle."""
        all_courses = []
        file_paths = [base_dir / file for file in files_list if (base_dir / file).exists()]
        
        if not file_paths:
            logger.warning("Aucun fichier JSON trouvé dans le répertoire spécifié.")
            return []
        
        # Traitement parallèle des fichiers
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_file = {executor.submit(self.process_json_file, file_path): file_path for file_path in file_paths}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    logger.info(f"Traitement terminé pour {file_path.name}: {len(result)} cours trouvés")
                    all_courses.extend(result)
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
        
        return all_courses


class DataAnalyzer:
    @staticmethod
    def analyze_categories(courses: List[Dict[str, Any]]) -> Counter:
        """Analyse la répartition des cours par catégorie."""
        categories_count = Counter([course["categorie"] for course in courses])
        logger.info("\nRépartition des cours par catégorie:")
        
        # Exclure "Autres Cours" des logs
        for cat, count in categories_count.most_common():
            if cat != "Autres Cours":
                logger.info(f"{cat}: {count}")
        
        return categories_count
    
    @staticmethod
    def plot_distribution(categories_count: Counter, output_path: Optional[Path] = None) -> None:
        """Affiche et enregistre un graphique de répartition des catégories, sans la catégorie 'Autres Cours'."""
        # Créer une copie pour ne pas modifier l'original
        filtered_categories = categories_count.copy()
        
        # Supprimer la catégorie "Autres Cours" si elle existe
        if "Autres Cours" in filtered_categories:
            del filtered_categories["Autres Cours"]
        
        plt.figure(figsize=(12, 6))
        
        # Trier par nombre de cours décroissant
        categories = [cat for cat, _ in filtered_categories.most_common()]
        counts = [filtered_categories[cat] for cat in categories]
        
        # Créer le graphique à barres
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(categories)))
        bars = plt.bar(categories, counts, color=colors)
        
        # Ajouter des étiquettes
        plt.title("Répartition des cours par catégorie", fontsize=14)
        plt.xlabel("Catégories", fontsize=12)
        plt.ylabel("Nombre de cours", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Enregistrer l'image si un chemin est spécifié
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique enregistré sous {output_path}")
        
        plt.show()
    
    @staticmethod
    def cluster_uncategorized_courses(courses: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """Utilise K-means pour regrouper les cours non classifiés."""
        other_courses = [c for c in courses if c["categorie"] == "Autres Cours"]
        
        if len(other_courses) < 2:
            logger.info("Pas assez de cours non classifiés pour effectuer un clustering.")
            return {}
        
        # Ajuster le nombre de clusters si nécessaire
        n_clusters = min(len(other_courses), n_clusters)
        
        # Vectoriser les textes
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform([c["texte"] for c in other_courses])
        
        # Appliquer K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Organiser les cours par cluster
        clustered_courses = {}
        for i, label in enumerate(clusters):
            if label not in clustered_courses:
                clustered_courses[label] = []
            clustered_courses[label].append(other_courses[i])
        
        # Analyser les mots les plus importants par cluster
        for cluster_id, cluster_courses in clustered_courses.items():
            # Extraire les textes du cluster
            cluster_texts = [c["texte"] for c in cluster_courses]
            
            # Vectoriser pour trouver les mots importants
            cluster_vectorizer = TfidfVectorizer(max_features=10)
            cluster_vectorizer.fit_transform(cluster_texts)
            
            # Obtenir les 5 mots les plus importants
            indices = np.argsort(cluster_vectorizer.idf_)[:5]
            top_words = [feat for feat, _ in 
                          sorted(zip(cluster_vectorizer.get_feature_names_out(), 
                                     cluster_vectorizer.idf_), 
                                 key=lambda x: x[1])[:5]]
            
            logger.info(f"Cluster {cluster_id} ({len(cluster_courses)} cours) - Mots-clés: {', '.join(top_words)}")
        
        return clustered_courses


def main():
    try:
        # Vérifier si le dossier existe
        if not Config.DOSSIER.exists():
            logger.error(f"Le dossier {Config.DOSSIER} n'existe pas.")
            return
        
        # Créer le classifier et le processeur
        classifier = CourseClassifier(CATEGORIES)
        processor = CourseProcessor(classifier)
        
        # Traiter tous les fichiers JSON
        logger.info("Début du traitement des fichiers...")
        all_courses = processor.process_all_files(Config.FICHIERS_JSON, Config.DOSSIER)
        logger.info(f"Total de {len(all_courses)} cours traités.")
        
        # Sauvegarder les résultats
        with open(Config.OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_courses, f, indent=4, ensure_ascii=False)
        logger.info(f"Classification sauvegardée dans {Config.OUTPUT_JSON}")
        
        # Sauvegarder en Excel
        df = pd.DataFrame(all_courses)
        df.to_excel(Config.OUTPUT_EXCEL, index=False)
        logger.info(f"Classification exportée dans {Config.OUTPUT_EXCEL}")
        
        # Analyser les catégories
        analyzer = DataAnalyzer()
        categories_count = analyzer.analyze_categories(all_courses)
        
        # Afficher le graphique
        graph_path = Config.DOSSIER / "categories_distribution.png"
        analyzer.plot_distribution(categories_count, graph_path)
        
        # Clustering des cours non catégorisés
        logger.info("\nAnalyse des cours non classifiés...")
        clustered_courses = analyzer.cluster_uncategorized_courses(all_courses)
        
        # Suggérer des améliorations pour les cours non classifiés
        uncategorized_count = categories_count.get("Autres Cours", 0)
        if uncategorized_count > 0:
            logger.info(f"\n{uncategorized_count} cours n'ont pas été classifiés.")
            logger.info("Suggestions pour améliorer la classification:")
            logger.info("1. Ajouter plus de mots-clés aux catégories existantes")
            logger.info("2. Créer de nouvelles catégories basées sur les clusters identifiés")
            logger.info("3. Diminuer le seuil de similarité (actuellement: {Config.MIN_SIMILARITY_THRESHOLD})")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()