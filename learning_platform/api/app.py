import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
from learning_platform.models.parcours_generation.recommender import recommend_courses
from learning_platform.models.parcours_generation.preprocessing import preprocess_courses
from learning_platform.models.parcours_generation.filtering import filter_courses
from learning_platform.models.parcours_generation.sequencing import order_courses
from learning_platform.models.course_pipeline.course_classifier import CourseClassifier, CATEGORIES
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
@app.route("/")
def home():
    return {"message": "InLearning API is running 🚀"}


CORS(app)

# 🔒 SÉCURISÉ: Clé API chargée depuis les variables d'environnement
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
logger.info(f"ANTHROPIC_API_KEY loaded: {'Yes' if ANTHROPIC_API_KEY else 'No'}")

# Vérification de la clé API au démarrage
if not ANTHROPIC_API_KEY:
    logger.error("⚠️ ANTHROPIC_API_KEY n'est pas définie dans les variables d'environnement!")
    logger.error("💡 Créez un fichier .env avec: ANTHROPIC_API_KEY=votre_cle_ici")
else:
    logger.info(f"✅ ANTHROPIC_API_KEY chargée (longueur: {len(ANTHROPIC_API_KEY)})")

DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

def call_claude_api(prompt, model_name=DEFAULT_CLAUDE_MODEL):
    """Appel sécurisé à l'API Claude avec fallback"""
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set")
        return "Clé API Claude manquante."

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_name,
            max_tokens=512,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.NotFoundError:
        logger.warning("⚠️ Modèle non trouvé, tentative avec claude-3-5-sonnet-20240620")
        return call_claude_api(prompt, model_name="claude-3-5-sonnet-20240620")
    except Exception as e:
        logger.error(f"Erreur Claude API: {e}")
        return f"Erreur lors de l'appel à Claude: {e}"


def load_models():
    models_dir = Path(__file__).parent.parent / 'models'
    logger.info(f"Loading models from: {models_dir}")
    
    try:
        # Load the student level model and its components
        student_level_model_path = models_dir / 'student_level' / 'best_model_rff.pkl'
        logger.info(f"Loading student level model from: {student_level_model_path}")
        
        if not student_level_model_path.exists():
            logger.error(f"Student level model file not found: {student_level_model_path}")
            raise FileNotFoundError(f"Model file not found: {student_level_model_path}")
            
        model_details = joblib.load(student_level_model_path)
        
        # Extract components with error handling
        model = model_details.get('model')
        scaler = model_details.get('scaler')
        label_encoder = model_details.get('label_encoder')
        feature_mappings = model_details.get('feature_mappings')
        features = model_details.get('features')
        
        if not all([model, scaler, label_encoder, feature_mappings, features]):
            raise ValueError("Missing required components in model file")
        
        logger.info("Student level model components loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading student level model: {str(e)}")
        raise
    
    try:
        # Load the course level model and vectorizer
        course_level_model_path = models_dir / 'course_pipeline' / 'level_model.joblib'
        vectorizer_path = models_dir / 'course_pipeline' / 'vectorizer.joblib'
        
        if not course_level_model_path.exists():
            logger.warning(f"Course level model not found: {course_level_model_path}")
            course_level_model = None
        else:
            course_level_model = joblib.load(course_level_model_path)
            logger.info("Course level model loaded successfully")
        
        if not vectorizer_path.exists():
            logger.warning(f"Vectorizer not found: {vectorizer_path}")
            vectorizer = None
        else:
            vectorizer = joblib.load(vectorizer_path)
            logger.info("Vectorizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading course level model or vectorizer: {str(e)}")
        course_level_model = None
        vectorizer = None
    
    # Initialize the course classifier with error handling
    try:
        course_classifier = CourseClassifier(CATEGORIES)
        logger.info("Course classifier initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing course classifier: {str(e)}")
        course_classifier = None
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_mappings': feature_mappings,
        'features': features,
        'course_level_model': course_level_model,
        'vectorizer': vectorizer,
        'course_classifier': course_classifier
    }

# Load models at startup with error handling
logger.info("Starting to load models...")
try:
    models = load_models()
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    models = None

@app.route('/api/calculate-level', methods=['POST'])
def calculate_user_level():
    if not models or not models.get('model'):
        return jsonify({
            'success': False,
            'error': 'Modèles non disponibles'
        }), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Données JSON manquantes'
            }), 400
            
        user_data = data.get('user_data', {})
        logger.info(f"Received user data: {user_data}")
        
        # Safely get mapped values with defaults
        def get_mapped_value(mapping_dict, key, default_value=0):
            try:
                if key in mapping_dict:
                    return mapping_dict[key]
                # Use first available value as default
                if mapping_dict:
                    default_val = next(iter(mapping_dict.values()))
                    logger.warning(f"Key '{key}' not found. Using default: {default_val}")
                    return default_val
                return default_value
            except Exception as e:
                logger.error(f"Error in get_mapped_value: {str(e)}")
                return default_value
        
        # Create input DataFrame with error handling
        try:
            input_data = pd.DataFrame({
                "age": [user_data.get('age', 25)],
                "gender": [get_mapped_value(models['feature_mappings']['gender'], 
                                         user_data.get('gender', ''))],
                "preferred_language": [get_mapped_value(models['feature_mappings']['preferred_language'],
                                                     user_data.get('preferred_language', ''))],
                "learning_mode": [get_mapped_value(models['feature_mappings']['learning_mode'],
                                                user_data.get('learning_mode', ''))],
                "highest_academic_level": [get_mapped_value(models['feature_mappings']['highest_academic_level'],
                                                         user_data.get('highest_academic_level', ''))],
                "total_experience_years": [user_data.get('total_experience_years', 0)],
                "fields_of_study": [get_mapped_value(models['feature_mappings']['fields_of_study'],
                                                  user_data.get('fields_of_study', ''))]
            })
        except KeyError as e:
            logger.error(f"Missing feature mapping: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Configuration manquante pour: {str(e)}'
            }), 500
        
        # Scale the features
        input_scaled = models['scaler'].transform(input_data[models['features']])
        
        # Make prediction
        prediction = models['model'].predict(input_scaled)
        prediction_label = models['label_encoder'].inverse_transform(prediction)[0]
        prediction_proba = models['model'].predict_proba(input_scaled)[0]
        
        logger.info(f"Prediction: {prediction_label}")
        
        return jsonify({
            'success': True,
            'level': prediction_label,
            'probabilities': prediction_proba.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error in calculate-level: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/api/predict-category', methods=['POST'])
def predict_category():
    if not models or not models.get('course_classifier'):
        return jsonify({
            'success': False,
            'error': 'Classificateur de cours non disponible'
        }), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Données JSON manquantes'
            }), 400
            
        course_data = data.get('course_data', {})
        logger.info(f"Received course data: {course_data}")
        
        # Extract text from course data
        title = course_data.get('title', '')
        description = course_data.get('description', '')
        content = course_data.get('content', '')
        
        # Combine text for classification
        full_text = f"{title} {description} {content}".strip()
        if not full_text:
            return jsonify({
                'success': False,
                'error': 'Aucun contenu à classifier'
            }), 400
        
        # Use the course classifier
        category, score = models['course_classifier'].classify_text(full_text)
        
        return jsonify({
            'success': True,
            'category': category,
            'confidence_score': float(score)
        })
        
    except Exception as e:
        logger.error(f"Error in predict-category: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict-level', methods=['POST'])
def predict_level():
    if not models or not models.get('course_level_model') or not models.get('vectorizer'):
        return jsonify({
            'success': False,
            'error': 'Modèle de niveau de cours non disponible'
        }), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Données JSON manquantes'
            }), 400
            
        course_data = data.get('course_data', {})
        
        # Extract and combine text
        title = course_data.get('title', '')
        description = course_data.get('description', '')
        content = course_data.get('content', '')
        full_text = f"{title} {description} {content}".strip()
        
        if not full_text:
            return jsonify({
                'success': False,
                'error': 'Aucun contenu à analyser'
            }), 400
        
        # Transform and predict
        text_vector = models['vectorizer'].transform([full_text])
        level = models['course_level_model'].predict(text_vector)
        
        return jsonify({
            'success': True,
            'level': int(level[0])
        })
        
    except Exception as e:
        logger.error(f"Error in predict-level: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-learning-path', methods=['POST'])
def generate_learning_path():
    try:
        logger.info("🟢 Début de generate_learning_path")
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False, 
                "error": "Données JSON manquantes"
            }), 400
            
        logger.info(f"📥 Données reçues: {data}")

        user_data = data.get('user_data', {})
        programming_language = user_data.get('subject', '').strip()
        interests = user_data.get('interests', [])
        user_level = user_data.get('level', '0')

        # Validation des données d'entrée
        if not programming_language:
            return jsonify({
                "success": False, 
                "error": "Le langage de programmation est requis"
            }), 400
            
        if not interests:
            return jsonify({
                "success": False, 
                "error": "Au moins un centre d'intérêt est requis"
            }), 400

        # Charger les cours depuis le fichier JSON
        courses_path = Path(__file__).parent.parent / 'data' / 'course_pipeline' / 'courses_fixed.json'
        if not courses_path.exists():
            logger.error(f"Fichier cours non trouvé: {courses_path}")
            return jsonify({
                "success": False, 
                "error": f"Fichier de cours non trouvé: {courses_path}"
            }), 500

        try:
            with open(courses_path, 'r', encoding='utf-8') as f:
                all_courses = json.load(f)
            logger.info(f"✅ {len(all_courses)} cours chargés")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Erreur lecture JSON: {str(e)}")
            return jsonify({
                "success": False, 
                "error": "Erreur lors du chargement des cours"
            }), 500

        if not all_courses:
            return jsonify({
                "success": False, 
                "error": "Aucun cours disponible"
            }), 500

        # Traitement sécurisé du niveau utilisateur
        try:
            highest_level = int(user_level) if str(user_level).isdigit() else 0
        except (ValueError, TypeError):
            highest_level = 0

        # Création du profil utilisateur
        user_profile = {
            "preferences": {"interests": [programming_language] + interests},
            "academic_background": {"highest_academic_level": highest_level}
        }

        # Pipeline de traitement avec gestion d'erreurs
        processed_courses = all_courses
        
        # Préprocessing
        try:
            if 'preprocess_courses' in globals():
                processed_courses = preprocess_courses(all_courses) or all_courses
                logger.info(f"✅ Préprocessing terminé: {len(processed_courses)} cours")
        except Exception as e:
            logger.warning(f"⚠️ Erreur préprocessing: {e}")

        # Filtrage
        try:
            if 'filter_courses' in globals():
                filtered_courses = filter_courses(user_profile, processed_courses) or processed_courses
                logger.info(f"✅ Filtrage terminé: {len(filtered_courses)} cours")
            else:
                filtered_courses = processed_courses
        except Exception as e:
            logger.warning(f"⚠️ Erreur filtrage: {e}")
            filtered_courses = processed_courses

        # Recommandation
        try:
            if 'recommend_courses' in globals():
                recommended_courses = recommend_courses(user_profile, filtered_courses, top_k=15)
                if not recommended_courses:
                    recommended_courses = filtered_courses[:15]
                logger.info(f"✅ Recommandation terminée: {len(recommended_courses)} cours")
            else:
                recommended_courses = filtered_courses[:15]
        except Exception as e:
            logger.warning(f"⚠️ Erreur recommandation: {e}")
            recommended_courses = filtered_courses[:15]

        # Fallback si aucun cours recommandé
        if not recommended_courses:
            logger.warning("⚠️ Aucun cours recommandé, utilisation des premiers cours disponibles")
            recommended_courses = all_courses[:15]

        # Formatage de la sortie avec gestion d'erreurs
        output = []
        for i, course in enumerate(recommended_courses):
            try:
                # Gestion flexible de la structure des cours
                if isinstance(course, dict):
                    if 'cours' in course:
                        # Structure imbriquée
                        course_info = course.get('cours', {})
                        title = course_info.get('titre', f'Cours {i+1}')
                        level = course_info.get('niveau', 0)
                    else:
                        # Structure plate
                        title = course.get('title', course.get('titre', f'Cours {i+1}'))
                        level = course.get('level', course.get('niveau', 0))
                    
                    url = course.get('url', '#')
                else:
                    # Si ce n'est pas un dictionnaire, créer une structure de base
                    title = f'Cours {i+1}'
                    level = 0
                    url = '#'
                
                output.append({
                    "title": str(title),
                    "level": int(level) if isinstance(level, (int, str)) and str(level).isdigit() else 0,
                    "url": str(url)
                })
            except Exception as e:
                logger.error(f"Erreur formatage cours {i}: {str(e)}")
                # Ajouter un cours par défaut en cas d'erreur
                output.append({
                    "title": f"Cours {i+1}",
                    "level": 0,
                    "url": "#"
                })

        logger.info(f"✅ Parcours généré avec {len(output)} cours")
        
        return jsonify({
            "success": True, 
            "recommended_courses": output,
            "total_courses": len(output),
            "user_profile": user_profile
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"🔥 ERREUR CRITIQUE dans generate_learning_path: {str(e)}")
        logger.error(f"TRACEBACK COMPLET:\n{error_trace}")
        
        return jsonify({
            "success": False,
            "error": f"Erreur interne: {str(e)}",
            "trace": error_trace if app.debug else None
        }), 500

@app.route('/api/improve-learning-path', methods=['POST'])
def improve_learning_path():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False, 
                'error': 'Données JSON manquantes'
            }), 400
            
        learning_path = data.get('learning_path')
        if not learning_path:
            return jsonify({
                'success': False, 
                'error': 'Parcours d\'apprentissage manquant'
            }), 400

        # Préparer le prompt pour Claude
        prompt = f"""En tant que conseiller pédagogique, analyse ce parcours d'apprentissage et propose des améliorations :

Parcours actuel :
- Langage : {learning_path.get('language', 'Non spécifié')}
- Niveau : {learning_path.get('level', 'Non spécifié')}
- Centres d'intérêt : {', '.join(learning_path.get('interests', []))}

Modules :
{json.dumps(learning_path.get('modules', []), indent=2, ensure_ascii=False)}

Analyse et propose :
1. Évaluation globale de la progression
2. Ajustements suggérés dans l'organisation
3. Cours complémentaires recommandés
4. Prérequis manquants

Garde tes suggestions concises et pratiques."""

        # Appel à l'API Claude
        improvements = call_claude_api(prompt)

        return jsonify({
            'success': True,
            'improvements': improvements,
            'original_path': learning_path
        })
        
    except Exception as e:
        logger.error(f"Error in improve-learning-path: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/claude-advice', methods=['POST'])
def claude_advice_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False, 
                'error': 'Données JSON manquantes'
            }), 400
            
        modules = data.get('modules')
        if not modules:
            return jsonify({
                'success': False, 
                'error': 'Modules manquants'
            }), 400

        # Prompt amélioré pour Claude
        prompt = f"""En tant que conseiller pédagogique, analyse ces modules et propose des conseils :

Modules actuels :
{json.dumps(modules, indent=2, ensure_ascii=False)}

IMPORTANT: Réponds UNIQUEMENT en JSON valide, sans texte supplémentaire.

Format JSON attendu :
{{
    "suggestions": [
        {{
            "title": "Titre du sujet suggéré",
            "relevance": "Explication de la pertinence",
            "level": "débutant/intermédiaire/avancé",
            "resources": [
                {{
                    "name": "Nom de la ressource",
                    "url": "Lien vers la ressource",
                    "type": "Type de ressource"
                }}
            ]
        }}
    ]
}}

Limite à 3-4 suggestions maximum."""

        # Appeler Claude avec la clé sécurisée
        if not ANTHROPIC_API_KEY:
            return jsonify({
                'success': False,
                'error': 'API Claude non configurée'
            }), 500
            
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0.7,
            system="Tu es un expert en pédagogie. Réponds UNIQUEMENT en JSON valide.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parser la réponse
        try:
            claude_response = json.loads(response.content[0].text)
            suggestions = claude_response.get('suggestions', [])
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON Claude: {e}")
            return jsonify({
                'success': False,
                'error': 'Erreur lors de l\'analyse des suggestions'
            }), 500

        return jsonify({
            'success': True,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"Error in claude-advice: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'Données JSON manquantes'
            }), 400
            
        learning_path = data.get('learning_path')
        if not learning_path:
            return jsonify({
                'success': False,
                'error': 'Parcours d\'apprentissage manquant'
            }), 400

        if not ANTHROPIC_API_KEY:
            return jsonify({
                'success': False,
                'error': 'API Claude non configurée'
            }), 500

        # Initialiser le client Anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Prompt pour Claude
        prompt = f"""Génère un quiz d'évaluation pour ce parcours d'apprentissage.
        IMPORTANT: Réponds UNIQUEMENT en JSON valide.

        Parcours:
        Langage: {learning_path.get('language', 'Non spécifié')}
        Niveau: {learning_path.get('level', 'Non spécifié')}
        Modules: {json.dumps(learning_path.get('modules', []), ensure_ascii=False)}

        Format JSON strict:
        {{
            "title": "Titre du quiz",
            "description": "Description du quiz",
            "passing_score": 70,
            "questions": [
                {{
                    "text": "Question",
                    "points": 1,
                    "answers": [
                        {{
                            "text": "Réponse",
                            "is_correct": true/false
                        }}
                    ]
                }}
            ]
        }}

        Crée 5-7 questions avec 4 réponses chacune (une seule correcte)."""

        # Appeler Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0.7,
            system="Tu es un expert en pédagogie. Réponds UNIQUEMENT en JSON valide.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parser la réponse
        try:
            quiz_data = json.loads(response.content[0].text)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON quiz: {e}")
            return jsonify({
                'success': False,
                'error': 'Erreur lors de la génération du quiz'
            }), 500
        
        return jsonify({
            'success': True,
            'quiz': quiz_data
        })

    except Exception as e:
        logger.error(f"Erreur génération quiz: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route de santé pour vérifier le statut de l'API
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': models is not None,
        'claude_api_configured': bool(ANTHROPIC_API_KEY)
    })

# Gestion d'erreur globale
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint non trouvé'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Erreur interne du serveur'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)