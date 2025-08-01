import os
import requests
import logging

logger = logging.getLogger(__name__)

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # modèle Hugging Face gratuit

BASE_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000")

def call_flask(endpoint, payload):
    """
    Envoie une requête POST à l'API Flask.
    """
    try:
        url = f"{BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}

        # Ajoute le token HuggingFace uniquement si défini
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Erreur API Flask : {str(e)}")
        return {"success": False, "error": str(e)}
