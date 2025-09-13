# backend/app.py
import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
DATABASE_URL = os.getenv("DATABASE_URL")  # ex: postgresql://user:pass@host:5432/db
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
CORS(app, resources={r"/**": {"origins": CORS_ORIGINS}})

# --- DB engine (facultatif, pour healthcheck) ---
engine = None
if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    except Exception as e:
        # on ne crash pas l'app si la DB n'est pas prête
        print(f"[WARN] create_engine failed: {e}")


@app.get("/health")
def health():
    db_status = "unknown"
    if engine is None:
        db_status = "not_configured"
    else:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "up"
        except SQLAlchemyError as e:
            db_status = f"down: {str(e).splitlines()[0]}"

    return jsonify(
        status="ok",
        service="inlearning-backend",
        db=db_status,
    ), 200


@app.post("/api/claude-advice")
def claude_advice():
    """
    Corps attendu (JSON) :
    {
      "context": "profil utilisateur, parcours actuel, objectif...",
      "question": "optionnel : la question précise"
    }
    """
    payload = request.get_json(silent=True) or {}
    context = (payload.get("context") or "").strip()
    question = (payload.get("question") or "").strip()

    if not context and not question:
        return jsonify(error="Missing 'context' or 'question' in body"), 400

    # Si la clé Anthropic n'est pas fournie, renvoyer un fallback utile
    if not ANTHROPIC_API_KEY:
        hint = (
            "⚠️ ANTHROPIC_API_KEY absente : je renvoie un conseil générique. "
            "Configure cette variable d'environnement en production."
        )
        advice = (
            "Conseil générique : clarifie le niveau de l'apprenant, propose un "
            "parcours de 3 modules (bases → pratique guidée → mini-projet), "
            "et termine par un quiz d’auto-évaluation avec 5 questions."
        )
        return jsonify(source="fallback", hint=hint, advice=advice), 200

    # Appel API Anthropic Messages (Claude 3.x)
    try:
        anthropic_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            # version publique stable (ok pour 2024/2025)
            "anthropic-version": "2023-06-01",
        }

        user_prompt = (
            "Tu es un assistant pédagogique pour la plateforme InLearning. "
            "À partir du contexte et de la question (si fournie), propose un "
            "conseil concis et actionnable pour améliorer le parcours :\n\n"
            f"Contexte:\n{context}\n\nQuestion:\n{question}\n\n"
            "Réponds avec :\n- 3 axes d’amélioration\n- 1 plan de séance rapide\n"
            "- 3 KPIs pour mesurer la progression"
        )

        data = {
            "model": "claude-3-haiku-20240307",  # remplace par ton modèle si besoin
            "max_tokens": 600,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        resp = requests.post(anthropic_url, headers=headers, data=json.dumps(data), timeout=60)
        resp.raise_for_status()
        j = resp.json()

        # La réponse Claude est dans j["content"][0]["text"]
        text_blocks = j.get("content", [])
        advice_text = ""
        if text_blocks and isinstance(text_blocks, list):
            first = text_blocks[0]
            advice_text = first.get("text", "") if isinstance(first, dict) else str(first)

        if not advice_text:
            advice_text = "Je n’ai pas pu générer une réponse exploitable."

        return jsonify(source="anthropic", advice=advice_text), 200

    except requests.RequestException as e:
        return jsonify(error=f"Anthropic request failed: {e}"), 502
    except Exception as e:
        return jsonify(error=f"Unexpected error: {e}"), 500


@app.get("/api/ping")
def ping():
    return jsonify(pong=True), 200


if __name__ == "__main__":
    # Pour debug local uniquement (prod via gunicorn)
    app.run(host="0.0.0.0", port=5000, debug=True)
