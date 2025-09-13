# frontend/app.py
import os
import requests
import streamlit as st

API_BASE = os.getenv("STREAMLIT_API_URL", "http://localhost:5000").rstrip("/")

st.set_page_config(page_title="InLearning", page_icon="🎓", layout="centered")
st.title("InLearning – Tableau de bord (démo)")

with st.sidebar:
    st.subheader("API Backend")
    st.code(API_BASE, language="bash")
    if st.button("Vérifier /health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            r.raise_for_status()
            st.success(r.json())
        except requests.RequestException as e:
            st.error(f"Échec healthcheck : {e}")
            st.info(
                "Si tu es en Docker Compose : STREAMLIT_API_URL=http://backend:5000\n"
                "Si tu es en prod Render : STREAMLIT_API_URL=https://<backend>.onrender.com"
            )

st.write("### Améliorer un parcours avec Claude")
with st.form("claude_form"):
    context = st.text_area(
        "Contexte / profil / objectifs (colle ici les infos apprenant)",
        placeholder="Ex. Étudiante débutante en Python, objectif : data engineering, 4 semaines..."
    )
    question = st.text_input(
        "Question (optionnel)",
        placeholder="Ex. Comment rendre le parcours plus pratique ?"
    )
    submitted = st.form_submit_button("Générer le conseil")

if submitted:
    try:
        payload = {"context": context, "question": question}
        r = requests.post(f"{API_BASE}/api/claude-advice", json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        st.success("Réponse reçue ✅")
        st.write("**Source** :", j.get("source", "n/a"))
        st.write("**Conseil** :")
        st.write(j.get("advice") or j.get("hint") or j)
    except requests.RequestException as e:
        st.error(f"Erreur d'appel API : {e}")
        if "127.0.0.1" in str(e) or "localhost" in str(e):
            st.info(
                "Ton frontend essaie d'appeler `127.0.0.1:5000`. "
                "En prod, configure `STREAMLIT_API_URL` avec l’URL publique du backend."
            )
