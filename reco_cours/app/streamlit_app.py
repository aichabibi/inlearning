import streamlit as st
st.set_page_config(page_title="Recommandation de Cours", layout="wide")

from src.recommender import recommend_courses
from src.preprocessing import load_courses, preprocess_courses

# 📦 Chargement des cours
@st.cache_data
def load_data():
    courses = load_courses("data/simplified_courses.json")
    return preprocess_courses(courses)

courses = load_data()

# Interface principale
st.title("🎓 Générateur de Parcours Personnalisé")

with st.form("student_form"):
    st.subheader("📚 Préférences")
    interests = st.multiselect(
        "Thèmes d'apprentissage",
        ["Python", "JavaScript", "PHP", "React", "Data Science", "Mobile Development", "UX Design", "React Tutorial"]
    )
    level = st.selectbox("Niveau académique", ["1", "2", "3", "4", "5"])
    mode = st.selectbox("Mode d'apprentissage préféré", ["en ligne", "présentiel", "mixte"])

    st.subheader("🎯 Objectifs")
    goals = st.text_area("Objectifs personnels (court et long terme)", height=100)

    submitted = st.form_submit_button("🔍 Générer les recommandations")

if submitted:
    if not interests:
        st.error("⚠️ Veuillez sélectionner au moins un thème.")
    else:
        # Profil simulé sans nom/email
        student_profile = {
            "preferences": {
                "preferred_language": "français",
                "learning_mode": mode,
                "interests": interests
            },
            "academic_background": {
                "highest_academic_level": int(level),
                "fields_of_study": []
            },
            "professional_background": {
                "total_experience_years": 0,
                "jobs": []
            },
            "goals": {
                "short_term_goals": [goals],
                "long_term_goals": [goals]
            }
        }

        #st.write("✅ Profil soumis :", student_profile)
        #st.write("✅ Exemple de cours :", courses[0])
        recommended = recommend_courses(student_profile, courses, top_k=5)

        if recommended:
            st.subheader("📚 Cours recommandés :")
            for course in recommended:
                with st.expander(course["titre"]):
                    st.markdown(f"**Durée** : {course.get('duree', 'non spécifiée')}")
                    st.markdown(f"**Niveau** : {course.get('niveau', 'non spécifié')}")
                    st.markdown(course.get("description", ""))
                    st.markdown(f"[📎 Lien vers le cours]({course.get('lien')})")
        else:
            st.warning("Aucun cours ne correspond à votre profil. Essayez d'autres choix.")
