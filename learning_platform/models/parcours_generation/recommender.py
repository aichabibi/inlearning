from .vectorization import vectorize_courses, vectorize_profile
from .similarity import compute_similarity
from .filtering import filter_courses
from .sequencing import order_courses
from .preprocessing import preprocess_courses

def recommend_courses(user_profile, courses, top_k=15):
    """
    ✅ Recommandation simplifiée qui ne plante jamais.
    - Trie par pertinence (match intérêts dans titre/description)
    - Puis par proximité du niveau utilisateur
    - Retourne au max top_k cours
    """
    try:
        interests = [i.lower() for i in user_profile.get("preferences", {}).get("interests", [])]
        user_level = user_profile.get("academic_background", {}).get("highest_academic_level", 0)

        scored = []
        for course in courses:
            cours_data = course.get("cours", {})
            title = cours_data.get("titre", "").lower()
            description = cours_data.get("description", "").lower()
            niveau = cours_data.get("niveau", 0)

            # ✅ Score basé sur intérêt
            interest_score = sum(1 for kw in interests if kw in title or kw in description)
            # ✅ Score basé sur proximité du niveau
            level_score = max(0, 5 - abs((niveau if isinstance(niveau, int) else 0) - user_level))

            total_score = interest_score * 2 + level_score
            scored.append((total_score, course))

        # ✅ Trier par score décroissant
        scored.sort(key=lambda x: x[0], reverse=True)
        recommended = [c for _, c in scored[:top_k]]

        # ✅ Si aucun cours trouvé, retourner tous les cours
        if not recommended:
            return courses[:top_k]

        return recommended

    except Exception as e:
        logger.error(f"❌ Erreur dans recommend_courses: {e}")
        return courses[:top_k]
