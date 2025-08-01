def filter_courses(user_profile, courses):
    """
    Filtre les cours en fonction des intérêts et du niveau de l'utilisateur.
    ✅ Compatible avec ton JSON (cours['cours']['titre'], cours['cours']['description'], cours['cours']['niveau'])
    """
    try:
        interests = [i.lower() for i in user_profile.get("preferences", {}).get("interests", [])]
        user_level = user_profile.get("academic_background", {}).get("highest_academic_level", 0)

        filtered = []
        for course in courses:
            cours_data = course.get("cours", {})
            title = cours_data.get("titre", "").lower()
            description = cours_data.get("description", "").lower()
            niveau = cours_data.get("niveau", 0)

            # ✅ Si le niveau du cours est <= niveau utilisateur + 1 → on garde
            level_ok = (isinstance(niveau, int) and niveau <= user_level + 1)

            # ✅ Si un des intérêts est dans le titre ou la description → on garde
            interest_ok = any(kw in title or kw in description for kw in interests)

            if interest_ok or level_ok:
                filtered.append(course)

        # ✅ Si aucun cours trouvé → retourne tous les cours
        if not filtered:
            return courses

        return filtered
    except Exception as e:
        logger.error(f"❌ Erreur dans filter_courses: {e}")
        return courses
