def filter_courses(profile, courses):
    preferences = profile.get("preferences", {})
    interests = preferences.get("interests", [])
    level = int(profile.get("academic_background", {}).get("highest_academic_level", 0))

    filtered = []

    for c in courses:
        titre = c.get("titre", "").lower()
        desc = c.get("description", "").lower()

        if any(kw.lower() in titre or kw.lower() in desc for kw in interests):
            if int(c.get("niveau", 0)) >= level:  # ✅ on veut des cours adaptés au niveau ou en dessous
                filtered.append(c)

    print(f"✅ {len(filtered)} cours trouvés pour les intérêts {interests} au niveau {level}")
    return filtered
