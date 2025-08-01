from sklearn.feature_extraction.text import TfidfVectorizer

# Modèle de vectorisation TF-IDF
model = TfidfVectorizer()

def get_course_text(course):
    return f"{course.get('titre', '')}. {course.get('description', '')}"

def get_profile_text(profile):
    prefs = profile.get('preferences', {})
    goals = profile.get('goals', {})
    text = ' '.join(prefs.get('interests', []))
    text += ' ' + ' '.join(goals.get('short_term_goals', []))
    text += ' ' + ' '.join(goals.get('long_term_goals', []))
    return text

def vectorize_courses(courses):
    texts = [get_course_text(c) for c in courses]
    texts = [t for t in texts if t.strip()]  # supprime les textes vides
    if not texts:
        return []
    return model.fit_transform(texts).toarray()


def vectorize_profile(profile):
    text = get_profile_text(profile)
    if not text.strip():
        return []
    return model.transform([text]).toarray()[0]

