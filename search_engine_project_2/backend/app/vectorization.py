import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Fonction pour vectoriser les textes
def vectorize_courses(courses):
    descriptions = [course['description'] for course in courses]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    return X
