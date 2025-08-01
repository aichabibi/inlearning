from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(profile_vector, course_vectors):
    sims = cosine_similarity([profile_vector], course_vectors)[0]
    return sims
