from src.vectorization import vectorize_courses, vectorize_profile
from src.similarity import compute_similarity
from src.filtering import filter_courses
from src.sequencing import order_courses
from src.preprocessing import preprocess_courses

def recommend_courses(profile, raw_courses, top_k=5):
    courses = preprocess_courses(raw_courses)
    filtered = filter_courses(profile, raw_courses)
    
    if not filtered:
        return []


    course_vectors = vectorize_courses(filtered)
    profile_vector = vectorize_profile(profile)

    sims = compute_similarity(profile_vector, course_vectors)
    scored_courses = list(zip(filtered, sims))
    ordered_courses = order_courses(scored_courses)

    return [c for c, _ in ordered_courses[:top_k]]
