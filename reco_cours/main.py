from src.recommender import recommend_courses
from src.preprocessing import load_courses, load_profiles

courses = load_courses("data/simplified_courses.json")
students = load_profiles("data/students_profiles.json")

for student in students:
    reco = recommend_courses(student, courses, top_k=3)
    print(f"\n🎓 Recommandations pour {student['person']['name']}:")
    for course in reco:
        print(f" - {course['cours']['titre']} ({course['cours'].get('duree', 'durée inconnue')})")
