import json

def get_recommendations(data):
    # Charger les cours
    with open('app/data/courses.json', 'r') as f:
        courses = json.load(f)

    user_interests = set(data.get("interests", []))
    recommended_courses = []

    for course in courses:
        if user_interests.intersection(set(course["tags"])):
            recommended_courses.append(course)

    return recommended_courses
