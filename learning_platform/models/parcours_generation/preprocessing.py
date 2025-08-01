import json
import pandas as pd

def load_courses(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_profiles(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def flatten_course(course):
    # Si le champ 'cours' existe → ancien format
    if 'cours' in course and isinstance(course['cours'], dict):
        flat = course['cours']
        flat['url'] = course.get('url', '')
        return flat
    # Sinon → cours déjà à plat (nouveau format)
    elif 'titre' in course:
        return course
    else:
        print("❌ Format inconnu :", course)
        return None


def preprocess_courses(courses):
    return [c for c in (flatten_course(c) for c in courses) if c]
