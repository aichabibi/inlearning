import json

def load_students():
    with open("app/data/students.json", "r") as f:
        return json.load(f)
