import requests

url = "http://127.0.0.1:5000/api/generate-learning-path"
payload = {
    "user_data": {
        "subject": "Python",
        "interests": ["Web", "Data"],
        "level": "1"
    }
}
resp = requests.post(url, json=payload)
print(resp.status_code)
print(resp.json())
