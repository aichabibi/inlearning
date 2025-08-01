from flask import Blueprint, request, jsonify
from app.recommendation import get_recommendations

routes = Blueprint('routes', __name__)

@routes.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    recommendations = get_recommendations(data)
    
    return jsonify(recommendations)
