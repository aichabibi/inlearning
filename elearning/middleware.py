# elearning/elearning/middleware.py
import logging
from django.urls import resolve
from django.conf import settings

logger = logging.getLogger(__name__)

class URLDebugMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Suppression des logs d'URL
        response = self.get_response(request)
        return response