# users/urls.py
from django.urls import path
from django.contrib.auth.views import LogoutView
from .views import (
    CustomLoginView, RegisterView, UserProfileView, 
    PersonDetailView, PersonListView, DashboardView
)

print("\n=== Loading Users URLs ===")

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/create/', UserProfileView.as_view(), name='create_profile'),
    path('profile/<uuid:pk>/', UserProfileView.as_view(), name='edit_profile'),
    path('profile/<uuid:pk>/detail/', PersonDetailView.as_view(), name='profile_detail'),
    path('profiles/', PersonListView.as_view(), name='profile_list'),
]

print("Users URLs patterns:")
for pattern in urlpatterns:
    print(f"- {pattern.pattern} -> {pattern.name}")
print("=== Users URLs loaded ===\n")