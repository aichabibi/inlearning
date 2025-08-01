# courses/urls.py
from django.urls import path
from .views import (
    DashboardView, CourseListView, CourseDetailView,
    EnrollCourseView, UnenrollCourseView,
    LessonDetailView, MarkLessonCompletedView,
    CourseCreateView, CourseUpdateView, CourseDeleteView,
    GenerateLearningPathView, LearningPathDetailView,
    claude_advice, improve_learning_path,
    QuizView, QuizResultView,
    generate_quiz
)

print("\n=== Loading Courses URLs ===")

urlpatterns = [
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('', CourseListView.as_view(), name='course_list'),
    path('<int:pk>/', CourseDetailView.as_view(), name='course_detail'),
    path('<int:pk>/enroll/', EnrollCourseView.as_view(), name='enroll_course'),
    path('<int:pk>/unenroll/', UnenrollCourseView.as_view(), name='unenroll_course'),
    path('lesson/<int:pk>/', LessonDetailView.as_view(), name='lesson_detail'),
    path('lesson/<int:lesson_pk>/complete/', MarkLessonCompletedView.as_view(), name='mark_lesson_completed'),
    path('course/create/', CourseCreateView.as_view(), name='course_create'),
    path('course/<int:pk>/update/', CourseUpdateView.as_view(), name='course_update'),
    path('course/<int:pk>/delete/', CourseDeleteView.as_view(), name='course_delete'),
    path('generate-learning-path/', GenerateLearningPathView.as_view(), name='generate_learning_path'),
    path('learning-path/<int:path_id>/', LearningPathDetailView.as_view(), name='learning_path_detail'),
    path('api/claude-advice/', claude_advice, name='claude_advice'),
    path('api/improve-learning-path/', improve_learning_path, name='improve_learning_path'),
    path('api/generate-quiz/', generate_quiz, name='generate_quiz'),
    path('quiz/<int:quiz_id>/', QuizView.as_view(), name='quiz'),
    path('quiz/result/<int:attempt_id>/', QuizResultView.as_view(), name='quiz_result'),
]

print("Courses URLs patterns:")
for pattern in urlpatterns:
    print(f"- {pattern.pattern} -> {pattern.name}")
print("=== Courses URLs loaded ===\n")