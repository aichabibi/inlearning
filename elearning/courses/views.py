from django.shortcuts import render
import os

# Create your views here.
# courses/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.db.models import Q
from django.urls import reverse_lazy, reverse
from .models import Course, Category, Enrollment, Lesson, LearningPath, Quiz, Question, Answer, QuizAttempt, UserAnswer
from .forms import CourseFilterForm
from django.http import JsonResponse
import requests
import json
import logging
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone


logger = logging.getLogger(__name__)

def convert_level_to_number(level):
    """
    Convertit le niveau textuel en nombre (1-5)
    Débutant -> 1
    Junior -> 2
    Intermédiaire -> 3
    Senior -> 4
    Expert -> 5
    """
    level_mapping = {
        'debutant': 1,
        'junior': 2,
        'intermediaire': 3,
        'senior': 4,
        'expert': 5,
        # Ajout des variantes avec accents
        'débutant': 1,
        'intermédiaire': 3
    }
    return level_mapping.get(level.lower(), 1)

class DashboardView(LoginRequiredMixin, View):
    template_name = 'courses/dashboard.html'
    
    def get(self, request):
        # Obtenir les inscriptions de l'utilisateur
        enrollments = Enrollment.objects.filter(user=request.user)
        
        # Obtenir les cours en cours (non terminés)
        in_progress_courses = enrollments.filter(completed=False)
        
        # Obtenir les cours terminés
        completed_courses = enrollments.filter(completed=True)
        
        # Obtenir les cours recommandés basés sur les préférences de l'utilisateur
        recommended_courses = []
        try:
            person = request.user.person_profile
            if hasattr(person, 'preferences'):
                # Obtenir le mode d'apprentissage préféré
                preferred_mode = person.preferences.learning_mode
                # Convertir le mode d'apprentissage au format attendu par le modèle Course
                mode_mapping = {
                    'video': 'video',
                    'text': 'text',
                    'practice': 'practice'
                }
                course_mode = mode_mapping.get(preferred_mode, '')
                
                # Filtrer les cours par mode d'apprentissage
                if course_mode:
                    recommended_courses = Course.objects.filter(
                        learning_mode=course_mode
                    ).exclude(
                        enrollments__user=request.user
                    )[:4]  # Limiter à 4 cours
                
        except Exception as e:
            # Si l'utilisateur n'a pas de profil ou de préférences, ne pas recommander de cours
            pass
        
        # Si aucun cours recommandé n'a été trouvé, montrer quelques cours populaires
        if not recommended_courses:
            recommended_courses = Course.objects.all().order_by('-enrollments')[:4]
        
        # Obtenir les catégories pour la navigation
        categories = Category.objects.all()
        
        context = {
            'in_progress_courses': in_progress_courses,
            'completed_courses': completed_courses,
            'recommended_courses': recommended_courses,
            'categories': categories,
        }
        
        return render(request, self.template_name, context)

class CourseListView(LoginRequiredMixin, ListView):
    model = Course
    template_name = 'courses/course_list.html'
    context_object_name = 'courses'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Course.objects.all()
        
        # Filtrer par la requête de recherche
        search_query = self.request.GET.get('search', '')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) | 
                Q(description__icontains=search_query) |
                Q(instructor__icontains=search_query)
            )
        
        # Filtrer par catégorie
        category_id = self.request.GET.get('category', '')
        if category_id and category_id.isdigit():
            queryset = queryset.filter(category_id=category_id)
        
        # Filtrer par niveau de difficulté
        difficulty = self.request.GET.get('difficulty', '')
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        
        # Filtrer par mode d'apprentissage
        learning_mode = self.request.GET.get('learning_mode', '')
        if learning_mode:
            queryset = queryset.filter(learning_mode=learning_mode)
        
        # Filtrer par prix (gratuit ou payant)
        price_filter = self.request.GET.get('price', '')
        if price_filter == 'free':
            queryset = queryset.filter(is_free=True)
        elif price_filter == 'paid':
            queryset = queryset.filter(is_free=False)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Ajouter le formulaire de filtre au contexte
        form = CourseFilterForm(self.request.GET)
        context['filter_form'] = form
        
        # Ajouter les paramètres de filtrage actuels pour la pagination
        context['current_filters'] = self.request.GET.copy()
        if 'page' in context['current_filters']:
            context['current_filters'].pop('page')
        
        # Ajouter toutes les catégories pour le filtre
        context['categories'] = Category.objects.all()
        
        # Ajouter les inscriptions de l'utilisateur
        user_enrollments = Enrollment.objects.filter(user=self.request.user).values_list('course_id', flat=True)
        context['user_enrollments'] = user_enrollments
        
        return context

class CourseDetailView(LoginRequiredMixin, DetailView):
    model = Course
    template_name = 'courses/course_detail.html'
    context_object_name = 'course'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Vérifier si l'utilisateur est inscrit à ce cours
        context['is_enrolled'] = Enrollment.objects.filter(
            user=self.request.user,
            course=self.object
        ).exists()
        
        # Obtenir toutes les leçons du cours
        context['lessons'] = self.object.lessons.all().order_by('order')
        
        # Si l'utilisateur est inscrit, obtenir sa progression
        if context['is_enrolled']:
            enrollment = Enrollment.objects.get(
                user=self.request.user,
                course=self.object
            )
            context['enrollment'] = enrollment
        
        # Obtenir d'autres cours de la même catégorie
        context['related_courses'] = Course.objects.filter(
            category=self.object.category
        ).exclude(id=self.object.id)[:4]
        
        return context

class EnrollCourseView(LoginRequiredMixin, View):
    def post(self, request, pk):
        course = get_object_or_404(Course, pk=pk)
        
        # Vérifier si l'utilisateur est déjà inscrit
        if Enrollment.objects.filter(user=request.user, course=course).exists():
            messages.warning(request, "Vous êtes déjà inscrit à ce cours.")
            return redirect('course_detail', pk=pk)
        
        # Créer une nouvelle inscription
        Enrollment.objects.create(
            user=request.user,
            course=course,
            progress=0,
            completed=False
        )
        
        messages.success(request, f"Vous êtes maintenant inscrit au cours : {course.title}")
        return redirect('course_detail', pk=pk)

class UnenrollCourseView(LoginRequiredMixin, View):
    def post(self, request, pk):
        course = get_object_or_404(Course, pk=pk)
        
        # Vérifier si l'utilisateur est inscrit
        enrollment = Enrollment.objects.filter(user=request.user, course=course).first()
        if not enrollment:
            messages.warning(request, "Vous n'êtes pas inscrit à ce cours.")
            return redirect('course_detail', pk=pk)
        
        # Supprimer l'inscription
        enrollment.delete()
        
        messages.success(request, f"Vous vous êtes désinscrit du cours : {course.title}")
        return redirect('course_list')

class LessonDetailView(LoginRequiredMixin, DetailView):
    model = Lesson
    template_name = 'courses/lesson_detail.html'
    context_object_name = 'lesson'
    
    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        
        # Vérifier si l'utilisateur est inscrit au cours
        is_enrolled = Enrollment.objects.filter(
            user=request.user,
            course=self.object.course
        ).exists()
        
        if not is_enrolled:
            messages.warning(request, "Vous devez être inscrit au cours pour accéder à cette leçon.")
            return redirect('course_detail', pk=self.object.course.pk)
        
        context = self.get_context_data(object=self.object)
        return self.render_to_response(context)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Ajouter le cours au contexte
        context['course'] = self.object.course
        
        # Obtenir l'inscription de l'utilisateur
        enrollment = Enrollment.objects.get(
            user=self.request.user,
            course=self.object.course
        )
        context['enrollment'] = enrollment
        
        # Obtenir la leçon précédente et suivante
        all_lessons = self.object.course.lessons.all().order_by('order')
        lesson_index = list(all_lessons).index(self.object)
        
        if lesson_index > 0:
            context['previous_lesson'] = all_lessons[lesson_index - 1]
        
        if lesson_index < len(all_lessons) - 1:
            context['next_lesson'] = all_lessons[lesson_index + 1]
        
        return context

class MarkLessonCompletedView(LoginRequiredMixin, View):
    def post(self, request, lesson_pk):
        lesson = get_object_or_404(Lesson, pk=lesson_pk)
        course = lesson.course
        
        # Vérifier si l'utilisateur est inscrit au cours
        enrollment = Enrollment.objects.filter(
            user=request.user,
            course=course
        ).first()
        
        if not enrollment:
            messages.warning(request, "Vous devez être inscrit au cours pour marquer une leçon comme complétée.")
            return redirect('course_detail', pk=course.pk)
        
        # Calculer le nouveau pourcentage de progression
        total_lessons = course.lessons.count()
        increment = 100 / total_lessons if total_lessons > 0 else 0
        enrollment.progress = min(enrollment.progress + increment, 100)
        
        # Vérifier si toutes les leçons sont complétées
        if enrollment.progress >= 100:
            enrollment.completed = True
            messages.success(request, f"Félicitations ! Vous avez terminé le cours : {course.title}")
        
        enrollment.save()
        
        # Rediriger vers la prochaine leçon si disponible
        next_lesson = course.lessons.filter(order__gt=lesson.order).order_by('order').first()
        if next_lesson:
            return redirect('lesson_detail', pk=next_lesson.pk)
        else:
            return redirect('course_detail', pk=course.pk)
        
class GenerateLearningPathView(LoginRequiredMixin, View):
    template_name = 'courses/generate_learning_path.html'

    def get(self, request):
        if not hasattr(request.user, 'person_profile'):
            return redirect('create_profile')
        return render(request, self.template_name)

    def post(self, request):
        try:
            subject = request.POST.get('subject')
            interests = json.loads(request.POST.get('interests', '[]'))
            provided_level = request.POST.get('level')

            if not subject:
                return JsonResponse({"success": False, "error": "Veuillez sélectionner une matière"})
            if not interests:
                return JsonResponse({"success": False, "error": "Veuillez sélectionner au moins un centre d'intérêt"})

            user_level = convert_level_to_number(provided_level) if provided_level else \
                         convert_level_to_number(request.user.person_profile.predicted_level)

            user_data = {
                "user_id": request.user.id,
                "language": subject,  # ✅ Ajouté pour correspondre à l'API
                "level": user_level,
                "subject": subject,
                "interests": interests
            }


            api_url = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000") + "/api/generate-learning-path"
            logger.info(f"[GenerateLearningPathView] Appel API Flask: {api_url}")
            logger.info(f"[GenerateLearningPathView] Payload: {json.dumps(user_data, indent=2)}")

            response = requests.post(api_url, json={"user_data": user_data}, timeout=30)


            if response.status_code != 200:
                logger.error(f"Erreur API Flask: {response.status_code} - {response.text}")
                return JsonResponse({"success": False, "error": "Erreur API Flask"})

            result = response.json()
            logger.info(f"[GenerateLearningPathView] Réponse complète API Flask: {json.dumps(result, indent=2)}")

            # ✅ Adapter au format que renvoie Flask
            if "recommended_courses" in result:
                learning_path_data = {
                    "language": user_data["language"],
                    "interests": user_data["interests"],
                    "modules": result["recommended_courses"]
                }

                # Sauvegarde en base
                learning_path = LearningPath.objects.create(
                    user=request.user,
                    language=learning_path_data["language"],
                    level=user_level,
                    interests=json.dumps(learning_path_data["interests"]),
                    modules=json.dumps(learning_path_data["modules"])
                )

                request.session["learning_path"] = learning_path_data
                return JsonResponse({
                    "success": True,
                    "message": "Parcours généré avec succès",
                    "redirect_url": reverse("learning_path_detail", kwargs={"path_id": learning_path.id})
                })

            # ❌ Si aucun parcours n’est renvoyé
            return JsonResponse({
                "success": False,
                "error": "L’API Flask n’a renvoyé aucun parcours d’apprentissage"
            })


        except Exception as e:
            logger.error(f"[GenerateLearningPathView] Exception: {str(e)}")
            return JsonResponse({"success": False, "error": "Erreur interne lors de la génération du parcours"})

class CourseCreateView(LoginRequiredMixin, CreateView):
    model = Course
    template_name = 'courses/course_form.html'
    fields = ['title', 'description', 'category', 'difficulty', 'learning_mode', 'is_free']
    success_url = reverse_lazy('course_list')

    def form_valid(self, form):
        form.instance.instructor = self.request.user.get_full_name() or self.request.user.username
        messages.success(self.request, 'Cours créé avec succès!')
        return super().form_valid(form)

class CourseUpdateView(LoginRequiredMixin, UpdateView):
    model = Course
    template_name = 'courses/course_form.html'
    fields = ['title', 'description', 'category', 'difficulty', 'learning_mode', 'is_free']
    success_url = reverse_lazy('course_list')

    def form_valid(self, form):
        messages.success(self.request, 'Cours mis à jour avec succès!')
        return super().form_valid(form)

class CourseDeleteView(LoginRequiredMixin, DeleteView):
    model = Course
    template_name = 'courses/course_confirm_delete.html'
    success_url = reverse_lazy('course_list')

    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Cours supprimé avec succès!')
        return super().delete(request, *args, **kwargs)

import json

class LearningPathDetailView(LoginRequiredMixin, View):
    template_name = 'courses/learning_path_detail.html'

    def get(self, request, path_id):
        learning_path = get_object_or_404(LearningPath, id=path_id, user=request.user)

        # ✅ Décoder modules JSON
        try:
            modules = json.loads(learning_path.modules)
        except Exception:
            modules = []

        # ✅ Décoder centres d’intérêt utilisateur
        try:
            interests_list = json.loads(learning_path.interests)
        except Exception:
            interests_list = []

        # ✅ Filtrer modules selon centres d’intérêt et niveau
        # ✅ Filtrer modules selon centres d’intérêt et niveau
        filtered_modules = []
        for m in modules:
            title = str(m.get("title", "")).lower()
            module_level = int(m.get("level", 0)) if str(m.get("level", "")).isdigit() else 0

            level_ok = module_level <= int(learning_path.level or 5)
            interest_ok = True
            if interests_list:
                interest_ok = any(interest.lower() in title for interest in interests_list)

            if level_ok and interest_ok:
                filtered_modules.append(m)

        if not filtered_modules:
            filtered_modules = modules

        modules = filtered_modules


        # ✅ Préparer les quiz
        quizzes_with_attempts = []
        for quiz in learning_path.quizzes.all():
            attempts = quiz.quizattempt_set.filter(user=request.user).order_by('-completed_at')
            quizzes_with_attempts.append({
                'quiz': quiz,
                'attempts': attempts
            })

        # ✅ Retourne au template avec interests_list bien défini
        return render(request, self.template_name, {
            'learning_path': learning_path,
            'modules': modules,
            'interests_list': interests_list,
            'quizzes_with_attempts': quizzes_with_attempts
        })

@require_http_methods(["GET"])
def claude_advice(request):
    try:
        parcours_id = request.GET.get('parcours_id')
        if not parcours_id:
            return JsonResponse({'success': False, 'error': 'ID du parcours manquant'}, status=400)

        learning_path = LearningPath.objects.get(id=parcours_id)
        
        # Préparer les données pour l'API Flask
        data = {
            'modules': learning_path.modules
        }
        
        # Appeler l'API Flask
        api_url = os.environ.get('FLASK_API_URL', 'http://127.0.0.1:5000')
        if not api_url.startswith('http'):
            api_url = f'http://{api_url}'
        api_url = f'{api_url}/api/claude-advice'
        
        logger.info(f"Tentative de connexion à l'API: {api_url}")
        logger.info(f"Données envoyées à l'API: {data}")
        
        try:
            response = requests.post(
                api_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=30  # Ajout d'un timeout de 30 secondes
            )
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            
            api_response = response.json()
            if api_response.get('success'):
                return JsonResponse({
                    'success': True,
                    'suggestions': api_response.get('suggestions', [])
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': api_response.get('error', 'Erreur inconnue')
                }, status=500)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de connexion à l'API Flask: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Erreur de connexion à l\'API: {str(e)}'
            }, status=503)
            
    except LearningPath.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Parcours non trouvé'
        }, status=404)
    except Exception as e:
        logger.error(f"Erreur dans claude_advice: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_POST
def improve_learning_path(request):
    try:
        data = json.loads(request.body)
        learning_path = data.get('learning_path')
        
        if not learning_path:
            return JsonResponse({
                'success': False,
                'error': 'Parcours d\'apprentissage manquant'
            }, status=400)

        # Appeler l'API Flask pour obtenir les améliorations
        flask_api_url = os.environ.get('FLASK_API_URL', 'http://127.0.0.1:5000')
        api_url = f"{flask_api_url}/api/generate-quiz"

        logger.info(f"[generate_quiz] Appel API Flask: {api_url} avec payload: {json.dumps(api_data, indent=2)}")

        try:
            response = requests.post(api_url, json=api_data, timeout=30)
        except Exception as e:
            logger.error(f"[generate_quiz] Impossible de contacter l'API Flask: {str(e)}")
            return JsonResponse({"success": False, "error": "Impossible de contacter l'API Flask"}, status=503)


        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            logger.error(f"Erreur API Flask: {response.status_code} - {response.text}")
            return JsonResponse({
                'success': False,
                'error': 'Erreur lors de l\'amélioration du parcours'
            }, status=500)

    except Exception as e:
        logger.error(f"Erreur dans improve_learning_path: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

class QuizView(LoginRequiredMixin, View):
    template_name = 'courses/quiz.html'

    def get(self, request, quiz_id):
        quiz = get_object_or_404(Quiz, id=quiz_id)
        
        # Vérifier si l'utilisateur a déjà une tentative en cours
        attempt = QuizAttempt.objects.filter(
            user=request.user,
            quiz=quiz,
            completed_at__isnull=True
        ).first()
        
        if not attempt:
            attempt = QuizAttempt.objects.create(
                user=request.user,
                quiz=quiz
            )
        
        questions = quiz.questions.all().order_by('order')
        return render(request, self.template_name, {
            'quiz': quiz,
            'questions': questions,
            'attempt': attempt
        })

    def post(self, request, quiz_id):
        quiz = get_object_or_404(Quiz, id=quiz_id)
        attempt = get_object_or_404(QuizAttempt, 
            user=request.user,
            quiz=quiz,
            completed_at__isnull=True
        )
        
        # Calculer le score
        total_points = 0
        earned_points = 0
        
        for question in quiz.questions.all():
            answer_id = request.POST.get(f'question_{question.id}')
            if answer_id:
                selected_answer = Answer.objects.get(id=answer_id)
                is_correct = selected_answer.is_correct
                
                # Enregistrer la réponse de l'utilisateur
                UserAnswer.objects.create(
                    attempt=attempt,
                    question=question,
                    selected_answer=selected_answer,
                    is_correct=is_correct
                )
                
                if is_correct:
                    earned_points += question.points
            
            total_points += question.points
        
        # Calculer le pourcentage
        score = (earned_points / total_points) * 100 if total_points > 0 else 0
        passed = score >= quiz.passing_score
        
        # Mettre à jour la tentative
        attempt.score = score
        attempt.passed = passed
        attempt.completed_at = timezone.now()
        attempt.save()
        
        return redirect('quiz_result', attempt_id=attempt.id)

class QuizResultView(LoginRequiredMixin, View):
    template_name = 'courses/quiz_result.html'
    
    def get(self, request, attempt_id):
        attempt = get_object_or_404(QuizAttempt, id=attempt_id, user=request.user)
        user_answers = attempt.user_answers.all().select_related('question', 'selected_answer')
        
        return render(request, self.template_name, {
            'attempt': attempt,
            'user_answers': user_answers
        })

@csrf_exempt
@require_POST
def generate_quiz(request):
    try:
        data = json.loads(request.body)
        learning_path_id = data.get('learning_path_id')
        
        if not learning_path_id:
            return JsonResponse({
                'success': False,
                'error': 'ID du parcours manquant'
            }, status=400)

        learning_path = get_object_or_404(LearningPath, id=learning_path_id)
        
        # Préparer les données pour l'API Flask
        api_data = {
            'learning_path': {
                'language': learning_path.language,
                'level': learning_path.level,
                'modules': learning_path.modules
            }
        }
        
        # Appeler l'API Flask
        flask_api_url = os.environ.get('FLASK_API_URL', 'http://127.0.0.1:5000')
        response = requests.post(
            f'{flask_api_url}/api/generate-quiz',
            json=api_data
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                quiz_data = result.get('quiz')
                
                # Créer le quiz dans la base de données
                quiz = Quiz.objects.create(
                    learning_path=learning_path,
                    title=quiz_data['title'],
                    description=quiz_data['description'],
                    passing_score=quiz_data['passing_score']
                )
                
                # Créer les questions et réponses
                for q_data in quiz_data['questions']:
                    question = Question.objects.create(
                        quiz=quiz,
                        text=q_data['text'],
                        points=q_data['points']
                    )
                    
                    for a_data in q_data['answers']:
                        Answer.objects.create(
                            question=question,
                            text=a_data['text'],
                            is_correct=a_data['is_correct']
                        )
                
                return JsonResponse({
                    'success': True,
                    'quiz_id': quiz.id,
                    'message': 'Quiz généré avec succès'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Erreur lors de la génération du quiz')
                }, status=500)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Erreur de communication avec l\'API'
            }, status=500)

    except Exception as e:
        logger.error(f"Erreur dans generate_quiz: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)