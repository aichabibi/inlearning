from django.shortcuts import render
from django.http import JsonResponse
import os

# Create your views here.
# users/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.contrib import messages
from django.db import transaction
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic.detail import DetailView
from django.views.generic.list import ListView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required
import requests
import logging
import json

from .models import (
    Person, Preferences, Interest, AcademicBackground, FieldOfStudy,
    ProfessionalBackground, Job, Goals, ShortTermGoal, LongTermGoal
)
from .forms import (
    UserRegisterForm, LoginForm, PersonForm, PreferencesForm, InterestFormSet, 
    AcademicBackgroundForm, FieldOfStudyFormSet, ProfessionalBackgroundForm, 
    JobFormSet, GoalsForm, ShortTermGoalFormSet, LongTermGoalFormSet
)

# Configuration du logging
logger = logging.getLogger(__name__)

class CustomLoginView(LoginView):
    form_class = LoginForm
    template_name = 'users/login.html'
    redirect_authenticated_user = True

class RegisterView(CreateView):
    form_class = UserRegisterForm
    template_name = 'users/register.html'
    success_url = reverse_lazy('login')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Votre compte a été créé avec succès! Vous pouvez maintenant vous connecter.")
        return response

class UserProfileView(LoginRequiredMixin, View):
    template_name = 'users/user_profile_form.html'
    
    def get(self, request, pk=None):
        logger.info("=== UserProfileView.get() appelé ===")
        if pk:
            # Mode édition
            person = get_object_or_404(Person, pk=pk)
            # Vérifier que l'utilisateur est autorisé à modifier ce profil
            if person.user and person.user != request.user and not request.user.is_staff:
                messages.error(request, "Vous n'êtes pas autorisé à modifier ce profil.")
                return redirect('profile_list')
        else:
            # Mode création
            person, created = Person.objects.get_or_create(user=request.user)
        
        preferences, created = Preferences.objects.get_or_create(person=person)
        academic, created = AcademicBackground.objects.get_or_create(person=person)
        professional, created = ProfessionalBackground.objects.get_or_create(person=person)

        return render(request, self.template_name, {
            'person_form': PersonForm(instance=person),
            'preferences_form': PreferencesForm(instance=preferences),
            'academic_form': AcademicBackgroundForm(instance=academic),
            'professional_form': ProfessionalBackgroundForm(instance=professional),
        })

    def post(self, request, pk=None):
        logger.info("=== UserProfileView.post() appelé ===")
        logger.info(f"Données POST: {request.POST}")
        
        try:
            if pk:
                # Mode édition
                person = get_object_or_404(Person, pk=pk)
                # Vérifier que l'utilisateur est autorisé à modifier ce profil
                if person.user and person.user != request.user and not request.user.is_staff:
                    messages.error(request, "Vous n'êtes pas autorisé à modifier ce profil.")
                    return redirect('profile_list')
            else:
                # Mode création
                person, created = Person.objects.get_or_create(user=request.user)
            
            preferences, created = Preferences.objects.get_or_create(person=person)
            academic, created = AcademicBackground.objects.get_or_create(person=person)
            professional, created = ProfessionalBackground.objects.get_or_create(person=person)

            # Création des formulaires
            person_form = PersonForm(request.POST, request.FILES, instance=person)
            preferences_form = PreferencesForm(request.POST, instance=preferences)
            academic_form = AcademicBackgroundForm(request.POST, instance=academic)
            
            # Création du formulaire professional avec les données validées
            professional_data = request.POST.copy()
            professional_form = ProfessionalBackgroundForm(professional_data, instance=professional)
            
            # Vérification de la validité des formulaires
            forms_valid = all([
                person_form.is_valid(),
                preferences_form.is_valid(),
                academic_form.is_valid(),
                professional_form.is_valid()
            ])
            
            if not forms_valid:
                form_errors = {}
                if not person_form.is_valid():
                    form_errors.update(person_form.errors)
                if not preferences_form.is_valid():
                    form_errors.update(preferences_form.errors)
                if not academic_form.is_valid():
                    form_errors.update(academic_form.errors)
                if not professional_form.is_valid():
                    form_errors.update(professional_form.errors)
                
                logger.error(f"Erreurs de validation: {form_errors}")
                return JsonResponse({
                    'success': False,
                    'form_errors': form_errors,
                    'global_errors': ['Veuillez corriger les erreurs dans le formulaire.']
                })

            # Sauvegarde des formulaires
            with transaction.atomic():
                person = person_form.save()
                preferences = preferences_form.save()
                academic = academic_form.save()
                professional = professional_form.save()

            # Préparation des données pour l'API
            user_data = {
                'age': person.age,
                'gender': person.gender,
                'preferred_language': preferences.preferred_language,
                'learning_mode': preferences.learning_mode,
                'highest_academic_level': academic.highest_academic_level,
                'total_experience_years': professional.total_experience_years,
                'fields_of_study': academic.fields_of_study.first().field_name if academic.fields_of_study.exists() else None
            }
            
            logger.info(f"Données préparées pour l'API: {user_data}")
            
            # Appel de l'API Flask
            try:
                api_url = os.environ.get('FLASK_API_URL', 'http://flask_api:5000') + '/api/calculate-level'
                logger.info(f"Tentative de connexion à l'API: {api_url}")
                response = requests.post(api_url, json={'user_data': user_data})
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        # Sauvegarder le niveau prédit dans le profil
                        person.predicted_level = result['level']
                        person.save()
                        return JsonResponse({
                            'success': True,
                            'message': 'Profil mis à jour avec succès!',
                            'predicted_level': result['level']
                        })
                    else:
                        return JsonResponse({
                            'success': True,
                            'message': 'Profil mis à jour avec succès!',
                            'warning': 'Impossible de prédire le niveau.'
                        })
                else:
                    return JsonResponse({
                        'success': True,
                        'message': 'Profil mis à jour avec succès!',
                        'warning': 'Erreur lors de la prédiction du niveau.'
                    })
                    
            except requests.exceptions.RequestException as e:
                return JsonResponse({
                    'success': True,
                    'message': 'Profil mis à jour avec succès!',
                    'warning': f'Erreur de connexion à l\'API: {str(e)}'
                })

        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}")
            return JsonResponse({
                'success': False,
                'global_errors': [str(e)]
            })

class PersonDetailView(LoginRequiredMixin, DetailView):
    model = Person
    template_name = 'users/person_detail.html'
    context_object_name = 'person'
    login_url = 'login'

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        # Vérifier que l'utilisateur actuel est autorisé à voir ce profil
        if obj.user and obj.user != self.request.user and not self.request.user.is_staff:
            messages.error(self.request, "Vous n'êtes pas autorisé à voir ce profil.")
            return redirect('profile_list')
        return obj

class PersonListView(LoginRequiredMixin, ListView):
    model = Person
    template_name = 'users/person_list.html'
    context_object_name = 'persons'
    login_url = 'login'

    def get_queryset(self):
        if self.request.user.is_staff:
            # Les administrateurs peuvent voir tous les profils
            return Person.objects.all()
        else:
            # Les utilisateurs normaux ne peuvent voir que leur propre profil
            return Person.objects.filter(user=self.request.user)

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'users/dashboard.html'
    login_url = 'login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        # Récupérer le profil de l'utilisateur
        try:
            person = Person.objects.get(user=self.request.user)
            context['person'] = person
            context['predicted_level'] = person.predicted_level
        except Person.DoesNotExist:
            context['person'] = None
            context['predicted_level'] = None
        return context
