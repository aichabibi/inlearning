# users/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid

class Person(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='person_profile', null=True, blank=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, verbose_name="Nom complet")
    age = models.PositiveIntegerField(blank=True, null=True, verbose_name="Âge")
    GENDER_CHOICES = [
        ('M', 'Masculin'),
        ('F', 'Féminin'),
        ('O', 'Autre'),
        ('P', 'Préfère ne pas préciser'),
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True, verbose_name="Sexe")
    email = models.EmailField(verbose_name="Adresse e-mail")
    phone = models.CharField(max_length=20, blank=True, verbose_name="Numéro de téléphone")
    profile_picture = models.ImageField(upload_to='profile_pics', blank=True, null=True, verbose_name="Photo de profil")
    predicted_level = models.CharField(max_length=50, blank=True, null=True, verbose_name="Niveau prédit")

    def __str__(self):
        return self.name

class Preferences(models.Model):
    person = models.OneToOneField(Person, on_delete=models.CASCADE, related_name='preferences')
    LANGUAGE_CHOICES = [
        ('fr', 'Français'),
        ('en', 'Anglais'),
        ('es', 'Espagnol'),
        ('de', 'Allemand'),
        ('zh', 'Chinois'),
        ('ar', 'Arabe'),
    ]
    preferred_language = models.CharField(max_length=2, choices=LANGUAGE_CHOICES, verbose_name="Langue préférée")
    
    LEARNING_MODE_CHOICES = [
        ('video', 'Vidéo'),
        ('text', 'Support de cours écrit'),
        ('practice', 'Learning by doing'),
    ]
    learning_mode = models.CharField(max_length=10, choices=LEARNING_MODE_CHOICES, verbose_name="Mode d'apprentissage préféré")

    def __str__(self):
        return f"Préférences de {self.person.name}"

class Interest(models.Model):
    preferences = models.ForeignKey(Preferences, on_delete=models.CASCADE, related_name='interests')
    name = models.CharField(max_length=100, verbose_name="Centre d'intérêt")

    def __str__(self):
        return self.name

class AcademicBackground(models.Model):
    person = models.OneToOneField(Person, on_delete=models.CASCADE, related_name='academic_background')
    ACADEMIC_LEVEL_CHOICES = [
        ('bac', 'Baccalauréat'),
        ('licence', 'Licence'),
        ('master', 'Master'),
        ('doctorat', 'Doctorat'),
        ('autre', 'Autre'),
    ]
    highest_academic_level = models.CharField(max_length=10, choices=ACADEMIC_LEVEL_CHOICES, verbose_name="Niveau académique atteint")

    def __str__(self):
        return f"Parcours académique de {self.person.name}"

class FieldOfStudy(models.Model):
    academic_background = models.ForeignKey(AcademicBackground, on_delete=models.CASCADE, related_name='fields_of_study')
    field_name = models.CharField(max_length=100, verbose_name="Domaine d'étude")
    institution = models.CharField(max_length=255, verbose_name="Nom de l'institution")
    year_of_completion = models.PositiveIntegerField(verbose_name="Année d'obtention")

    def __str__(self):
        return f"{self.field_name} à {self.institution}"

class ProfessionalBackground(models.Model):
    person = models.OneToOneField(Person, on_delete=models.CASCADE)
    total_experience_years = models.IntegerField(verbose_name="Années d'expérience totale", null=True, blank=True)

    def __str__(self):
        return f"Expérience professionnelle de {self.person.name}"

class Job(models.Model):
    professional_background = models.ForeignKey(ProfessionalBackground, on_delete=models.CASCADE, related_name='jobs')
    position = models.CharField(max_length=100, verbose_name="Poste occupé")
    company = models.CharField(max_length=255, verbose_name="Nom de l'entreprise")
    start_date = models.DateField(verbose_name="Date de début")
    end_date = models.DateField(null=True, blank=True, verbose_name="Date de fin")
    is_current = models.BooleanField(default=False, verbose_name="Poste actuel")
    description = models.TextField(blank=True, verbose_name="Description des responsabilités")

    def __str__(self):
        return f"{self.position} chez {self.company}"

class Goals(models.Model):
    person = models.OneToOneField(Person, on_delete=models.CASCADE, related_name='goals')

    def __str__(self):
        return f"Objectifs de {self.person.name}"

class ShortTermGoal(models.Model):
    goals = models.ForeignKey(Goals, on_delete=models.CASCADE, related_name='short_term_goals')
    description = models.CharField(max_length=255, verbose_name="Objectif à court terme")

    def __str__(self):
        return self.description

class LongTermGoal(models.Model):
    goals = models.ForeignKey(Goals, on_delete=models.CASCADE, related_name='long_term_goals')
    description = models.CharField(max_length=255, verbose_name="Objectif à long terme")

    def __str__(self):
        return self.description