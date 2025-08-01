from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

# Create your models here.
# courses/models.py
from django.db import models
from users.models import Person

class Category(models.Model):
    name = models.CharField(max_length=100, verbose_name="Nom de la catégorie")
    description = models.TextField(blank=True, verbose_name="Description")
    icon = models.CharField(max_length=50, blank=True, verbose_name="Icône (classe Bootstrap)")
    
    class Meta:
        verbose_name = "Catégorie"
        verbose_name_plural = "Catégories"
        
    def __str__(self):
        return self.name

class Course(models.Model):
    title = models.CharField(max_length=200, verbose_name="Titre du cours")
    description = models.TextField(verbose_name="Description")
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True, blank=True)
    instructor = models.CharField(max_length=100, verbose_name="Instructeur")
    level = models.IntegerField(default=1)
    duration = models.PositiveIntegerField(verbose_name="Durée (en heures)")
    difficulty_choices = [
        ('beginner', 'Débutant'),
        ('intermediate', 'Intermédiaire'),
        ('advanced', 'Avancé')
    ]
    difficulty = models.CharField(max_length=20, choices=difficulty_choices, verbose_name="Niveau de difficulté")
    image = models.ImageField(upload_to='course_images', blank=True, null=True, verbose_name="Image du cours")
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, verbose_name="Prix")
    is_free = models.BooleanField(default=False, verbose_name="Gratuit")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Date de mise à jour")
    learning_mode_choices = [
        ('video', 'Vidéo'),
        ('text', 'Support de cours écrit'),
        ('practice', 'Learning by doing')
    ]
    learning_mode = models.CharField(max_length=20, choices=learning_mode_choices, verbose_name="Mode d'apprentissage")
    
    class Meta:
        verbose_name = "Cours"
        verbose_name_plural = "Cours"
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title

class Enrollment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="enrollments", verbose_name="Utilisateur")
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="enrollments", verbose_name="Cours")
    enrolled_at = models.DateTimeField(auto_now_add=True, verbose_name="Date d'inscription")
    progress = models.PositiveIntegerField(default=0, verbose_name="Progression (%)")
    completed = models.BooleanField(default=False, verbose_name="Terminé")
    
    class Meta:
        verbose_name = "Inscription"
        verbose_name_plural = "Inscriptions"
        unique_together = ['user', 'course']
    
    def __str__(self):
        return f"{self.user.username} - {self.course.title}"

class Lesson(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="lessons", verbose_name="Cours")
    title = models.CharField(max_length=200, verbose_name="Titre de la leçon")
    content = models.TextField(verbose_name="Contenu")
    order = models.PositiveIntegerField(default=1, verbose_name="Ordre")
    video_url = models.URLField(blank=True, null=True, verbose_name="URL de la vidéo")
    
    class Meta:
        verbose_name = "Leçon"
        verbose_name_plural = "Leçons"
        ordering = ['course', 'order']
        unique_together = ['course', 'order']
    
    def __str__(self):
        return f"{self.course.title} - {self.title}"

class LearningPath(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='learning_paths', verbose_name="Utilisateur")
    language = models.CharField(max_length=50, verbose_name="Langage")
    level = models.IntegerField(verbose_name="Niveau")
    interests = models.JSONField(verbose_name="Centres d'intérêt")
    modules = models.JSONField(verbose_name="Modules")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    
    class Meta:
        verbose_name = "Parcours d'apprentissage"
        verbose_name_plural = "Parcours d'apprentissage"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Parcours {self.language} - {self.user.username}"

class Quiz(models.Model):
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='quizzes')
    title = models.CharField(max_length=200)
    description = models.TextField()
    passing_score = models.IntegerField(default=70, validators=[MinValueValidator(0), MaxValueValidator(100)])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Quiz pour {self.learning_path.language} - {self.title}"

class Question(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    order = models.IntegerField(default=0)
    points = models.IntegerField(default=1)

    def __str__(self):
        return f"Question {self.order}: {self.text[:50]}..."

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='answers')
    text = models.TextField()
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return f"Réponse: {self.text[:50]}..."

class QuizAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)
    score = models.IntegerField(null=True, blank=True)
    passed = models.BooleanField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Tentative de {self.user.username} pour {self.quiz.title}"

class UserAnswer(models.Model):
    attempt = models.ForeignKey(QuizAttempt, on_delete=models.CASCADE, related_name='user_answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    selected_answer = models.ForeignKey(Answer, on_delete=models.CASCADE)
    is_correct = models.BooleanField()

    def __str__(self):
        return f"Réponse de {self.attempt.user.username} pour {self.question.text[:50]}..."