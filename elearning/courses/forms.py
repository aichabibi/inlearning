# courses/forms.py
from django import forms
from .models import Course, Category, Lesson

class CourseFilterForm(forms.Form):
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Rechercher un cours'
        })
    )
    
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        required=False,
        empty_label="Toutes les catégories",
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    DIFFICULTY_CHOICES = [
        ('', 'Tous les niveaux'),
        ('beginner', 'Débutant'),
        ('intermediate', 'Intermédiaire'),
        ('advanced', 'Avancé')
    ]
    
    difficulty = forms.ChoiceField(
        choices=DIFFICULTY_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    LEARNING_MODE_CHOICES = [
        ('', 'Tous les modes'),
        ('video', 'Vidéo'),
        ('text', 'Support de cours écrit'),
        ('practice', 'Learning by doing')
    ]
    
    learning_mode = forms.ChoiceField(
        choices=LEARNING_MODE_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    PRICE_CHOICES = [
        ('', 'Tous les prix'),
        ('free', 'Gratuit'),
        ('paid', 'Payant')
    ]
    
    price = forms.ChoiceField(
        choices=PRICE_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )