# users/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.forms import inlineformset_factory
from .models import (
    Person, Preferences, Interest, AcademicBackground, FieldOfStudy,
    ProfessionalBackground, Job, Goals, ShortTermGoal, LongTermGoal
)

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nom d\'utilisateur ou Email'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Mot de passe'}))

class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Prénom'}))
    last_name = forms.CharField(max_length=30, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nom'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'}))

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nom d\'utilisateur'}),
        }

    def __init__(self, *args, **kwargs):
        super(UserRegisterForm, self).__init__(*args, **kwargs)
        self.fields['password1'].widget = forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Mot de passe'})
        self.fields['password2'].widget = forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirmer le mot de passe'})

class PersonForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ['name', 'age', 'gender', 'email', 'phone', 'profile_picture']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
        }

class PreferencesForm(forms.ModelForm):
    class Meta:
        model = Preferences
        fields = ['preferred_language', 'learning_mode']
        widgets = {
            'preferred_language': forms.Select(attrs={'class': 'form-select'}),
            'learning_mode': forms.Select(attrs={'class': 'form-select'}),
        }

class InterestForm(forms.ModelForm):
    class Meta:
        model = Interest
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
        }

InterestFormSet = inlineformset_factory(
    Preferences, Interest, form=InterestForm,
    extra=2, can_delete=True
)

class AcademicBackgroundForm(forms.ModelForm):
    class Meta:
        model = AcademicBackground
        fields = ['highest_academic_level']
        widgets = {
            'highest_academic_level': forms.Select(attrs={'class': 'form-select'}),
        }

class FieldOfStudyForm(forms.ModelForm):
    class Meta:
        model = FieldOfStudy
        fields = ['field_name', 'institution', 'year_of_completion']
        widgets = {
            'field_name': forms.TextInput(attrs={'class': 'form-control'}),
            'institution': forms.TextInput(attrs={'class': 'form-control'}),
            'year_of_completion': forms.NumberInput(attrs={'class': 'form-control'}),
        }

FieldOfStudyFormSet = inlineformset_factory(
    AcademicBackground, FieldOfStudy, form=FieldOfStudyForm,
    extra=1, can_delete=True
)

class ProfessionalBackgroundForm(forms.ModelForm):
    class Meta:
        model = ProfessionalBackground
        fields = ['total_experience_years']
        widgets = {
            'total_experience_years': forms.NumberInput(attrs={'class': 'form-control'}),
        }

class JobForm(forms.ModelForm):
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
        required=False
    )
    
    class Meta:
        model = Job
        fields = ['position', 'company', 'start_date', 'end_date', 'is_current', 'description']
        widgets = {
            'position': forms.TextInput(attrs={'class': 'form-control'}),
            'company': forms.TextInput(attrs={'class': 'form-control'}),
            'start_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'is_current': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

JobFormSet = inlineformset_factory(
    ProfessionalBackground, Job, form=JobForm,
    extra=1, can_delete=True
)

class GoalsForm(forms.ModelForm):
    class Meta:
        model = Goals
        fields = []  # No fields needed as this is just a container

class ShortTermGoalForm(forms.ModelForm):
    class Meta:
        model = ShortTermGoal
        fields = ['description']
        widgets = {
            'description': forms.TextInput(attrs={'class': 'form-control'}),
        }

ShortTermGoalFormSet = inlineformset_factory(
    Goals, ShortTermGoal, form=ShortTermGoalForm,
    extra=2, can_delete=True
)

class LongTermGoalForm(forms.ModelForm):
    class Meta:
        model = LongTermGoal
        fields = ['description']
        widgets = {
            'description': forms.TextInput(attrs={'class': 'form-control'}),
        }

LongTermGoalFormSet = inlineformset_factory(
    Goals, LongTermGoal, form=LongTermGoalForm,
    extra=2, can_delete=True
)