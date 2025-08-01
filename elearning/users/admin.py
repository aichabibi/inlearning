from django.contrib import admin

# Register your models here.
# users/admin.py
from django.contrib import admin
from .models import (
    Person, Preferences, Interest, AcademicBackground, FieldOfStudy,
    ProfessionalBackground, Job, Goals, ShortTermGoal, LongTermGoal
)

class InterestInline(admin.TabularInline):
    model = Interest
    extra = 1

class PreferencesAdmin(admin.ModelAdmin):
    inlines = [InterestInline]

class FieldOfStudyInline(admin.TabularInline):
    model = FieldOfStudy
    extra = 1

class AcademicBackgroundAdmin(admin.ModelAdmin):
    inlines = [FieldOfStudyInline]

class JobInline(admin.TabularInline):
    model = Job
    extra = 1

class ProfessionalBackgroundAdmin(admin.ModelAdmin):
    inlines = [JobInline]

class ShortTermGoalInline(admin.TabularInline):
    model = ShortTermGoal
    extra = 1

class LongTermGoalInline(admin.TabularInline):
    model = LongTermGoal
    extra = 1

class GoalsAdmin(admin.ModelAdmin):
    inlines = [ShortTermGoalInline, LongTermGoalInline]

admin.site.register(Person)
admin.site.register(Preferences, PreferencesAdmin)
admin.site.register(AcademicBackground, AcademicBackgroundAdmin)
admin.site.register(ProfessionalBackground, ProfessionalBackgroundAdmin)
admin.site.register(Goals, GoalsAdmin)