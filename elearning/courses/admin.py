from django.contrib import admin

# Register your models here.
# courses/admin.py
from django.contrib import admin
from .models import Category, Course, Enrollment, Lesson

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    search_fields = ('name',)

class LessonInline(admin.TabularInline):
    model = Lesson
    extra = 1
    ordering = ('order',)

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'instructor', 'difficulty', 'learning_mode', 'price', 'is_free', 'created_at')
    list_filter = ('category', 'difficulty', 'learning_mode', 'is_free')
    search_fields = ('title', 'instructor', 'description')
    inlines = [LessonInline]
    list_editable = ('price', 'is_free')
    date_hierarchy = 'created_at'

@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = ('user', 'course', 'enrolled_at', 'progress', 'completed')
    list_filter = ('completed',)
    search_fields = ('user__username', 'course__title')
    date_hierarchy = 'enrolled_at'

@admin.register(Lesson)
class LessonAdmin(admin.ModelAdmin):
    list_display = ('title', 'course', 'order')
    list_filter = ('course',)
    search_fields = ('title', 'content', 'course__title')
    ordering = ('course', 'order')