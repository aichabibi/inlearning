# elearning/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from users.views import CustomLoginView

# Define the urlpatterns list directly
urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('users.urls')),
    path('courses/', include('courses.urls')),  # Simplified include
    path('', CustomLoginView.as_view(), name='home'),
]

if settings.DEBUG:
    import debug_toolbar
    urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]
    
# Add static/media URLs for development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Debug URL printing (optional)
if settings.DEBUG:
    import logging
    logger = logging.getLogger(__name__)
    
    def debug_urls():
        from django.urls import get_resolver
        resolver = get_resolver()
        print("\n=== All Available URLs ===")
        def _list_urls(patterns, prefix=''):
            for pattern in patterns:
                if hasattr(pattern, 'url_patterns'):
                    new_prefix = prefix + str(pattern.pattern)
                    print(f"\nURL Group: {new_prefix}")
                    _list_urls(pattern.url_patterns, new_prefix)
                else:
                    print(f"URL Pattern: {prefix}{pattern.pattern}")
                    if hasattr(pattern.callback, 'view_class'):
                        print(f"  View Class: {pattern.callback.view_class.__name__}")
                    else:
                        print(f"  View Function: {pattern.callback.__name__}")
        _list_urls(resolver.url_patterns)
        print("=== End URLs List ===\n")

    debug_urls()

# Debug: Print all registered URLs
print("\nRegistered URL patterns:")
def list_urls(patterns):
    for pattern in patterns:
        if hasattr(pattern, 'url_patterns'):
            # This is an include() pattern
            print(f"\nIncluded URLs from: {pattern.app_name if hasattr(pattern, 'app_name') else pattern.namespace}")
            list_urls(pattern.url_patterns)
        else:
            # This is a regular URL pattern
            print(f"- {pattern.pattern}")

list_urls(urlpatterns)

print("\n=== URLs loading complete ===\n")
