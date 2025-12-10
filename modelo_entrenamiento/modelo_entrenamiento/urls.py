"""modelo_entrenamiento URL Configuration."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
codex/create-django-project-for-random-forest-model-5nj43e
    path('', RedirectView.as_view(pattern_name='train_model', permanent=False)),
    path('', include('prediccion.urls')),
]


    # Cuando entren a la raíz "/", redirige a la vista de entrenamiento
    path('', RedirectView.as_view(pattern_name='train_model', permanent=False)),
    # URLs de la app "prediccion" (train/, predict/, etc.)
    path('', include('prediccion.urls')),
]

# Para servir archivos subidos (CSV, etc.) en modo DEBUG
 main
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
