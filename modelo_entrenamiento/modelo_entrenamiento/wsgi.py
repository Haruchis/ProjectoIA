"""WSGI config for modelo_entrenamiento project."""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'modelo_entrenamiento.settings')
application = get_wsgi_application()
