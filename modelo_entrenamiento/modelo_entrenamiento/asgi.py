"""ASGI config for modelo_entrenamiento project."""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'modelo_entrenamiento.settings')
application = get_asgi_application()
