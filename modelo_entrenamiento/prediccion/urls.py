"""URL configuration for the prediccion app."""
from django.urls import path

codex/create-django-project-for-random-forest-model-qz73b2
from .views import predict, train_model

urlpatterns = [
    path("train/", train_model, name="train_model"),
    path("predict/", predict, name="predict"),

from .views import predict_view, train_view

urlpatterns = [
    path("train/", train_view, name="train_model"),
    path("predict/", predict_view, name="predict"),
 main
]
