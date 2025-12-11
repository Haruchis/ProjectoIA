"""URL configuration for the prediccion app."""
from django.urls import path

from .views import predict, train_model

urlpatterns = [
    path("train/", train_model, name="train_model"),
    path("predict/", predict, name="predict"),
]
