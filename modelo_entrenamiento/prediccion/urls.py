"""URL configuration for the prediccion app."""
from django.urls import path

from .views import predict_view, train_view

urlpatterns = [
    path("train/", train_view, name="train_model"),
    path("predict/", predict_view, name="predict"),
]
