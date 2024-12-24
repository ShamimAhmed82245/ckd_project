from django.urls import path
from . import views

urlpatterns = [
    path("compare/", views.compare_models, name="compare_models"),
]
