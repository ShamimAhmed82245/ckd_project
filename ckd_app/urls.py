from django.urls import path
from . import views

urlpatterns = [
    path('compare/', views.compare_models, name='compare'),
    path('metrics/', views.metrics, name='metrics'),
    path('metric_bar/', views.metric_bar, name='metric_bar'),
    path('metric_pie/', views.metric_pie, name='metric_pie'),
    path('confusion_matrix/', views.confusion_matrix_view, name='confusion_matrix'),
    path('classification_report/', views.classification_report_view, name='classification_report'),
    path('', views.predict_ckd, name='predict_ckd'),
]
