from django.urls import path
from . import views


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),
    path('ml-report/', views.ml_report, name="ml-report"),
    path('ml-variability/', views.ml_variability, name="ml-variability"),
    path('ml-manual/', views.ml_manual, name="ml-manual"),
    path('export-model-results/', views.export_model_results, name="export-model-results"),
    path('delete-file/<str:filename>/', views.delete_file, name='delete_file'),
    path('export-file/<str:filename>/', views.export_file, name='export_file'),
]