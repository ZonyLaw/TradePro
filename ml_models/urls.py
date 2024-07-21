from django.urls import path
from . import views


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),
    path('about/', views.about, name="about"),
    path('ml-report/', views.ml_report, name="ml-report"),
    path('ml-report2/', views.ml_report2, name="ml-report2"),
    path('ml-variability/', views.ml_variability, name="ml-variability"),
    path('ml-manual/', views.ml_manual, name="ml-manual"),
    path('ml-news-model/', views.ml_news_model, name="ml-news-model"),
    path('export-model-results/', views.export_model_results, name="export-model-results"),
    path('delete-file/<str:filename>/', views.delete_file, name='delete_file'),
    path('export-file/<str:filename>/', views.export_file, name='export_file'),
    path('model-accuracy/', views.display_model_accuracy, name='model-accuracy'),
    path('notes/', views.notes, name='notes'),
]