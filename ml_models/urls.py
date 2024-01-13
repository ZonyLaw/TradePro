from django.urls import path
from . import views


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),
    path('ml-manual/', views.ml_manual, name="ml-manual"),
    path('export-model-results/', views.export_model_results, name="export-model-results"),

]