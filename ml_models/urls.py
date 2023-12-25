from django.urls import path
from . import views


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),
    path('ml-parameters/', views.ml_parameters, name="ml-parameters"),
    path('ml-manual-run/', views.ml_manual_run, name="ml-manual-run"),

]