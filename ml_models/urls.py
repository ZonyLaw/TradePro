from django.urls import path
from . import views


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),
    path('ml-manual/', views.ml_manual, name="ml-manual"),

]