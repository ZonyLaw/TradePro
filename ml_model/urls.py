from django.urls import path
from . import views, utils


urlpatterns = [
    path('ml-predictions/', views.ml_predictions, name="ml-predictions"),

]