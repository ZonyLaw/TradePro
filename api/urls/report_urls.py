from django.urls import path
from api.views import report_view as views


urlpatterns = [

    path('<str:pk>/', views.getPredictions, name="get-prediction"),
]
