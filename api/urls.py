from django.urls import path
from . import views


urlpatterns = [
    
    path('', views.getRoutes),
    path('tickers/', views.getTickers),
    path('tickers/<str:pk>/', views.getTicker),
    path('prices/<str:pk>/', views.getPrices),
]
