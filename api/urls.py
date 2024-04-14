from django.urls import path
from . import views


urlpatterns = [
    
    path('', views.getRoutes),
    path('tickers/', views.getTickers),
    path('tickers/<str:pk>/', views.getTicker),
    path('tickers/ticker/<str:pk>/', views.getPrices),
]
