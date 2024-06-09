from django.urls import path
from api.views import ticker_views as views


urlpatterns = [
    
    path('', views.getRoutes),
    path('tickers/', views.getTickers),
    path('tickers/<str:pk>/', views.getTicker),
    path('prices/<str:pk>/', views.getPrices),
]
