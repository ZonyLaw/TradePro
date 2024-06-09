from django.urls import path
from api.views import ticker_views as views


urlpatterns = [
    
    # path('/', views.getRoutes),
    path('', views.getTickers),
    path('<str:pk>/', views.getTicker),
    # path('prices/<str:pk>/', views.getPrices),
]
