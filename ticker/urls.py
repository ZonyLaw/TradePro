from django.urls import path
from . import views

urlpatterns = [
    path('tickers/', views.tickers, name="tickers"),
    path('ticker/<str:pk>/', views.ticker, name="ticker"),
]