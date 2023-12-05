from django.urls import path
from . import views

urlpatterns = [
    path('', views.tickers, name="tickers"),
    path('ticker/<str:pk>/', views.ticker, name="ticker"),
]