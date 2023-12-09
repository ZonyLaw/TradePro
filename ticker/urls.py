from django.urls import path
from . import views

urlpatterns = [
    path('', views.tickers, name="tickers"),
    path('ticker/<str:pk>/', views.ticker, name="ticker"),
    path('create-ticker/', views.createTicker, name="create-ticker"),
    path('update-ticker/<str:pk>/', views.updateTicker, name="update-ticker"),
    path('delete-ticker/<str:pk>/', views.deleteTicker, name="delete-ticker"),

]