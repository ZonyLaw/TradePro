from django.urls import path
from . import views


urlpatterns = [
    path('create-price/', views.createPrice, name="create-price"),
    path('update-price/<str:pk>/', views.updatePrice, name="update-price"),

]