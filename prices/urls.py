from django.urls import path
from . import views, utils


urlpatterns = [
    path('create-price/', views.createPrice, name="create-price"),
    path('update-price/<str:pk>/', views.updatePrice, name="update-price"),
    path('upload-prices/', views.upload_prices, name='upload-prices'),
    path('export-prices/', views.export_prices, name='export-prices'),
    path('delete-price/<str:pk>/', views.delete_price, name='delete-price'),
     path('delete-prices/', views.delete_prices, name='delete-prices'),

]