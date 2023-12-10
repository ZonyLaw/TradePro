from django.urls import path
from . import views, utils


urlpatterns = [
    path('create-price/', views.createPrice, name="create-price"),
    path('update-price/<str:pk>/', views.updatePrice, name="update-price"),
    # path('import_prices/', utils.import_prices_from_csv, name='import_prices'),
    path('upload/', views.upload_file, name='upload-file'),

]