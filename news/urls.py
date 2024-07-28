from django.urls import path
from . import views


urlpatterns = [
    path('upload-news/', views.upload_news, name='upload-news'),

]