from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name="home"),
    path('login/', views.loginUser, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerUser, name="register"),
    path('profiles/', views.profiles, name="profiles"),
    path('profile/<str:pk>', views.userProfile, name="user-profile"),
    path('edit-profile/', views.editProfile, name="edit-profile"),
]