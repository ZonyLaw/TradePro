from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name="home"),
    path('login/', views.loginUser, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerUser, name="register"),
    path('users-list/', views.users_list, name="users-list"),
    path('profile/<str:pk>', views.userProfile, name="user-profile"),
    path('edit-profile/', views.editProfile, name="edit-profile"),
    path('account/', views.userAccount, name="account"),
]