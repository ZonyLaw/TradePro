from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.home, name="home"),
    path('backend-control/', views.backend_control, name="backend-control"),
    path('login/', views.loginUser, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerUser, name="register"),
    path('users-list/', views.users_list, name="users-list"),
    path('profile/<str:pk>', views.userProfile, name="user-profile"),
    path('edit-profile/', views.editProfile, name="edit-profile"),
    path('account/', views.userAccount, name="account"),
    
    path('reset_password/', auth_views.PasswordResetView.as_view(template_name="users/reset_password.html"),
         name="reset_password"),
    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name="users/reset_password_sent.html"),
         name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="users/reset.html"),
         name="password_reset_confirm"),
    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name="users/reset_password_complete.html"),
         name="password_reset_complete"),
]