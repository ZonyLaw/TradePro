# forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from custom_user.models import User

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Required. Enter a valid email address.')

    class Meta:
        model = User
        fields = ['email', 'password1', 'password2']
