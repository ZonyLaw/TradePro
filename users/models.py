import uuid

from django.db import models
from custom_user.models import User
from timezone_field import TimeZoneField


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=200, blank=True, null=True)
    first_name = models.CharField(max_length=200, blank=True, null=True)
    last_name = models.CharField(max_length=200, blank=True, null=True)
    email = models.EmailField(max_length = 500, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    timezone = TimeZoneField(default='UTC') 
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True,
                          primary_key=True, editable=False)
    
    def __str__(self):
        return  str(self.email)
