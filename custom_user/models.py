from django_use_email_as_username.models import BaseUser, BaseUserManager


class User(BaseUser):
    objects = BaseUserManager()

    class Meta:
        abstract = False  # Ensure this is a concrete model