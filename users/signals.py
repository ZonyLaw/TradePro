from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from custom_user.models import User
from .models import Profile


# @receiver(post_save, sender=Profile)
def createProfile(sender, instance, created, **kwargs):
    if created:
        user = instance
        profile = Profile.objects.create(
            user=user,
            email=user.email,
        )
    
    
def deleteUser(sender, instance, **kwargs):
    try:
        user = instance.user
        if user:
            user.delete()
            print('Deleting user...')
    except User.DoesNotExist:
        print("Related user does not exist")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    
post_save.connect(createProfile, sender=User)
post_delete.connect(deleteUser, sender=Profile)