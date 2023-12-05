from django.contrib import admin

# Register your models here.
from .models import Ticker

admin.site.register(Ticker)
