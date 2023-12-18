from django.db import models
import uuid

class Ticker(models.Model):
    symbol = models.CharField(max_length=20)
    full_name = models.CharField(max_length=200)
    info = models.TextField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True, editable=False)
    
    def __str__(self):
        return  self.symbol
    
