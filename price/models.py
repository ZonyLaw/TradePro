from django.db import models
import uuid
from ticker.models import Ticker

# Create your models here.
class Price(models.Model):
    ticker = models.ForeignKey(
        Ticker, blank=False, on_delete=models.CASCADE)
    date = models.DateField(default=0.0, null=True, blank=True)
    open = models.FloatField(default=0.0, null=True, blank=True)
    close = models.FloatField(default=0.0, null=True, blank=True)
    high = models.FloatField(default=0.0, null=True, blank=True)
    low = models.FloatField(default=0.0, null=True, blank=True)
    ask = models.FloatField(default=0.0, null=True, blank=True)
    bid = models.FloatField(default=0.0, null=True, blank=True)
    volume = models.FloatField(default=0.0, null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True,
                          primary_key=True, editable=False)
    
    def __str__(self):
        return f"{self.ticker} - {self.date}"
    