from rest_framework import serializers
from tickers.models import Ticker

class TickerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ticker
        fields = '__all__'
        