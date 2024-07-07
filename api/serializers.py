from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken
from tickers.models import Ticker
from prices.models import Price
from custom_user.models import User


class UserSerializer(serializers.ModelSerializer):
    name=serializers.SerializerMethodField(read_only=True)
    isAdmin = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'name', 'isAdmin', 'token']
        
    def get__id(self, obj):
        return obj.id
        
    def get_isAdmin(self, obj):
        return obj.is_staff
        
    def get_name(self, obj):
        name = obj.first_name
        if name == '':
            name = obj.email
        
        return name
    
    
class UserSerializerWithToken(UserSerializer):
    token=serializers.SerializerMethodField(read_only=True)
    class Meta:
        model=User
        fields = ['id', 'username', 'email', 'name', 'isAdmin', 'token']
        
    def get_token(self, obj):
        token = RefreshToken.for_user(obj)
        return str(token.access_token)

class TickerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ticker
        fields = '__all__'

class PriceSerializer(serializers.ModelSerializer):
    trade = serializers.SerializerMethodField()

    class Meta:
        model = Price
        fields = ['id', 'date', 'open', 'close', 'volume', 'trade']

    def get_trade(self, obj):
        return 'Buy' if obj.open < obj.close else 'Sell'