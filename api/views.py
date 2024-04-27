from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import TickerSerializer
from tickers.models import Ticker
from django.shortcuts import get_object_or_404


@api_view(['GET', 'POST'])
def getRoutes(request):
    
    routes = [
        {'GET':'/api/tickers'},
        {'GET':'/api/tickers/id'},
        {'GET':'/api/tickers/ticker/id'},
        
        {'POST':'/api/users/token'},
        {'POST':'/api/users/token/refresh'},
        
    ]
    
    return Response(routes)

@api_view(['GET'])
def getTickers(request):
    tickers = Ticker.objects.all()
    serializer = TickerSerializer(tickers, many=True)
    
    return Response(serializer.data)

@api_view(['GET'])
def getTicker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    serializer = TickerSerializer(ticker, many=False)
    
    return Response(serializer.data)

@api_view(['GET'])
def getPrices(request, pk):
    ticker = get_object_or_404(Ticker, id=pk)
    prices = ticker.price_set.all()
    
    profile = request.user.profile  # Assuming you have a one-to-one relationship between User and Profile
    user_timezone = profile.timezone
    
    # Convert prices to user's timezone
    prices_user_timezone = []
    for price in prices:
        localized_date = price.date.astimezone(user_timezone)
        prices_user_timezone.append({
            'id': price.id,
            'date': localized_date,
            'open': price.open,
            'close': price.close,
            'volume': price.volume,
            'trade': 'Buy' if price.open - price.close < 0 else 'Sell'
        })
    
    sorted_prices = sorted(prices_user_timezone, key=lambda x: x['date'], reverse=True)
    
    serializer = TickerSerializer(ticker)
    data = serializer.data
    data['prices'] = sorted_prices
    data['user_timezone'] = str(user_timezone)
    
    return Response(data)