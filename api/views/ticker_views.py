from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ..serializers import TickerSerializer, PriceSerializer
from tickers.models import Ticker
from prices.models import Price
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
def getPrices2(request, pk):
    ticker = get_object_or_404(Ticker, id=pk)
    prices = ticker.price_set.all()
    
    #TODO: Need to add user authentication. Find out how to add token.
    
    prices_user_timezone = []
    for price in prices:
        prices_user_timezone.append({
            'id': price.id,
            'date': price.date,
            'open': price.open,
            'close': price.close,
            'volume': price.volume,
            'trade': 'Buy' if price.open - price.close < 0 else 'Sell'
        })
    
    sorted_prices = sorted(prices_user_timezone, key=lambda x: x['date'], reverse=True)
    
    serializer = TickerSerializer(ticker)
    data = serializer.data
    data['prices'] = sorted_prices
    
    return Response(data)


@api_view(['GET'])
def getPrices(request, pk):
    prices = Price.objects.filter(ticker__id=pk).order_by('-date')  # Filter using the primary key

    # Serialize filtered prices
    price_serializer = PriceSerializer(prices, many=True)
    serialized_prices = price_serializer.data

    # Retrieve and serialize the ticker
    ticker = get_object_or_404(Ticker, id=pk)
    ticker_serializer = TickerSerializer(ticker)
    data = ticker_serializer.data
    data['prices'] = serialized_prices  # Attach the sorted, serialized prices

    return Response(data)