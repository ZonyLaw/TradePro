from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.utils import timezone  # Import Django's timezone module
from tickers.models import Ticker
from prices.models import Price
from api.serializers import TickerSerializer, PriceSerializer

class GetPricesAPITest(APITestCase):

    def setUp(self):
        # Create a ticker object
        self.ticker = Ticker.objects.create(symbol="GBPUSD")

        # Create some price objects related to the ticker with timezone-aware datetime
        Price.objects.create(ticker=self.ticker, date=timezone.datetime(2024, 7, 1), close=100)
        Price.objects.create(ticker=self.ticker, date=timezone.datetime(2024, 7, 2), close=110)
        Price.objects.create(ticker=self.ticker, date=timezone.datetime(2024, 7, 3), close=105)

        # URL for the getPrices API view
        self.url = reverse('get-prices', kwargs={'pk': self.ticker.id})


    def test_get_prices(self):
        # Make the GET request to the API view
        response = self.client.get(self.url)

        # Ensure the request was successful
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Ensure the response contains the correct ticker data
        ticker_serializer = TickerSerializer(self.ticker)
        self.assertEqual(response.data['id'], ticker_serializer.data['id'])
        self.assertEqual(response.data['symbol'], ticker_serializer.data['symbol'])

        # Ensure the response contains the correct prices data
        prices = Price.objects.filter(ticker__id=self.ticker.id).order_by('-date')
        price_serializer = PriceSerializer(prices, many=True)
        self.assertEqual(response.data['prices'], price_serializer.data)

