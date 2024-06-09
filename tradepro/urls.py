from django.contrib import admin
from django.urls import path, include
import django



urlpatterns = [
    path('admin/', admin.site.urls),
    path('tickers/', include('tickers.urls')),
    path('prices/', include('prices.urls')),
    path('ml_models/', include('ml_models.urls')),
    path('', include('users.urls')),
    path('api/tickers/', include('api.urls.ticker_urls')),
    path('api/users/', include('api.urls.user_urls')),

]
