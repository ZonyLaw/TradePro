from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('tickers/', include('tickers.urls')),
    path('prices/', include('prices.urls')),
    path('ml_models/', include('ml_models.urls')),
    path('', include('users.urls')),
    

]
