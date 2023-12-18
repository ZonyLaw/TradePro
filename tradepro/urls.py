from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('ticker.urls')),
    path('', include('price.urls')),
    path('', include('ml_model.urls')),
    path('user/', include('user.urls')),
]
