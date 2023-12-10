from django.contrib import admin
from .models import Price

class PriceAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'date', 'open', 'close', 'high', 'low','ask','bid', 'volume','open_next' )


admin.site.register(Price, PriceAdmin)
