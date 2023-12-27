from django.forms import ModelForm
from .models import Price
from tickers.models import Ticker
from django import forms


class PriceForm(ModelForm):
    class Meta:
        model = Price
        fields = ['ticker', 'date', 'open', 'close', 'high', 'low', 'volume']
        labels = {'ticker': 'Ticker',   
                  'data': 'Date',
                  'open': 'Open',
                  'close': 'Close',
                  'high': 'High',
                  'low': 'Low',
                  'volume': 'Volume'
                  }

    def __init__(self, *args, **kwargs):
        super(PriceForm, self).__init__(*args, **kwargs)

        for name, field in self.fields.items():
            field.widget.attrs.update(
                {'class': 'input'},
            )
            

class FileUploadForm(forms.Form):
    file = forms.FileField(label='Select a file')
    

class ExportForm(forms.Form):
    folder_path = forms.CharField(label='Folder Path', max_length=255)
    

class PriceRangeForm(forms.Form):
    # ticker = forms.ModelChoiceField(queryset=Price.objects.values_list('ticker', flat=True).distinct())
    ticker = forms.ModelChoiceField(queryset=Ticker.objects.all(), to_field_name='symbol')
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    
    def __init__(self, *args, **kwargs):
        super(PriceRangeForm, self).__init__(*args, **kwargs)
        # Customize the label for the ticker field if needed
        self.fields['ticker'].label = 'Select Ticker'
