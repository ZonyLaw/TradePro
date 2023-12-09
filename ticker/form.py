from django.forms import ModelForm
from .models import Ticker
from django import forms


class TickerForm(ModelForm):
    class Meta:
        model = Ticker
        fields = ['symbol', 'full_name', 'info']

        labels = {'symbol': 'Symbol',
                  'full_name': 'Full name',
                  'info': 'Information',
                  }

    def __init__(self, *args, **kwargs):
        super(TickerForm, self).__init__(*args, **kwargs)

        for name, field in self.fields.items():
            field.widget.attrs.update(
                {'class': 'input'},
            )