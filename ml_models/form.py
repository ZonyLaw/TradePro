from django import forms

class ModelParameters(forms.Form):
    ma20 = forms.FloatField(initial=0.0, label='MA 20 Days')
    ma50 = forms.FloatField(initial=0.0, label='MA 50 Days')
    ma100 = forms.FloatField(initial=0.0, label='MA 100 Days')
    bb_high = forms.FloatField(initial=0.0, label='Bollinger Bands High')
    bb_low = forms.FloatField(initial=0.0, label='Bollinger Bands Low')
    open_price = forms.FloatField(initial=0.0, label='Open Price')
    close_price = forms.FloatField(initial=0.0, label='Close Price')
    high_price = forms.FloatField(initial=0.0, label='High Price')
    low_price = forms.FloatField(initial=0.0, label='Low Price')