from django import forms

BB_CHOICES = [
    ('inside_bb', 'inside_bb'),
    ('lower_crossover', 'Lower Crossover'),
    ('upper_near', 'Upper Near'),
    ('upper_crossover', 'Upper Crossover'),
    ('lower_hit', 'Lower Hit'),
    ('lower_near', 'Lower Near'),
    ('upper_hit', 'Upper Hit'),
]


MODEL_LIST = [('v4', 'v4'), ('v5', 'v5'), ('1h_v5', '1h_v5'), ('1h_v5_trade', '1h_v5_trade')]

class ModelParameters(forms.Form):
    model_version = forms.ChoiceField(choices=MODEL_LIST, initial='v4', label='Select the model version')
    open = forms.FloatField(initial=0.0, label='Open Price')
    close = forms.FloatField(initial=0.0, label='Close Price')
    open_lag1 = forms.FloatField(initial=0.0, label='Open Price from previous hour')
    close_lag1 = forms.FloatField(initial=0.0, label='Close Price from previous hour')
    ma50 = forms.FloatField(initial=0.0, label='MA 50 Days for 1 hour')
    bb20_high = forms.FloatField(initial=0.0, label='Bollinger Bands 20 High for 1 hour')
    bb20_low = forms.FloatField(initial=0.0, label='Bollinger Bands 20 Low for 1 hour')
    
    bb20_high_4 = forms.FloatField(initial=0.0, label='Bollinger Bands 20 High for 4 hour')
    bb20_low_4 = forms.FloatField(initial=0.0, label='Bollinger Bands 20 Low for 4 hour')
    close_4 = forms.FloatField(initial=0.0, label='Close Price from 4 hour timeframe')
    ma20_4 = forms.FloatField(initial=0.0, label='MA 20 Days for 4 hour')
    ma50_4 = forms.FloatField(initial=0.0, label='MA 50 Days for 4 hour')
    ma100_4 = forms.FloatField(initial=0.0, label='MA 100 Days for 4 hour')
    
    hour = forms.IntegerField(initial=0, label='Hour')
    trend_strength_1 = forms.IntegerField(initial=0.0, label='What is the strength of trend')
    bb_status_1 = forms.ChoiceField(choices=BB_CHOICES, initial='inside_bb', label='Select the situation of the candle stick')

  
class ModelSelection(forms.Form):
      
      model_version = forms.ChoiceField(choices=MODEL_LIST, initial='v5', label='Select the model version')