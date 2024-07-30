from django import forms

class FileUploadForm(forms.Form):
    file = forms.FileField(label='Select a file')
    
    
class CurrencyFilterForm(forms.Form):
    CURRENCY_CHOICES = [
        ('USD', 'USD'),
        ('EUR', 'EUR'),
        ('GBP', 'GBP'),
        ('JPY', 'JPY'),
        # Add more currencies as needed
    ]
    
    currency = forms.ChoiceField(choices=CURRENCY_CHOICES, label="Select Currency")