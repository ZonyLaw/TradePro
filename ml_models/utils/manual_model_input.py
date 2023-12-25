import os
import joblib
import pandas as pd
from pathlib import Path
from .data_processing import scenario_reverse, scenario_continue
import csv

def manaul_price_input(form):
    
    
    open_close_diff_1 = form.cleaned_data['open'] - form.cleaned_data['close']
    open_close_diff1_lag1 = form.cleaned_data['open_lag1'] - form.cleaned_data['close_lag1']
    close_ma50_1_diff_1	= form.cleaned_data['close'] - form.cleaned_data['ma50']
    bb_status_1 = form.cleaned_data['bb_status_1']
    up_bb20_low_bb20_diff_1 = form.cleaned_data['bb20_high'] - form.cleaned_data['bb20_low']	
    
    trend_strength_1 = form.cleaned_data['trend_strength_1']
    
    lagged_close_1 = 	form.cleaned_data['close_lag1']
    hr = form.cleaned_data['hour']
    up_bb20_low_bb20_diff_4 = form.cleaned_data['bb20_high_4'] - form.cleaned_data['bb20_low_4']	
    ma50_4_ma100_4_diff_4	= form.cleaned_data['ma50_4'] - form.cleaned_data['ma100_4']	
    ma20_4_ma50_4_diff_4 = form.cleaned_data['ma20_4'] - form.cleaned_data['ma50_4']	
    close_ma100_4_diff_4 = form.cleaned_data['close_4'] - form.cleaned_data['ma100_4']	
    
    print(open_close_diff_1)

    model_input = []
    
    return model_input