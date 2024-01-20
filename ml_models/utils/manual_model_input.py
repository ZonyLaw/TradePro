import pandas as pd
from pathlib import Path


def manual_price_input(form):

    # Assuming form.cleaned_data is a dictionary containing the required keys

    model_input = pd.DataFrame()

    # Add columns to the DataFrame one by one using loc
    model_input['open_close_diff_1'] = [form.cleaned_data['open'] - form.cleaned_data['close']]
    model_input['hr'] = [form.cleaned_data['hour']]
    model_input['ma100_4'] = [form.cleaned_data['ma100_4']]
    model_input['ma50_4_ma100_4_diff_4'] = [form.cleaned_data['ma50_4'] - form.cleaned_data['ma100_4']]
    model_input['up_bb20_low_bb20_diff_4'] = [form.cleaned_data['bb20_high_4'] - form.cleaned_data['bb20_low_4']]
    model_input['close_ma100_4_diff_4'] = [form.cleaned_data['close_4'] - form.cleaned_data['ma100_4']]
    model_input['ma20_4_ma50_4_diff_4'] = [form.cleaned_data['ma20_4'] - form.cleaned_data['ma50_4']]
    model_input['up_bb20_low_bb20_diff_1'] = [form.cleaned_data['bb20_high'] - form.cleaned_data['bb20_low']]
    model_input['close_ma50_1_diff_1'] = [form.cleaned_data['close'] - form.cleaned_data['ma50']]
    model_input['trend_strength_1'] = [form.cleaned_data['trend_strength_1']]
    model_input['open_close_diff1_lag1'] = [form.cleaned_data['open_lag1'] - form.cleaned_data['close_lag1']]
    model_input['bb_status_1'] = [form.cleaned_data['bb_status_1']]
    model_input['lagged_close_1'] = [form.cleaned_data['close_lag1']]

    return model_input