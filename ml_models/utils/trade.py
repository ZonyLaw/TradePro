import os
import sys
import importlib.util
import datetime


def trade_direction(trade_diff):
    """_summary_
    This function determines the direction of the candle stick for any timeframe

    Args:
        trade_diff (float): the difference between open and current price.

    Returns:
        string: the trade direction of the candle stick
    """
    
    if trade_diff == 0:
        return "Neutral / Doji"
    elif trade_diff > 0:
        return "Buy"
    else:
        return "Sell"
    
