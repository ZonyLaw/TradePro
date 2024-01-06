


def trade_direction(trade_diff):
    """_summary_
    This is a function to determine the direction of the candle stick for specific timeframe

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