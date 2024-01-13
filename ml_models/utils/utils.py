import csv
from io import StringIO

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
    
def export_results(df):
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)

    # Write header
    csv_writer.writerow(['date', 'ticker', 'open', 'close', 'high', 'low'])

    # Write data
    for price in prices:
        csv_writer.writerow([price.date, price.ticker, price.open, price.close, price.high, price.low])

    # Get CSV data as a string
    csv_data = csv_buffer.getvalue()

    # Close the buffer
    csv_buffer.close()

    return csv_data