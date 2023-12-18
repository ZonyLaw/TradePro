import pandas as pd
import numpy as np
from datetime import datetime
import os

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def import_data(file_path):
 
    # script_directory = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_directory, file_name)
    # file_path = file_path.replace("/", "\\")
    # file_path = f"{script_directory}/{file_name}"
    print(file_path)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the DataFrame
    return df

  
    
def calc_moving_average(df, timeframe):
    df[f'ma20_{timeframe}'] = df['close'].rolling(window=20).mean()
    df[f'ma50_{timeframe}'] = df['close'].rolling(window=50).mean()
    df[f'ma100_{timeframe}'] = df['close'].rolling(window=100).mean()
    
    df[f'dev20_{timeframe}'] = df['close'].rolling(window=20).std()
    df[f'dev50_{timeframe}'] = df['close'].rolling(window=50).std()
    df[f'dev100_{timeframe}'] = df['close'].rolling(window=100).std()

    return df

def calc_bb(df, timeframe):
    df[f'upper_bb20_{timeframe}'] = df[f'ma20_{timeframe}']+2*df[f'dev20_{timeframe}']
    df[f'lower_bb20_{timeframe}'] = df[f'ma20_{timeframe}']-2*df[f'dev20_{timeframe}']
    
    return df

def crossing_bb(df, timeframe):
    """
    This function determines if prices have crossover, hit the Bollinger Bands. The other case is moving inside the bb.
    Note: the 20_2 denotes it is based on 20 days average with a scaler of 2 to create this confidence level.

    Args:
        df (dataframe): DataFrame containing 'high', 'low', 'close', 'upper_bb20_2', 'lower_bb20_2'
    """

    df[f'bb_status_{timeframe}'] = 'inside_bb'  # Default value

    #check for crossing the BB
    conditions_upper_hit = ((df["high"] > df[f'upper_bb20_{timeframe}']) & (df["low"] < df[f'upper_bb20_{timeframe}'])) 
    conditions_lower_hit = (df["high"] > df[f'lower_bb20_{timeframe}']) & (df["low"] < df[f'lower_bb20_{timeframe}']) 
    
    conditions_upper_near = (abs(df[f'upper_bb20_{timeframe}'] - df["high"]) < 0.04)
    conditions_lower_near = (abs(df["low"] - df[f'lower_bb20_{timeframe}']) < 0.04)
    

    df.loc[conditions_upper_hit, f'bb_status_{timeframe}'] = 'upper_hit'
    df.loc[conditions_lower_hit, f'bb_status_{timeframe}'] = 'lower_hit'
    
    df.loc[conditions_upper_near, f'bb_status_{timeframe}'] = 'upper_near'
    df.loc[conditions_lower_near, f'bb_status_{timeframe}'] = 'lower_near'

    #check for closing the price outside the BB
    conditions_upper_region = df["close"] > df[f'upper_bb20_{timeframe}']
    conditions_lower_region = df["close"] < df[f'lower_bb20_{timeframe}']

    df.loc[conditions_upper_region, f'bb_status_{timeframe}'] = 'upper_crossover'
    df.loc[conditions_lower_region, f'bb_status_{timeframe}'] = 'lower_crossover'

    return df

def profit_calc(df, col, lag):
    """
    Does not need timeframe parameter!
    This function create the lag price and calculates the profit for the time period specified.

    Args:
        df (dataframe): dataframe contain the data
        col (string): price column to lag (e.g. high, low, open, close)
        lag (integer): time period to lag the column

    Returns:
        dataframe: containing the new columns
    """
    
    df[f'lagged_{col}_{lag}'] = df[col].shift(lag)
    df[f'pl_{col}_{lag}_hr'] = df[f'lagged_{col}_{lag}'] - df[col]
    
    #check if it should be a buy or sell given if entry was made at the point of date and time.
    buy_condition  = (df[f'pl_{col}_{lag}_hr'] > 0)

    
    #0 for sell and 1 for buy
    df[f'trade_{col}_{lag}_hr'] = 0 #'sell'
    df.loc[buy_condition, f'trade_{col}_{lag}_hr'] = 1 #'buy'

    
    
    return df

def date_split(df):
    """
    Does not need timeframe parameter!

    Args:
        df (dataframe): raw dataframe with the data column

    Returns:
        dataframe: df
    """

    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
    
    

    # Convert the entire 'date' column to datetime objects and apply the transformations
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y %H:%M")
    
    # Extract year, month and time
    df['year'] = df['date'].dt.strftime("%Y")
    df['month'] = df['date'].dt.strftime("%B")  # Full month name
    df['day'] = df['date'].dt.strftime("%d")
    df['time'] = df['date'].dt.strftime("%H:%M:%S")  # Hour:Minute:Second format

    # Extract the hour as an integer
    df['hr'] = df['date'].dt.hour
    df['4hr_tf'] = (df['hr'] // 4)*4
    
    # Get weekday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    df['weekday'] = df['date'].dt.weekday
    df['weekday_name'] = df['date'].dt.strftime("%A")  # Full weekday name
   
    return df


def price_difference(df, price1, price2, timeframe, label1=None, label2=None):
    """
    This takes the difference between two prices.

    Args:
        df (dataframe): This is the dataframe containing the data.
        price1 (string): This is the column name of the dataframe for the first price.
        price2 (string): This is the column name of the dataframe for the second price.
        label1 (string): Label for the first price (default is None).
        label2 (string): Label for the second price (default is None).

    Returns:
        None
    """

    if label1 is None:
        label1 = price1
        
    if label2 is None:
        label2 = price2

    df[f"{label1}_{label2}_diff_{timeframe}"] = df[price1] - df[price2]

    return df


def support_difference(df, price1, price2, timeframe, label1=None, label2=None):
    """
    This takes the difference between two prices against the peak and trough. If the value is negative it will add the positive price.

    Args:
        df (dataframe): This is the dataframe containing the data.
        price1 (string): This is the column name of the dataframe for the first price.
        price2 (string): This is the column name of the dataframe for the second price.
        label1 (string): Label for the first price (default is None).
        label2 (string): Label for the second price (default is None).

    Returns:
        None
    """

    if label1 is None:
        label1 = price1
        
    if label2 is None:
        label2 = price2

    
    sign_comparison = np.sign(df['price1']) * np.sign(df['price2'])

    df[f"{label1}_{label2}_diff_{timeframe}"] = np.where(sign_comparison == -1,
                                                        df['price1'] + df['price2'],
                                                        df['price1'] - df['price2'])

    return df


def price_relative_difference(df, close, upper_bound, lower_bound, timeframe, label1=None, label2=None):
    """
    This is to calculate the % price difference from the upper and lower bound. In other words, 
    measuring the relative distance from the boundary

    Args:
        df (dataframe): contining the prices, and bollinger bnad
        close (float): close price
        upper_bound (float): upper bollinger band price
        lower_bound (float): lower bollinger band price
        timeframe (integer): what this new calculated column represent
        label1 (string, optional): alternative name to the upper_bound
        label2 (string, optional): alternative name to the lower_bound

    Returns:
        _type_: _description_
    """
    
    df[f"{label1}_{label2}_rel_diff_{timeframe}"] = 0
    condition = (df[close] < df[upper_bound]) & (df[close] > df[lower_bound])
    df.loc[condition, f"{label1}_{label2}_rel_diff_{timeframe}"] = abs((df[close]-df[lower_bound]) / (df[upper_bound] - df[lower_bound]))
    
    return df


def trend_measure(df, timeframe):
    """
    The function indicate how strong a buy or sell
    This function needs one column:
     *   trade_close column where buy is 1 and sell is 0


    Args:
        df (dataframe): dataframe containing column trade_close and bb_status
        timeframe (integer): the timeframe to derive the strength for

    Returns:
        _type_: _description_
    """
    df[f'trend_strength_{timeframe}'] = 0
    #condition is to check that current trade is the same as previous
    condition = (df[f'trade_close_{timeframe}_hr'] == df[f'trade_close_{timeframe}_hr'].shift(1))

    cumulative_sum = 0
    for i in range(len(df)):
        if condition.iloc[i]:
            #if the trade is sell it will get a -1 value and decrement cumulatively
            if df.iloc[i, df.columns.get_loc(f'trade_close_{timeframe}_hr')] == 0:
                cumulative_sum += -1
            else:            
                #otherwise it will take on 1 which is a buy in the trade_close
                cumulative_sum += df.iloc[i, df.columns.get_loc(f'trade_close_{timeframe}_hr')]
        else:
            #this is when there is a change in the trade so different from previous trade (e.g. buy vs sell)
            cumulative_sum = 0

        df.iloc[i, df.columns.get_loc(f'trend_strength_{timeframe}')] = cumulative_sum

    return df


def trend_measure2(df, timeframe):
    """
    The function indicate how strong a buy or sell
    This function needs two columns:
     *   trade_close column where buy is 1 and sell is 0
     *   bb_status where it indicate if the price is close to the bollinger band

    Args:
        df (dataframe): dataframe containing column trade_close and bb_status
        timeframe (integer): the timeframe to derive the strength for

    Returns:
        _type_: _description_
    """
    df[f'trend_strength2_{timeframe}'] = 0
    #condition is to check that current trade is the same as previous
    condition = (df['trade_close_1_hr'] == df['trade_close_1_hr'].shift(1))

    cumulative_sum = 0
    for i in range(len(df)):
        if condition.iloc[i]:
            #if the trade is sell it will get a -1 value and decrement cumulatively
            if df.iloc[i, df.columns.get_loc(f'trade_close_{timeframe}_hr')] == 0:
                cumulative_sum += -1
            else:            
                #otherwise it will take on 1 which is a buy in the trade_close
                cumulative_sum += df.iloc[i, df.columns.get_loc(f'trade_close_{timeframe}_hr')]
        else:
            #this is when there is a change in the trade so different from previous trade (e.g. buy vs sell)
            cumulative_sum = 0
        
      
        if (df.iloc[i, df.columns.get_loc(f'bb_status_{timeframe}')] == "upper_near") | (df.iloc[i, df.columns.get_loc(f'bb_status_{timeframe}')] == "upper_hit"):
            cumulative_sum = 99
        elif (df.iloc[i, df.columns.get_loc(f'bb_status_{timeframe}')] == "lower_near") | (df.iloc[i, df.columns.get_loc(f'bb_status_{timeframe}')] == "lower_hit"):
            cumulative_sum = -99
        
        df.iloc[i, df.columns.get_loc(f'trend_strength2_{timeframe}')] = cumulative_sum

    return df


def create_4hr_table(df_1hr):
    
    """
    to create a 4 hr table based on the 1 hr table.
    """    
    
    # Filter rows based on the pattern
    df_4hr = df_1hr.copy()
    df_4hr = df_1hr[df_1hr['hr'] % 4 == 0].copy()
    df_4hr = df_4hr[['day','month', 'year','4hr_tf','open', 'high', 'low', 'close']]

    return df_4hr

def find_support(df):

    # Set a threshold for similarity
    threshold = 0.1  # Adjust this based on your definition of similarity

    # Count occurrences of similar values
    count_dict = {}
    for i in range(len(df)):
        current_value = df.at[i, 'close']
        similar_values = df[(df['close'] >= current_value - threshold) & (df['close'] <= current_value + threshold)]
        count = len(similar_values)
        count_dict[df.loc[i, 'close']] = count

    print(count_dict)
    # Print the results
    # for timestamp, count in count_dict.items():
    #     print(f"At {timestamp}, there are {count} similar values.")
    
    
def find_peaks_and_troughs(df, column_name, prominence=1, width=1):
    # Assuming you have a DataFrame with a column representing your curve
    curve_data = df[column_name].values

    # Find peaks
    peaks, _ = find_peaks(curve_data, prominence=prominence, width=width)

    # Invert the curve to find troughs
    inverted_curve_data = -curve_data

    # Find peaks in the inverted curve (which correspond to troughs in the original curve)
    troughs, _ = find_peaks(inverted_curve_data, prominence=prominence, width=width)

    # Plot the curve with detected peaks and troughs
    # plt.plot(curve_data)
    # plt.plot(peaks, curve_data[peaks], "o", label="peaks")
    # plt.plot(troughs, curve_data[troughs], "x", label="troughs")
    # plt.legend()
    # plt.show()

    # Get the peak and trough values and indices
    peak_values = curve_data[peaks]
    peak_indices = peaks
    trough_values = curve_data[troughs]
    trough_indices = troughs

    # print(f"The local peak values are {peak_values} at indices {peak_indices}.")
    # print(f"The local trough values are {trough_values} at indices {trough_indices}.")
    
    return peak_values, trough_values


def tag_peak_trough_values(df, column_name, peak_values, trough_values, proximity_threshold, timeframe):
    # Create a new column for the peak indicator
    df[f'support_indicator_{timeframe}'] = 0

    # Tag values near peaks
    for peak_value in peak_values:
        mask = (np.abs(df[column_name] - peak_value) <= proximity_threshold)
        df.loc[mask, f'support_indicator_{timeframe}'] = peak_value

    # Tag values near troughs
    for trough_value in trough_values:
        mask = (np.abs(df[column_name] - trough_value) <= proximity_threshold)
        df.loc[mask, f'support_indicator_{timeframe}'] = -trough_value

    return df



def main(file_path):
    file_name = "inputs/USDJPY_prices.csv"
    df = import_data(file_path)
    df = calc_moving_average(df,1)
    df = calc_bb(df,1)
    
    df = date_split(df)
    df = crossing_bb(df,1)
    df = profit_calc(df, "close", 1)
    df = profit_calc(df, "close", 4)
    
    df = price_difference(df, "upper_bb20_1", "lower_bb20_1",1, "up_bb20", "low_bb20"  )
    df = price_difference(df, "close", "ma20_1", 1 )
    df = price_difference(df, "close", "ma50_1", 1 )
    df = price_difference(df, "close", "ma100_1", 1 )
    df = price_difference(df, "ma20_1", "ma50_1", 1 )
    df = price_difference(df, "ma50_1", "ma100_1", 1 )
    df = price_difference(df, "open", "close", 1 )
    df['open_close_diff1_lag1'] = df['open_close_diff_1'].shift(1)
    # #find the supports
    # peaks, troughs = find_peaks_and_troughs(df, 'close', prominence=1, width=1)
    # df = tag_peak_trough_values(df, 'close', peaks, troughs, 0.04, 1)
    # df = price_difference(df, "close", "support_indicator_1", 1, "close", "peak" )
    
    df = price_relative_difference(df,"close", "upper_bb20_1", "lower_bb20_1", 1, "up_bb20", "low_bb20" )
    df = trend_measure(df,1)
    df = trend_measure2(df,1)
    
    
    df = df.dropna()
    columns = ['dev20_1', 'dev50_1', 'dev100_1', "lower_bb20_1",  "upper_bb20_1" ]
    df = df.drop(columns, axis=1)
    

    #create 4hr table with indicators
    # df_4hr_expand = df.copy()
    # df_4hr_expand = df_4hr_expand[['day','month', 'year','4hr_tf']]
    
    df_4hr = create_4hr_table(df)
    df_4hr = calc_moving_average(df_4hr,4)
    df_4hr = calc_bb(df_4hr,4)
    df_4hr = price_difference(df_4hr, "upper_bb20_4", "lower_bb20_4", 4, "up_bb20", "low_bb20"  )
    df_4hr = price_difference(df_4hr, "close", "ma20_4",4 )
    df_4hr = price_difference(df_4hr, "close", "ma50_4",4 )
    df_4hr = price_difference(df_4hr, "close", "ma100_4",4 )
    df_4hr = price_difference(df_4hr, "ma20_4", "ma50_4",4 )
    df_4hr = price_difference(df_4hr, "ma50_4", "ma100_4",4 )
    
    # #find the supports
    # peaks, troughs = find_peaks_and_troughs(df_4hr, 'close', prominence=1, width=1)
    # df_4hr = tag_peak_trough_values(df_4hr, 'close', peaks, troughs, 0.04, 4)
    # df_4hr = price_difference(df_4hr, "close", "support_indicator_4", 4, "close", "peak" )
    
    df_4hr = price_relative_difference(df_4hr, "close", "upper_bb20_4", "lower_bb20_4", 4, "up_bb20", "low_bb20" )
    columns = ['high', 'low', 'open', 'close', 'dev20_4', 'dev50_4', 'dev100_4', "lower_bb20_4",  "upper_bb20_4" ]
    df_4hr = df_4hr.drop(columns, axis=1)
    df_4hr = df_4hr.dropna()
    
  
    
    #print(df_4hr)
    # #merged the content from 4hr table into 1 hr.
    merged_df = pd.merge(df, df_4hr, on=['day', 'month', 'year','4hr_tf'], how='left')
    merged_df = merged_df.dropna()

    

    print(merged_df.head(50))
    print(merged_df.columns.to_list())
    print(merged_df['open_close_diff1_lag1'])
    
    return merged_df
 

# def main2(file_path):
#     file_name = "inputs/USDJPY_prices.csv"
#     df = import_data(file_path)
#     df = calc_moving_average(df,1)
#     df = calc_bb(df,1)
    
#     df = date_split(df)
#     df = crossing_bb(df,1)
#     df = profit_calc(df, "close", 1)
#     df = profit_calc(df, "close", 4)

#     peaks, troughs = find_peaks_and_troughs(df, 'close', prominence=1, width=1)
#     df = tag_peak_trough_values(df, 'close', peaks, troughs, 0.04)
#     print(peaks)
#     print(troughs)
#     print(df.head(10))
#     return df

# if __name__ == '__main__':
#     current_directory = os.getcwd()
#     # Go one level up from the current working directory
#     parent_directory = os.path.abspath(os.path.join(current_directory))

#     file_path = rf"{parent_directory}\inputs\USDJPY_prices.csv"
    
#     df = main(file_path)
    
#     df.to_csv(rf"{parent_directory}\outputs\test.csv", index=False)