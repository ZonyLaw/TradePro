import pandas as pd
import numpy as np

from prices.models import Price
from tickers.models import Ticker
from datetime import timedelta
from scipy.signal import find_peaks

class StandardPriceProcessing():
    """
    The purpose of this is calculate the different attributes from the open, high, low, and close prices.
    The class attribute accessiable is ticker.
        
    """
    def __init__(self, ticker):
        self.ticker = ticker
    
    @staticmethod
    def calc_moving_average(df, timeframe):
        """
        Calculates the moving averages and standard deviation.

        Args:
            df (dataframe): price dataframe containing the close price
            timeframe (integer): specify the timeframe to calculate the moving average and standard deviation

        Returns:
            dataframe: a new dataframe with the moving average and standard deviation
        """
        
        #calculates the moving averages
        df[f'ma20_{timeframe}'] = df['close'].rolling(window=20).mean()
        df[f'ma50_{timeframe}'] = df['close'].rolling(window=50).mean()
        df[f'ma100_{timeframe}'] = df['close'].rolling(window=100).mean()
        
        #calculates the standard deviation
        df[f'dev20_{timeframe}'] = df['close'].rolling(window=20).std()
        df[f'dev50_{timeframe}'] = df['close'].rolling(window=50).std()
        df[f'dev100_{timeframe}'] = df['close'].rolling(window=100).std()

        return df

    @staticmethod
    def calc_bb(df, timeframe):
        """
        Calculates the bollinger band based on ma20 which commonly used.

        Args:
            df (dataframe): price dataframe containing the close price
            timeframe (integer): specify the timeframe to calculate the moving average and standard deviation

        Returns:
            dataframe: a new dataframe with the bollinger band based on ma20
        """
        
        df[f'upper_bb20_{timeframe}'] = df[f'ma20_{timeframe}']+2*df[f'dev20_{timeframe}']
        df[f'lower_bb20_{timeframe}'] = df[f'ma20_{timeframe}']-2*df[f'dev20_{timeframe}']
        
        return df

    @staticmethod
    def crossing_bb(df, timeframe):
        """
        This function determines four cases that prices could have compared to the bollinger band location. 
         1) crossover one extreme point of the bollinger band, 
         2) hit one extreme point of the bollinger band,
         3) near one extreme point of the bollinger band, and 
         4) other cases within the bollinger band.
        Note: the 20_2 denotes it is based on 20 days average with a scaler of 2 to create this confidence level.

        Args:
            df (dataframe): DataFrame containing 'high', 'low', 'close', 'upper_bb20_2', 'lower_bb20_2'
            timeframe (integer): specify the timeframe to calculate the moving average and standard deviation
            
        Returns:
            dataframe: a new dataframe with bb_status classfying the different cases.
        """

        df[f'bb_status_{timeframe}'] = 'inside_bb'  # Default value

        #check for crossing the BB
        conditions_upper_hit = ((df["high"] > df[f'upper_bb20_{timeframe}']) & (df["low"] < df[f'upper_bb20_{timeframe}'])) 
        conditions_lower_hit = (df["high"] > df[f'lower_bb20_{timeframe}']) & (df["low"] < df[f'lower_bb20_{timeframe}']) 
        
        conditions_upper_near = (abs(df[f'upper_bb20_{timeframe}'] - df["high"]) < 0.04)
        conditions_lower_near = (abs(df["low"] - df[f'lower_bb20_{timeframe}']) < 0.04)
        
        #hit the bollinger band
        df.loc[conditions_upper_hit, f'bb_status_{timeframe}'] = 'upper_hit'
        df.loc[conditions_lower_hit, f'bb_status_{timeframe}'] = 'lower_hit'
        
        #near the bollinger band
        df.loc[conditions_upper_near, f'bb_status_{timeframe}'] = 'upper_near'
        df.loc[conditions_lower_near, f'bb_status_{timeframe}'] = 'lower_near'

        #crossover the bollinger band
        conditions_upper_region = df["close"] > df[f'upper_bb20_{timeframe}']
        conditions_lower_region = df["close"] < df[f'lower_bb20_{timeframe}']

        df.loc[conditions_upper_region, f'bb_status_{timeframe}'] = 'upper_crossover'
        df.loc[conditions_lower_region, f'bb_status_{timeframe}'] = 'lower_crossover'

        return df

    @staticmethod
    def profit_calc(df, col1, col2, offset_hr):
        """

        This function creates the lead/lag price and calculates the profit for the offset_hr specified.
        when offset_hr is negative you are leading and when positive you are lagging.
        NOTE: please read comments in the if else statment

        Args:
            df (dataframe): dataframe contain the data
            col1 (string): price column to shift (e.g. high, low, open, close)
            col2 (string): price column to shift (e.g. high, low, open, close)
            offset_hr (integer): time period to lag the column as negative number and lead with positive number

        Returns:
            dataframe: containing the new columns for profit/loss for the price specified.
            TODO could rename as trade in the future.
        """
        timeframe = abs(offset_hr)
        
        if offset_hr < 0:
            
            # this bring future price which is exit to present price which is the entry; So we are trying to predict potential profit and loss. 
            # the offset is negative to bring future to current price.
            df[f'lead_{col2}_{timeframe}'] = df[col2].shift(offset_hr)
            df[f'pl_{col2}_f{timeframe}_hr'] =  df[f'lead_{col2}_{timeframe}'] - df[col1]
            
            buy_condition  = (df[f'pl_{col1}_f{timeframe}_hr'] > 0)

            #0 for sell and 1 for buy
            df[f'trade_{col1}_{timeframe}_hr'] = 0 #'sell'
            df.loc[buy_condition, f'trade_{col1}_{timeframe}_hr'] = 1 #'buy'
            
            
        else:
            # this brings the past price which become the entry and the present price is the exit price. 
            # the offset is positive to bring price forward.
            # normally, we use previous price to explain current price but this case is probably not that useful as the model will be
            # explaining previous timeframe to achieve current profit or loss. This is not useful as it already happened.
            df[f'lag_{col2}_{timeframe}'] = df[col2].shift(offset_hr)
            df[f'pl_{col2}_{timeframe}_hr'] = df[col1] - df[f'lag_{col2}_{timeframe}'] 
            
            #check if it should be a buy or sell given if entry was made at the point of date and time.
            buy_condition  = (df[f'pl_{col1}_{timeframe}_hr'] > 0)

            #0 for sell and 1 for buy
            df[f'trade_{col1}_{timeframe}_hr'] = 0 #'sell'
            df.loc[buy_condition, f'trade_{col1}_{timeframe}_hr'] = 1 #'buy'
        

        return df
    
    
    @staticmethod
    def reverse_point(df, col2, offset_hr):
        timeframe = abs(offset_hr)
        if offset_hr < 0:
            pl_header = f'pl_{col2}_f{timeframe}_hr'
        else:
            pl_header = f'pl_{col2}_{timeframe}_hr'
        
             
        df['reverse_point'] = 0

        # Corrected conditional indexing
        condition = (df[pl_header].abs() > 0.02) & (df[pl_header].abs().shift(1) > 0.02)

        # Check for sign change and update reverse_point where condition is True
        sign_change_condition = df[pl_header] * df[pl_header].shift(1) < 0
        df.loc[condition & sign_change_condition, 'reverse_point'] = 1

        # Shift 'reverse_point' column by one position backward
        df['reverse_point'] = df['reverse_point'].shift(-1)
        

        return df


    @staticmethod
    def profit_calc2(df, col1, col2, offset_hr):
        """
        This function creates the lead/lag price and calculates the profit for the offset_hr specified.
        when offset_hr is negative you are lagging the x data (independent variables) to the y data (dependent variables).

        Args:
            df (dataframe): dataframe contain the data
            col1 (string): price column to shift (e.g. high, low, open, close)
            col2 (string): price column to shift (e.g. high, low, open, close)
            offset_hr (integer): time period to lag the column as negative number and lead with positive number

        Returns:
            dataframe: containing the new columns for profit/loss for the price specified.
            TODO could rename as trade in the future.
        """
        timeframe = abs(offset_hr)
        
        if offset_hr < 0:
            
            # this bring future price which is exit to present price which is the entry; So we are trying to predict potential profit and loss. 
            # the offset is negative to bring future to current price.
            df[f'lag_{col2}_{timeframe}'] = df[col2].shift(offset_hr)
            df[f'pl_{col2}_f{timeframe}_hr'] =  df[f'lag_{col2}_{timeframe}'] - df[col1]
            
            buy_condition  = (df[f'pl_{col2}_{timeframe}_hr'] > 0)

            #0 for sell and 1 for buy
            df[f'trade_{col1}_{timeframe}_hr'] = 0 #'sell'
            df.loc[buy_condition, f'trade_{col1}_{timeframe}_hr'] = 1 #'buy'
            
        else:

            df[f'lag_{col2}_{timeframe}'] = df[col2].shift(offset_hr)
            df[f'pl_f{col2}_{timeframe}_hr'] = df[col1] - df[f'lead_f{col2}_{timeframe}'] 
            
            #check if it should be a buy or sell given if entry was made at the point of date and time.
            buy_condition  = (df[f'pl_f{col2}_{timeframe}_hr'] > 0)

            #0 for sell and 1 for buy
            df[f'trade_{col1}_{timeframe}_hr'] = 0 #'sell'
            df.loc[buy_condition, f'trade_{col1}_{timeframe}_hr'] = 1 #'buy'
        

        return df

    @staticmethod
    def date_split(df):
        """
        This function split up the date into month, day, year, & time.

        Args:
            df (dataframe): raw dataframe with the date column

        Returns:
            dataframe: return new dataframe with the date split up.
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

    @staticmethod
    def price_difference(df, price1, price2, timeframe, label1=None, label2=None):
        """
        This takes the difference between two prices and also allows the user to specify different label of the variables.
        Otherwise it will use the price name to create the new column in the dataframe.

        Args:
            df (dataframe): This is the dataframe containing the data.
            price1 (string): This is the column name of the dataframe for the first price.
            price2 (string): This is the column name of the dataframe for the second price.
            label1 (string): Label for the first price (default is None).
            label2 (string): Label for the second price (default is None).

        Returns:
            dataframe: return new dataframe with price difference calculations.
        """

        if label1 is None:
            label1 = price1
            
        if label2 is None:
            label2 = price2

        df[f"{label1}_{label2}_diff_{timeframe}"] = df[price1] - df[price2]

        return df

    @staticmethod
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
            dataframe: return new dataframe.
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

    @staticmethod
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
            dataframe: return new dataframe.
        """
        
        df[f"{label1}_{label2}_rel_diff_{timeframe}"] = 0
        condition = (df[close] < df[upper_bound]) & (df[close] > df[lower_bound])
        df.loc[condition, f"{label1}_{label2}_rel_diff_{timeframe}"] = abs((df[close]-df[lower_bound]) / (df[upper_bound] - df[lower_bound]))
        
        return df

    @staticmethod
    def trend_measure(df, timeframe):
        """
        The function creates a indicator to measure the strenght of the trend direction.

        Args:
            df (dataframe): dataframe containing the standard prices - open, close
            timeframe (integer): specify the timeframe to derive the strength of the trend

        Returns:
            dataframe: return new dataframe.
        """
        
        #check if it should be a buy or sell given if entry was made at the point of date and time.
        df['pl_close'] = df['close'] - df['open']
        buy_condition  = (df['pl_close'] > 0)

        #a new column to save the trade direction for the timeframe specified.
        df[f'trade_close_{timeframe}_hr'] = 0 #'sell'
        df.loc[buy_condition, f'trade_close_{timeframe}_hr'] = 1 #'buy'
        
        #setup a condition to check that current trade is the same as previous
        condition = (df[f'trade_close_{timeframe}_hr'] == df[f'trade_close_{timeframe}_hr'].shift(1))
        #initialise the new column saving the trend strength indicator
        df[f'trend_strength_{timeframe}'] = 0

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
                #this is when there is a change in the trade direction and the cumulative_sum will be reset to 0
                cumulative_sum = 0

            df.iloc[i, df.columns.get_loc(f'trend_strength_{timeframe}')] = cumulative_sum

        return df

    @staticmethod
    def create_4hr_table(df_1hr):
        
        """
        The function creates four price data set using the 1hr price data. It makes a copy of the dataframe
        so it doesn't affect the original dataframe.

        Args:
            df_1hr (dataframe): dataframe contains 1 hour prices.

        Returns:
            dataframe: return new a dataframe containing only the 4hr prices.
        """    
        
        # Filter rows based on the pattern
        df_4hr = df_1hr.copy()
        df_4hr = df_1hr[df_1hr['hr'] % 4 == 0].copy()
        df_4hr = df_4hr[['day','month', 'year','4hr_tf','open', 'high', 'low', 'close']]

        return df_4hr

    @staticmethod
    def find_peaks_and_troughs(df, column_name, prominence=1, width=1):
        """
        This function determines the peaks and troughs.

        Args:
            df (dataframe): dataframe containing the standard prices.
            column_name (string): defines the new column name in the dataframe.
            prominence (int, optional): Define the parameters of the find_peaks(). Defaults to 1.
            width (int, optional): Define the parameters of the find_peaks(). Defaults to 1.

        Returns:
            _type_: _description_
        """
        
        
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

    @staticmethod
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

    @staticmethod
    def priceDB_to_df(ticker):
        """
        This extract prices and turn it into a dataframe with recent price at the bottom.
        """
        
        queryset = Price.objects.filter(ticker=ticker)

        # Convert queryset to a list of dictionaries
        data = list(queryset.values())

        # Create a Pandas DataFrame
        df = pd.DataFrame(data)
        df.sort_values(by='date', inplace=True)
    
        return df

    @staticmethod
    def scenario_builder(df, close_adjustment, scenario):
        """
        Build a scenario of price data for the next hour. This can be fed into a new
        model run to see what results are generated. There are two scenarios:
            1) continue
            2) reverse
        
        The assumption is that the current candle will close with the close_adjustment applied. 
        This sets the stage for the next candlestick to be created with the same close_adjustment. 
        As a result, the current prediction differs from the actual historical close. 
        The historical close represents the actual closing price

        Args:
            df (pd.DataFrame): DataFrame containing the original prices.
            close_adjustment (float): Specify the pips to create the scenario. 
            scenario (str): Either 'continue' or 'reverse' scenario.

        Returns:
            pd.DataFrame: DataFrame with two extra rows of forecast prices for the next hour.
        """
        
        # Adjust close_adjustment for 'continue' or 'reverse' scenario
        if scenario == 'continue':
            close_adjustment = -close_adjustment
        
        # Get the last row of the DataFrame and set direction, then generate another 
        # row continuing the direction. 
        last_row = df.iloc[-1].copy()
        print(len(df.index))
        last_row['date'] += timedelta(hours=1)
        if(last_row['close'] > last_row['open']):
            last_row['open'] = last_row['close']
            last_row['close'] = last_row['open'] - close_adjustment
            df.loc[len(df.index)] = last_row
       
            
        else:
            last_row['open'] = last_row['close']
            last_row['close'] = last_row['open'] + close_adjustment
            df.loc[len(df.index)] = last_row
     
        
        # Optionally save the new DataFrame to CSV
        # df.to_csv(r"C:\Users\sunny\Downloads\df_scenario.csv", index=False)

        return df


    @staticmethod
    def calculate_flatness_column(df, ori_col, new_col, window_size=10):
        """
        Calculate the flatness for each position in the specified column of the DataFrame where there are 
        at least `window_size` preceding values. Flatness is measured by the sum of absolute differences 
        between consecutive numbers in the last `window_size` values.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            ori_col (str): The name of the original column to calculate flatness for.
            new_col (str): The name of the new column to store flatness values.
            window_size (int): The number of preceding values to consider for calculating flatness.

        Returns:
            pd.DataFrame: The original DataFrame with a new column for flatness values.

        Raises:
            ValueError: If the DataFrame has fewer elements than `window_size`.
        """
        # Ensure the DataFrame has at least `window_size` numbers in the specified column
        if len(df) < window_size:
            raise ValueError(f"The DataFrame must contain at least {window_size} rows.")
        
        # Create a new column for flatness values
        df[new_col] = None
        
        # Iterate over the DataFrame starting from the `window_size`-th row
        for i in range(window_size - 1, len(df)):
            # Get the last `window_size` values from the specified column
            last_n_numbers = df[ori_col].iloc[i - window_size + 1:i + 1].values
            
            # Calculate the flatness as the sum of absolute differences between consecutive numbers
            flatness = sum(abs(last_n_numbers[j + 1] - last_n_numbers[j]) for j in range(window_size - 1))
            
            # Store the flatness value in the new column
            df.at[df.index[i], new_col] = flatness
        
        return df

        
    def stats_df_gen(self, df, subset_rows):
        """
        This is a function for generating general statistics results based on the scenario prices.
        It pulls all the standard functions together. NOTE any update here need to be brought over to the 
        bespoke_model.py!

        Args:
            df (dataframe): this is expecting prices for the scenario to generate the stats
            subset_rows (integer): specify the number of rows to retain from the dataframe to help minimise computation.

        Returns:
            dataframe: Returns the dataframe with all the statistics based on the prices provided.
    
        """
        df = self.calc_moving_average(df, 1)
        df = self.calc_bb(df, 1)
        
        df = self.date_split(df)
        df = self.crossing_bb(df, 1)
        df = self.profit_calc(df, "open", "open", -1)
        df = self.profit_calc(df, "open", "open", -4)
        df = self.profit_calc(df, "open", "open", 1)
        df = self.profit_calc(df, "open", "open", 4)
        df = self.reverse_point(df, "open", -1)
        
        df = self.calculate_flatness_column(df, 'upper_bb20_1', 'up_bb20_1_flat_5', 5)
        df = self.price_difference(df, "upper_bb20_1", "lower_bb20_1", 1, "up_bb20", "low_bb20"  )
        df = self.price_difference(df, "close", "ma20_1", 1 )
        df = self.price_difference(df, "close", "ma50_1", 1 )
        df = self.price_difference(df, "close", "ma100_1", 1 )
        df = self.price_difference(df, "ma20_1", "ma50_1", 1 )
        df = self.price_difference(df, "ma50_1", "ma100_1", 1 )
        df = self.price_difference(df, "open", "close", 1 )
        df['open_close_diff1_lag1'] = df['open_close_diff_1'].shift(1)
        df = self.price_difference(df, "high", "upper_bb20_1", 1 )
        df = self.price_difference(df, "low", "lower_bb20_1", 1 )

        df = self.trend_measure(df, 1)   
        df['pl_open_f1_hr'] = df['pl_open_f1_hr'].ffill()
        df['pl_open_f4_hr'] = df['pl_open_f4_hr'].ffill()
        df['pl_open_1_hr'] = df['pl_open_1_hr'].ffill()
        df['pl_open_4_hr'] = df['pl_open_4_hr'].ffill()
        columns = ['dev20_1', 'dev50_1', 'dev100_1', 'lead_open_1', 'lead_open_4']
        
        df = df.drop(columns, axis=1)
        #fill empty cell with zero, may need to review as this may not be appropriate.
        df['reverse_point'] = df['reverse_point'].fillna(0)
        df = df.dropna()
   
        # create 4hr table with indicators
        df_4hr = self.create_4hr_table(df)
        df_4hr = self.calc_moving_average(df_4hr, 4)
        df_4hr = self.calc_bb(df_4hr, 4)
        df_4hr = self.price_difference(df_4hr, "upper_bb20_4", "lower_bb20_4", 4, "up_bb20", "low_bb20")
        df_4hr = self.price_difference(df_4hr, "close", "ma20_4", 4)
        df_4hr = self.price_difference(df_4hr, "close", "ma50_4", 4)
        df_4hr = self.price_difference(df_4hr, "close", "ma100_4", 4)
        df_4hr = self.price_difference(df_4hr, "ma20_4", "ma50_4", 4)
        df_4hr = self.price_difference(df_4hr, "ma50_4", "ma100_4", 4)
        df_4hr = self.price_difference(df_4hr, "high", "upper_bb20_4", 4)
        df_4hr = self.price_difference(df_4hr, "low", "lower_bb20_4", 4)        
        
        columns = ['high', 'low', 'open', 'close', 'dev20_4', 'dev50_4', 'dev100_4']
        df_4hr = df_4hr.drop(columns, axis=1)
        df_4hr = df_4hr.dropna()

       
       
        # merged the content from 4hr table into 1 hr.
        merged_df = pd.merge(df, df_4hr, on=['day', 'month', 'year','4hr_tf'], how='left')
        merged_df = merged_df.dropna()
 

        # Check the last two rows
        last_row_df = merged_df.tail(2)[0:1]
        # print("Last two rows of dataframe>>>>",last_row_df)
    
        X_live = merged_df.tail(subset_rows)
               
        return X_live
      
          
    def scenario_reverse(self):
        """
        This function is to generate a reverse scneario based on reversal of candle sticks.

        Returns:
            dataframe: contains the new prices for the next row creating the scenario 
            for the next hour.
        """
        
        ticker = Ticker.objects.get(symbol="USDJPY")
        df = self.priceDB_to_df(ticker)
        
        #build the reverse candle stick scenario
        #the base scenario is retained for the first dataframe
        reverse_df = self.scenario_builder(df.copy(), 0.1, "reverse")
        reverse_df_2pips = self.stats_df_gen(reverse_df, 2)
        
        #this is looking at 20 pips movement.
        reverse_df = self.scenario_builder(df.copy(), 0.2, "reverse")
        reverse_df_4pips = self.stats_df_gen(reverse_df, 2)
        
        #collect all the two sensitivity scenarios to feed into the model
        last_row = reverse_df_4pips.tail(1)
        combined_df = pd.concat([reverse_df_2pips, last_row])

        return (combined_df)

    
    def scenario_continue(self):
        """
        This function is to generate a continue scneario based on continuing trend.

        Returns:
            dataframe: contains the new prices for the next row creating the scenario 
            for the next hour.
        """
        
        ticker = Ticker.objects.get(symbol="USDJPY")
        df = self.priceDB_to_df(ticker)
        #build the reverse candle stick scenario
        #the base scenario is retained for the first dataframe
        continue_df = self.scenario_builder(df.copy(), 0.1, "continue")
        continue_df_2pips = self.stats_df_gen(continue_df, 2)
        
        #this is looking at 20 pips movement.
        continue_df = self.scenario_builder(df.copy(), 0.2, "continue")
        continue_df_4pips = self.stats_df_gen(continue_df, 2)
        
        #collect all the two sensitivity scenarios to feed into the model
        last_row = continue_df_4pips.tail(1)
        combined_df = pd.concat([continue_df_2pips, last_row])
        
        return (combined_df)
    

    def historical_record(self, num_rows):
        """
        This function is to generate a historical table.
        TODO the ticker assignment needs updating for dynamic usage.
        
        Args:
            num_rows (integer): number of rows to retrieve historical data

        Returns:
            dataframe: contains the prices for the specific rows inputted
        """
        
        ticker = Ticker.objects.get(symbol="USDJPY")
        df = self.priceDB_to_df(ticker)
        historical_df = self.stats_df_gen(df, num_rows)
        # historical_df.to_csv(r"C:\Users\sunny\Desktop\Development\history4_df-1.csv", index=False)
    
        return (historical_df)

    def prediction_variability(self, adjustment):
        """
        This function is to test the sensitivity of the model predictions by flipping direction of the candle stick.
        A consistent predictions means a stronger confidence.
        The last two rows are increase and decrease by the adjustment. This is because the last row is dropped in the process 
        due to NA content. Therefore calculation takes the 2nd last row to produce model results.
        
        Args:
            adjustment (float): adjustment is the amount to flip the candle movement to test the sensitivity of the model results

        Returns:
            dataframe: contains the results of prediction after a small adjustment to the current price.
        """
        
        ticker = Ticker.objects.get(symbol="USDJPY")
        df = self.priceDB_to_df(ticker)

        # Make adjustments directly to the last two rows in the DataFrame.
        # it seems the stats_df_gen() function ignores the last row because of NA so the 2nd last row is used.
        # TODO: review the code so it handle last row better and should this be called from view?
        variability_df = df.copy()
        variability_df['scenario'] = 0.0 
        last_row1 = variability_df.iloc[-1]
        last_row2 = variability_df.iloc[-2]

        
        #Prediction based on positive candle stick
        # print("looking at var>>>>>>>>", variability_df)
        variability_df.loc[variability_df.index[-1], 'scenario'] = adjustment
        variability_df.loc[variability_df.index[-1], 'close'] = last_row1['open'] + adjustment
        variability_df.loc[variability_df.index[-2], 'close'] = last_row2['open'] + adjustment
        variability_df_pos = self.stats_df_gen(variability_df,2)
      
        # print("last row", variability_df.loc[last_row.id])
        # print("AFTER POS>>>>>>>>", variability_df_pos[['day','month','year', 'open','close']])
    
        #Takes the last trend strength to minismise the overly switching
        # variability_df_pos.loc[variability_df_pos.index[1], 'trend_strength_1'] = variability_df_pos.loc[variability_df_pos.index[0], 'trend_strength_1']
        
        #Prediction based on positive candle stick
        # print("looking at var>>>>>>>>", variability_df)
        variability_df.loc[variability_df.index[-1], 'scenario'] = -adjustment
        variability_df.loc[variability_df.index[-1], 'close'] = last_row1['open'] - adjustment
        variability_df.loc[variability_df.index[-2], 'close'] = last_row2['open'] - adjustment
        variability_df_neg = self.stats_df_gen(variability_df,2)
        # print("last row", variability_df.loc[last_row.id])   
        #print("AFTER NEG>>>>>>>>", variability_df_neg['close'])
        #Takes the last trend strength to minismise the overly switching
        # variability_df_neg.loc[variability_df_neg.index[1], 'trend_strength_1'] = variability_df_neg.loc[variability_df_neg.index[0], 'trend_strength_1']
        
    
        variability_all = pd.concat([variability_df_pos.tail(1), variability_df_neg.tail(1)])
        # variability_all.to_csv(r"C:\Users\sunny\Downloads\variability_all.csv")
        # print("variability all >>>>>>" ,variability_all['scenario']) 
        
        return (variability_all)