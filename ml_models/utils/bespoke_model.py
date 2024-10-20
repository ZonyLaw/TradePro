import pandas as pd
from .price_processing import StandardPriceProcessing


class v4Processing(StandardPriceProcessing):
    
    @staticmethod
    def profit_calc(df, col, lag):
        """
        Does not need timeframe parameter!
        This function create the lag price and calculates the profit for the time period specified. 
        However the difference between entry and exit point is swap compared to the standard model calculation.
        This is just another way to look at the profit and loss.

        Args:
            df (dataframe): dataframe contain the data
            col (string): price column to lag (e.g. high, low, open, close)
            lag (integer): time period to lag the column

        Returns:
            dataframe: containing the new columns
        """
        
        df[f'lagged_{col}_{lag}'] = df[col].shift(lag)
        df[f'pl_{col}_{lag}_hr'] =  df[col] - df[f'lagged_{col}_{lag}']
        
        #check if it should be a buy or sell given if entry was made at the point of date and time.
        buy_condition  = (df[f'pl_{col}_{lag}_hr'] > 0)

        
        #0 for sell and 1 for buy
        df[f'trade_{col}_{lag}_hr'] = 0 #'sell'
        df.loc[buy_condition, f'trade_{col}_{lag}_hr'] = 1 #'buy'
        

        return df
    
    def stats_df_gen(self, df, subset_rows):
        """
        This is a function for generating general statistics results based on the scenario prices.
        It pulls all the standard functions together.

        Args:
            df (dataframe): this is expecting prices for the scenario to generate the stats
            subset_rows (integer): specify the number of rows to retain from the dataframe to help minimise computation.

        Returns:
            dataframe: Returns the dataframe with all the statistics based on the prices provided.
    
        """

        df = self.calc_moving_average(df,1)
        df = self.calc_bb(df,1)
        
        df = self.date_split(df)
        df = self.crossing_bb(df,1)
        df = self.profit_calc(df,  "close", 1)
        df = self.profit_calc(df,  "close", 4)
        #note that reverse_point arguement should match that of the profit_calc
        #TODO: This needs a bit of cleaning up by making the arguement being initialised.
        df = self.reverse_point(df, "close", 1)
        
        df = self.calculate_flatness_column(df, 'upper_bb20_1', 'up_bb20_1_flat_5', 5)
        df = self.calculate_flatness_column(df, 'lower_bb20_1', 'low_bb20_1_flat_5', 5)
        df = self.price_difference(df, "upper_bb20_1", "lower_bb20_1",1, "up_bb20", "low_bb20"  )
        df = self.price_difference(df, "close", "ma20_1", 1 )
        df = self.price_difference(df, "close", "ma50_1", 1 )
        df = self.price_difference(df, "close", "ma100_1", 1 )
        df = self.price_difference(df, "ma20_1", "ma50_1", 1 )
        df = self.price_difference(df, "ma50_1", "ma100_1", 1 )
        df = self.price_difference(df, "open", "close", 1 )
        df['open_close_diff1_lag1'] = df['open_close_diff_1'].shift(1)
     

        df = self.trend_measure(df,1)   
        df['pl_close_1_hr'] = df['pl_close_1_hr'].ffill()
        df['pl_close_4_hr'] = df['pl_close_4_hr'].ffill()
        columns = ['dev20_1', 'dev50_1', 'dev100_1' ]
        df = df.drop(columns, axis=1)
        df['reverse_point'] = df['reverse_point'].fillna(0)
        df = df.dropna()
        # df.to_csv(r"C:\Users\sunny\Desktop\Development\before_df-1.csv", index=False)
        # columns = ['dev20_1', 'dev50_1', 'dev100_1', "lower_bb20_1",  "upper_bb20_1" ]
        
        #create 4hr table with indicators
        df_4hr = self.create_4hr_table(df)
        df_4hr = self.calc_moving_average(df_4hr,4)
        df_4hr = self.calc_bb(df_4hr,4)
        df_4hr = self.price_difference(df_4hr, "upper_bb20_4", "lower_bb20_4", 4, "up_bb20", "low_bb20"  )
        df_4hr = self.price_difference(df_4hr, "close", "ma20_4",4 )
        df_4hr = self.price_difference(df_4hr, "close", "ma50_4",4 )
        df_4hr = self.price_difference(df_4hr, "close", "ma100_4",4 )
        df_4hr = self.price_difference(df_4hr, "ma20_4", "ma50_4",4 )
        df_4hr = self.price_difference(df_4hr, "ma50_4", "ma100_4",4 )
        df_4hr = self.price_difference(df_4hr, "high", "upper_bb20_4", 4)
        df_4hr = self.price_difference(df_4hr, "low", "lower_bb20_4", 4) 
        
        # columns = ['high', 'low', 'open', 'close', 'dev20_4', 'dev50_4', 'dev100_4', "lower_bb20_4",  "upper_bb20_4" ]
        columns = ['high', 'low', 'open', 'close', 'dev20_4', 'dev50_4', 'dev100_4'  ]
        df_4hr = df_4hr.drop(columns, axis=1)
        df_4hr = df_4hr.dropna()
        
        
        # #merged the content from 4hr table into 1 hr.
        merged_df = pd.merge(df, df_4hr, on=['day', 'month', 'year','4hr_tf'], how='left')
        # merged_df.to_csv(r"C:\Users\sunny\Desktop\Development\new_4.csv", index=False)
        merged_df = merged_df.dropna()

        #Check the last two rows
        last_row_df = merged_df.tail(2)[0:1]
        # print("Last two rows of dataframe>>>>",last_row_df)
    
        X_live = merged_df.tail(subset_rows)
        
        return X_live
    
   
    
    