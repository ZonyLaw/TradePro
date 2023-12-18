from data_processing import main

import os
current_directory = os.getcwd()
# Go one level up from the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory))

print(rf"{parent_directory}\inputs\USDJPY_prices.csv")
file_path = rf"{parent_directory}\inputs\USDJPY_prices.csv"

print(file_path)

df = main(file_path)

# print(df_4hr.head(10))

# print(df.columns.tolist())



# selected_columns = [ 'trade_close_1_hr', 'lagged_close_4', 'pl_close_4_hr', 'trade_close_4_hr', 'up_bb20_low_bb20_diff_1', 'close_ma20_1_diff_1', 'close_ma50_1_diff_1', 'close_ma100_1_diff_1', 'ma20_1_ma50_1_diff_1', 'ma50_1_ma100_1_diff_1', 'up_bb20_low_bb20_rel_diff_1', 'trend_strength_1']
selected_columns = [ 'date','trade_close_1_hr','close_ma100_4_diff_4', 'ma20_4_ma50_4_diff_4', 'ma50_4_ma100_4_diff_4', 'up_bb20_low_bb20_rel_diff_4']
temp = df[selected_columns]
print(temp.tail(50))
# print(df.tail(20))

# filtered_df = df[df['bb_status'] == 'inside_bb']

# Display the filtered DataFrame
print("hello")
#print("filtered data", filtered_df)

#print("reverse", df['market_bb'])