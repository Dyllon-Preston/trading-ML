import pandas as pd
import backtrader as bt
from utility import CustomCSVData


def data_import_test():
    try:
        # Indices of columns in the CSV
        column_indices = {
            'datetime': 0,  # Ensure 'datetime' field is correctly named for BackTrader
            'open': 3,
            'high': 1,
            'low': 2,
            'close': 4,
            'volume': 5,
            'openinterest': -1
        }

        # Create the CustomCSVData feed
        data1 = CustomCSVData(
            dataname="./stocks/aapl.csv", # AAPL TEST
            column_mapping=column_indices,
            dtformat="%Y-%m-%d"  # Adjust to your date format
        )

        ercotda = pd.read_csv('./stocks/ERCOTDA_price.csv')

        # Convert 'Hour_of_Day' to string, ensuring two-digit formatting for hours
        ercotda["Hour_of_Day"] = (ercotda["Hour_of_Day"] - 1).astype(str).str.zfill(2) + ":00:00"

        # Combine 'Date' and 'Hour_of_Day' into a single datetime column
        ercotda["datetime"] = pd.to_datetime(ercotda["Date"] + " " + ercotda["Hour_of_Day"])

        # Sort by the new datetime column
        ercotda = ercotda.sort_values(by="datetime").reset_index(drop=True)

        # Need to create processed CSV file since we sorted based on datetime
        ercotda.to_csv('./stocks/processed_ercotda.csv', index = False)

        # Indices of columns in the CSV
        column_indices = {
            'datetime': 0,
            'time': 1,
            'open': -1,
            'high': -1,
            'low': -1,
            'close': 2,
            'volume': -1,
            'openinterest': -1
        }

        # Create the CustomCSVData feed
        data2 = CustomCSVData(
            dataname="./stocks/processed_ercotda.csv", # ERCOTDA TEST
            column_mapping=column_indices,
            dtformat ="%m/%d/%y",
            tmformat = "%H:%M:%S"
        )

        # Indices of columns in the CSV
        column_indices = {
            'datetime': 0,
            'open': -1,
            'high': 3,
            'low': 4,
            'close': 2,
            'volume': 5,
            'openinterest': -1
        }

        # Create the CustomCSVData feed
        data3 = CustomCSVData(
            dataname="./stocks/002054.XSHE.csv", # XSHE TEST
            column_mapping=column_indices,
            dtformat ="%Y-%m-%d",
        )

        # Initialize BackTrader
        cerebro = bt.Cerebro()
        cerebro.adddata(data1)
        cerebro.adddata(data2)
        cerebro.adddata(data3)
        cerebro.run()
        print('CustomCSVData test complete ...')
    except Exception as e:
        print(f'CustomCSVData test failed with exception {e}.')



def run_tests():
    print('Running tests...')
    data_import_test()
    print('All tests successfully passed.')