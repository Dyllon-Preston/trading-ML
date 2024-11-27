import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import backtrader as bt
import quantstats as qs
from utility import build_indicators, CustomCSVData, MLStrategy

def generate_reports(stocks, models, report_dir='./docs/', performance_dir='./docs/', start_date = '2019-01-01', end_date = '2024-11-23',
                     training_start_date='2005-01-01', training_end_date='2024-11-23', lookahead=7, threshold=0.5, starting_cash = 100_000, commission = 0.001):
    """
    Generate performance reports for multiple stocks using given ML models.

    Args:
        stocks (list): List of stock names.
        models (list): List of trained ML models.
        report_dir (str): Directory to save reports.
        performance_dir (str): Directory to save performance summaries.
        start_date (str): Start date for backtesting.
        end_date (str): End date for backtesting.
        training_start_date (str): Start date for model training.
        training_end_date (str): End date for model training.
        lookahead (int): Lookahead period for indicator generation.
        threshold (float): Threshold for classification.
        starting_cash (float): Starting cash balance for backtesting trading strategy.
        commission (float): Commission percentage for each transaction.

    Returns:
        pd.DataFrame: DataFrame summarizing model performance.
    """
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(performance_dir, exist_ok=True)
    results = []

    for stock in stocks:
        # Load and preprocess stock data
        df = pd.read_csv(f'./snp_stocks/{stock}.csv', parse_dates=['Date'])
        df = df[(df['Date'] >= training_start_date) & (df['Date'] <= training_end_date)]
        df, indicators, flag = build_indicators(df, lookahead=lookahead, threshold=threshold)

        # Prepare features and target
        X, y = df[indicators], df[flag].to_numpy()
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, shuffle=False, random_state=42)

        row = {'Stock': stock}

        for i, model in enumerate(models):
            model_name = type(model).__name__
            model.fit(X_train, y_train) # TRAIN MODEL
            y_pred = model.predict(X_test)

            # Record performance metrics
            row.update({
                f'Model Name {i}': model_name,
                f'Accuracy {i}': accuracy_score(y_test, y_pred),
                f'Precision {i}': precision_score(y_test, y_pred)
            })

            # Backtesting setup
            cerebro = bt.Cerebro()
            cerebro.addanalyzer(bt.analyzers.PyFolio)

            # Define column mapping for CustomCSVData
            column_indices = {
                'datetime': 0,
                'open': 1,
                'high': 2,
                'low': 3,
                'close': 4,
                'volume': 6,
                'openinterest': -1
            }

            # Add stock data using CustomCSVData
            cerebro.adddata(CustomCSVData(
                dataname=f'./snp_stocks/{stock}.csv',
                column_mapping=column_indices,
                dtformat="%Y-%m-%d",
                fromdate=pd.Timestamp(start_date),
                todate=pd.Timestamp(end_date)
            ), name=stock)

            # Add VIX data using CustomCSVData
            cerebro.adddata(CustomCSVData(
                dataname='./stocks/^VIX.csv',
                column_mapping=column_indices,
                dtformat="%Y-%m-%d",
                fromdate=pd.Timestamp(start_date),
                todate=pd.Timestamp(end_date)
            ), name='VIX')

            cerebro.addstrategy(MLStrategy, model=model, scaler=scalar, feature_names=indicators)
            cerebro.broker.setcash(starting_cash)
            cerebro.broker.setcommission(commission=commission)

            strat = cerebro.run()[0] # RUN BACKTRADER
            returns = strat.analyzers.getbyname('pyfolio').get_pf_items()[0]
            returns.index = returns.index.tz_convert(None)

            # Generate QuantStats report
            qs.reports.html(
                returns, df.set_index('Date')['Adj Close'].pct_change(),
                output=f'{report_dir}{stock}_{model_name}.html',
                title=f'{stock} vs {model_name} Strategy'
            )
            print(f'Report saved for stock {stock} with model {model_name} to: {report_dir}{stock}_{model_name}.html')
        
        results.append(row)

    performance_df = pd.DataFrame(results)
    performance_df.to_csv(f'{performance_dir}performance.csv', index=False)
    print(f'Performances saved to: {performance_dir}performance.csv')
    return performance_df