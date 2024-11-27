import os
import yfinance as yf
import pandas as pd
import backtrader as bt

def download_ticker_data(tickers, data_dir, start_date="2010-01-01", end_date = "2024-11-23", period = None, force_redownload = False):
    """
    Download historical stock data for a list of tickers and save them as CSV files.

    Parameters:
        tickers (list): List of ticker symbols.
        data_dir (str): Directory to save the downloaded CSV files.
        start_date (str): Start date for historical data.

    Returns:
        None
    """
    
    if isinstance(tickers, str):
        tickers = [tickers]
        
    os.makedirs(data_dir, exist_ok=True)

    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}.csv")

        if os.path.exists(file_path) and not force_redownload:
            print(f"Data for {ticker} already exists. Skipping.")
            continue
            
        try:
            print(f"Downloading data for {ticker}...")

            if period:
                data = yf.download(ticker, period=period)
            else:
                data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                data.to_csv(file_path)
                print(f"Saved data for {ticker} to {file_path}.")
            else:
                print(f"No data found for {ticker}.")
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

class CustomCSVData(bt.feeds.GenericCSVData):
    """
    Custom CSV Data Feed to dynamically map columns in a CSV to BackTrader fields.
    """

    def __init__(self, column_mapping, **kwargs):
        """
        Initialize CustomCSVData.
        
        :param filename: Path to the CSV file.
        :param column_mapping: A dictionary mapping BackTrader fields to column indices.
        :param date_format: Optional date format (e.g., '%Y-%m-%d').
        :param kwargs: Additional parameters.
        """
        # Ensure that filename is explicitly passed to super()

        # Map column indices to BackTrader parameters
        for field, col_idx in column_mapping.items():
            if hasattr(self.params, field):
                setattr(self.params, field, col_idx)
            else:
                raise ValueError(f"Unknown field '{field}' in column_mapping.")

        # Pass dataname and other parameters directly to the parent class
        super().__init__(**kwargs)


def build_indicators(df, lookahead = 7, threshold = 0.5):
    """
    Enhance the given DataFrame with various technical indicators commonly used in stock market analysis.
    Also include a flag for if after a period determined by the lookahead parameter the stock raised by the threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['Adj Close', 'Volume'].

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Modified DataFrame with new indicator columns added.
            - list: A list of column names for the added indicators.
            - str: The column name of the flag.
    """
    # Moving averages and standard deviations
    for window in [20, 100, 150]:
        df[f'{window}D_MA'] = df['Adj Close'].rolling(window=window).mean()
        df[f'{window}D_STD'] = df['Adj Close'].rolling(window=window).std()

    # Bollinger Bands
    df['Upper_Band'] = df['20D_MA'] + 2 * df['20D_STD']
    df['Lower_Band'] = df['20D_MA'] - 2 * df['20D_STD']

    # RSI (Relative Strength Index)
    delta = df['Adj Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volume-based RSI
    delta_volume = df['Volume'].diff()
    volume_gain, volume_loss = delta_volume.clip(lower=0), -delta_volume.clip(upper=0)
    avg_volume_gain = volume_gain.rolling(window=14, min_periods=1).mean()
    avg_volume_loss = volume_loss.rolling(window=14, min_periods=1).mean()
    volume_rs = avg_volume_gain / avg_volume_loss
    df['Volume_RSI'] = 100 - (100 / (1 + volume_rs))

    # MACD (Moving Average Convergence Divergence)
    df['Fast_EMA'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['Slow_EMA'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['Fast_EMA'] - df['Slow_EMA']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    # VIX
    vix = pd.read_csv('./stocks/^VIX.csv', parse_dates=['Date'])
    vix = vix.ffill()
    vix.rename(columns={'Adj Close': 'VIX'}, inplace=True) # Rename the VIX column for clarity
    df = pd.merge(df, vix[['Date', 'VIX']], on='Date', how='left')  # Merge VIX data with the main DataFrame on 'Date'

    # Lookahead percentage change and flag
    df['PC'] = (df['Adj Close'].shift(-lookahead) - df['Adj Close']) / df['Adj Close'] * 100
    df['Flag'] = (df['PC'] > threshold).astype(int)

    df = df[149:-lookahead]

    indicator_columns = ['Adj Close'] + [
        f'{window}D_MA' for window in [20, 100, 150]
    ] + [
        f'{window}D_STD' for window in [20, 100, 150]
    ] + ['Upper_Band', 'Lower_Band', 'RSI', 'Volume_RSI', 'Fast_EMA', 'Slow_EMA', 'MACD_Line', 'Signal_Line', 'VIX']

    return df, indicator_columns, 'Flag'


class MLStrategy(bt.Strategy):
    """
    A Backtrader strategy that uses a machine learning model to make trading decisions
    based on a set of technical indicators.
    """

    def __init__(self, model, scaler, feature_names, verbose = False):
        # Define indicators used in the strategy
        self.ma_20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.ma_100 = bt.indicators.SimpleMovingAverage(self.data.close, period=100)
        self.ma_150 = bt.indicators.SimpleMovingAverage(self.data.close, period=150)

        self.std_20 = bt.indicators.StandardDeviation(self.data.close, period=20)
        self.std_100 = bt.indicators.StandardDeviation(self.data.close, period=100)
        self.std_150 = bt.indicators.StandardDeviation(self.data.close, period=150)

        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=20)
        self.upper_band = self.bollinger.lines.top
        self.lower_band = self.bollinger.lines.bot

        # Relative Strength Index (RSI)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

        # Volume RSI
        self.volume_rsi = bt.indicators.RSI(self.data.volume, period=14)

        # Exponential Moving Averages (EMAs) and MACD
        self.fast_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=12)
        self.slow_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=26)
        self.macd_line = self.fast_ema - self.slow_ema
        self.signal_line = bt.indicators.ExponentialMovingAverage(self.macd_line, period=9)

        # VIX (Volatility Index)
        self.vix = self.datas[1].close  # Assuming VIX data is loaded as a second data feed

        # Machine learning model
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

        # Logging
        self.verbose = verbose


    def next(self):
        """
        Defines the strategy logic for each time step. Uses the ML model to decide
        whether to go all in or close the position.
        """
        # Prepare indicator values for the model
        features = [
            self.data.close[0],
            self.ma_20[0],
            self.ma_100[0],
            self.ma_150[0],
            self.std_20[0],
            self.std_100[0],
            self.std_150[0],
            self.upper_band[0],
            self.lower_band[0],
            self.rsi[0],
            self.volume_rsi[0],
            self.fast_ema[0],
            self.slow_ema[0],
            self.macd_line[0],
            self.signal_line[0],
            self.vix[0],
        ]

        features_df = pd.DataFrame([features], columns=self.feature_names)

        # Scale the features
        scaled_features = self.scaler.transform(features_df)

        # Make prediction with the model
        prediction = self.model.predict(scaled_features)[0]  # Model should output 1 for "buy" and 0 for "sell"

        # Implement the model's decision
        if prediction == 1:  # Buy signal
            if self.position:  # Already in a position
                self.log("Holding position.")
            else:
                self.log("All in (Buy).")
                self.buy(size=0.90*self.broker.get_cash() // self.data.close[0])
        elif prediction == 0:  # Sell signal
            if self.position:  # Only close if there's an open position
                self.log("Closing position (Sell).")
                self.close()

    def log(self, txt):
        """
        Logs a message with the current date.
        """
        if self.verbose:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def notify_order(self, order):
        """
        Notifies when an order is completed, canceled, or rejected.
        """
        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f"Executed BUY (Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f})"
                )
            else:
                self.log(
                    f"Executed SELL (Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f})"
                )
        elif order.status == order.Canceled:
            self.log("Order was canceled.")
        elif order.status == order.Margin:
            self.log("Order canceled on margin.")
        elif order.status == order.Rejected:
            self.log("Order was rejected.")

        self.order = None
