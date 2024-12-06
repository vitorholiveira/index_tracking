import pandas as pd
import numpy as np
import yfinance as yf

def build_dataset(index_ticker, stock_tickers, start_date, end_date, max_missing_days=30):
    """
    Builds a dataset by downloading, cleaning, and calculating variance for a list of stock tickers.

    Parameters:
    - index_ticker: Ticker symbol for the index to include in the dataset
    - start_date: Starting date for the historical data download
    - end_date: Ending date for the historical data download
    - max_missing_days: Maximum allowed consecutive days of missing data per stock
    
    Returns:
    - DataFrame with historical stock data and variance
    """
    # Download data
    values = download_data(index_ticker, stock_tickers, start_date, end_date)
    values = clean_data(values, max_missing_days)

    # Calculate variance and save
    variance = calculate_variance(values)
    variance_filename = f"stock_variance_{index_ticker}.csv"
    variance.to_csv(variance_filename)
    print(f"Stock variance saved to {variance_filename}")

    values_filename = f"stock_values_{index_ticker}.csv"
    values.to_csv(values_filename)
    print(f"Stock values saved to {values_filename}")

    # Generate shares outstanding and calculate weights
    #shares_df = generate_shares_outstanding(stock_tickers)
    #weights = calculate_weights(data, shares_df, index_ticker)
    #weights_filename = f"stock_weights_{index_ticker}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
    #weights.to_csv(weights_filename)
    #print(f"Stock weights saved to {weights_filename}")

    return values, variance

def download_data(index_ticker, stock_tickers, start_date, end_date):
    """
    Downloads historical adjusted close prices for an index and a list of stocks.

    :param index_ticker: Ticker symbol for the index
    :param start_date: Starting date for data download
    :param end_date: Ending date for data download
    :return: DataFrame containing the adjusted close prices for all tickers
    """
    try:
        # Combine the index and stock tickers into a single list
        all_tickers = [index_ticker] + stock_tickers
        data: pd.DataFrame = pd.DataFrame()
        data = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
        # Remove timezone information from the index (dates)
        data.index = data.index.strftime('%Y-%m-%d')
        print("Data downloaded successfully.")
        return data
    except Exception as e:
        print(f"An error occurred on data download: {e}")
        return None

def clean_data(data, max_missing_days=30):
    """
    Cleans stock data by removing stocks with prolonged missing data and interpolating gaps.

    :param data: DataFrame with stock price data
    :param max_missing_days: Maximum allowed consecutive missing days per stock
    :return: Cleaned DataFrame with interpolated data and stocks with prolonged gaps removed
    """
    try:
        data = remove_stocks_with_prolonged_missing_data(data, days_limit=max_missing_days)
        # Interpolate remaining missing values linearly using adjacent data points
        data = data.interpolate(method='linear')

        print("Data cleaned successfully.")
        return data
    except Exception as e:
        print(f"An error occurred on clean data: {e}")
        return None

def remove_stocks_with_prolonged_missing_data(data, days_limit=30):
    """
    Removes stocks with consecutive missing data days exceeding the specified limit.

    :param data: DataFrame with stock price data
    :param days_limit: Maximum allowed consecutive missing days per stock
    :return: DataFrame with only stocks that meet the missing data criteria
    """
    def check_consecutive_nans(column):
        # Create a binary series (1 for NaN, 0 for data) to identify NaN sequences
        nan_seq = column.isna().astype(int)
        # Calculate the longest consecutive sequence of NaNs
        max_nan_streak = nan_seq.groupby((nan_seq != nan_seq.shift()).cumsum()).transform('sum').max()

        return max_nan_streak >= days_limit

    # Select stocks that do not exceed the allowed limit of consecutive missing data days
    try:
        valid_columns = [col for col in data.columns if not check_consecutive_nans(data[col])]
        filtered_data = data[valid_columns]

        print(f"Removed stocks: {set(data.columns) - set(valid_columns)}")
        return filtered_data
    except Exception as e:
        print(f"An error occurred o remove stocks with consecutive missing data: {e}")
        return None

def calculate_variance(data):
    """
    Calculates daily variance for the given stock data.

    :param data: DataFrame with cleaned stock price data
    :return: DataFrame with daily variance for each stock
    """
    try:
        # Calculate the daily variance by percentage change in stock prices
        variance = data.pct_change()
        variance = variance.iloc[1:, :]

        return variance
    except Exception as e:
        print(f"An error occurred in calculate variance: {e}")
        return None
    
def generate_shares_outstanding(stock_tickers):
    """
    Generates a DataFrame containing the shares outstanding for a list of stock tickers.

    :param stock_tickers: List of stock tickers
    :return: DataFrame containing Ticker and Shares Outstanding
    """
    data = [
        {'Ticker': ticker, 'Shares Outstanding': yf.Ticker(ticker).info.get('sharesOutstanding')}
        for ticker in stock_tickers
        if yf.Ticker(ticker).info.get('sharesOutstanding') is not None
    ]
    
    # Cria um DataFrame diretamente a partir da lista
    return pd.DataFrame(data)


def get_tickers(csv_file):
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(csv_file)
        
        symbols = df['Symbol'].tolist()
        print(symbols)
        return symbols
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        return []
