import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_price_change(days=2, ticker="btc"):
    """
    Fetches Bitcoin price data for a specified number of days ago.

    Parameters:
    - days (int): Number of days ago for which to fetch data (default is 1).
    - ticker (str): Ticker symbol for the cryptocurrency (default is "BTC").

    Returns:
    - float: Closing price of Bitcoin 'days' days ago.
    """

    # Calculate the start and end dates based on the provided number of days
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

    #print(f"end: {end_date}\nstart: {start_date}")
    # Fetch historical price data using yfinance
    data = yf.download(f"{ticker}-USd", start=start_date, end=end_date,progress=False)

    #print(data.Close)
    # Extract the closing price 'days' days ago
    #print(len(data), days)
    if len(data) >= days:
        closing_price = data['Close'].iloc[-days]
        return closing_price
    else:
        return None  # Return None if there is not enough data for the specified days




if __name__ == "__main__":
    print(get_price_change())