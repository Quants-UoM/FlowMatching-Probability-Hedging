import yfinance as yf
import pandas as pd


START_DATE = "1996-01-01"


def fetch_stock_data(ticker, period="30y", interval="1d"):
    """
    Fetch historical data for a given ticker using yfinance.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data


def clean_series(series):
    """Convert index to datetime, sort, and trim to START_DATE."""
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[series.index >= START_DATE]   
    return series


def main():
    # Fetch raw data
    snp500raw = fetch_stock_data("^GSPC")
    djiaraw = fetch_stock_data("^DJI")
    nasdaqraw = fetch_stock_data("^IXIC")
    russel2000raw = fetch_stock_data("^RUT")
    nyseraw = fetch_stock_data("^NYA")
    nysesmcapraw = fetch_stock_data("^XAX")

    snp500 = clean_series(snp500raw["Close"])
    djia = clean_series(djiaraw["Close"])
    nasdaq = clean_series(nasdaqraw["Close"])
    russel2000 = clean_series(russel2000raw["Close"])
    nyse = clean_series(nyseraw["Close"])
    nysesmcap = clean_series(nysesmcapraw["Close"])

    snp500.to_csv("data/snp500.csv")
    djia.to_csv("data/djia.csv")
    nasdaq.to_csv("data/nasdaq.csv")
    russel2000.to_csv("data/russel2000.csv")
    nyse.to_csv("data/nyse.csv")
    nysesmcap.to_csv("data/nysesmcap.csv")

    print("CSV files saved (starting from 1996-01-01).")


if __name__ == "__main__":
    main()
