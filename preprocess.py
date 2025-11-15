import pandas as pd
import numpy as np


def z_score(series, span=100):
    """
    Calculate rolling z score of percentage returns.
    """
    ret = series.pct_change()
    mean = ret.rolling(span).mean()
    std = ret.rolling(span).std()
    z_ret = (ret - mean) / std
    return z_ret


def ema_signal(series, span_fast=20, span_slow=100):
    """
    Calculate EMA signal as fast minus slow.
    """
    ema_fast = series.ewm(span=span_fast).mean()
    ema_slow = series.ewm(span=span_slow).mean()
    ema_sig = ema_fast - ema_slow
    return ema_sig


def vol_scaled_return(series, window=100, target_vol=0.20):
    """
    Volatility scaled log return with rolling realised volatility.
    """
    log_ret = np.log(series / series.shift(1))
    realized_vol = log_ret.rolling(window).std() * np.sqrt(252.0)
    vol_scaled_ret = log_ret * (target_vol / realized_vol)
    return vol_scaled_ret


def main():

    djia = pd.read_csv("data/djia.csv", index_col=0, parse_dates=True)
    snp500 = pd.read_csv("data/snp500.csv", index_col=0, parse_dates=True)
    nasdaq = pd.read_csv("data/nasdaq.csv", index_col=0, parse_dates=True)
    russel2000 = pd.read_csv("data/russel2000.csv", index_col=0, parse_dates=True)
    nyse = pd.read_csv("data/nyse.csv", index_col=0, parse_dates=True)
    nysesmcap = pd.read_csv("data/nysesmcap.csv", index_col=0, parse_dates=True)

    data_dict = {
        "djia": djia,
        "snp500": snp500,
        "nasdaq": nasdaq,
        "russel2000": russel2000,
        "nyse": nyse,
        "nysesmcap": nysesmcap,
    }

    for name, df in data_dict.items():
        if "Close" not in df.columns:
            df.columns = ["Close"]

        close = df["Close"]

        df["z_score"] = z_score(close)
        df["ema_signal"] = ema_signal(close)
        df["vol_scaled_ret"] = vol_scaled_return(close)

        df_clean = df.dropna()

        df_clean.to_csv(f"data/{name}_features.csv")

    print("Saved per index feature files with z_score, ema_signal, vol_scaled_ret.")


if __name__ == "__main__":
    main()
