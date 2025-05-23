import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def compute_indicators(df, timeframe='daily'):
    """
    Compute technical indicators on the given OHLCV DataFrame.
    Returns the DataFrame with new indicator columns.
    """
    required_cols = {'Open','High','Low','Close','Volume'}
    if df.empty or not required_cols.issubset(df.columns):
        return df

    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
    df[f'RSI_{timeframe}'] = rsi

    # ADX
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df[f'ADX_{timeframe}'] = adx.adx()
    df[f'ADX_pos_{timeframe}'] = adx.adx_pos()
    df[f'ADX_neg_{timeframe}'] = adx.adx_neg()

    # Stochastic
    stoch = StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=14, smooth_window=3
    )
    df[f'STOCHk_{timeframe}'] = stoch.stoch()
    df[f'STOCHd_{timeframe}'] = stoch.stoch_signal()

    # Commodity Channel Index (CCI)
    cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20, constant=0.015)
    df[f'CCI_{timeframe}'] = cci.cci()

    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df[f'MACD_{timeframe}'] = macd.macd()
    df[f'MACD_signal_{timeframe}'] = macd.macd_signal()
    df[f'MACD_hist_{timeframe}'] = macd.macd_diff()

    # EMAs for trend
    if timeframe == 'daily':
        df['EMA50_daily'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA200_daily'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
    elif timeframe == 'hourly':
        df['EMA50_hourly'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA200_hourly'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df[f'BB_upper_{timeframe}'] = bb.bollinger_hband()
    df[f'BB_lower_{timeframe}'] = bb.bollinger_lband()
    df[f'BB_middle_{timeframe}'] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    )
    df[f'ATR_{timeframe}'] = atr.average_true_range()

    return df

def compute_anchored_vwap(df_1d):
    """
    Compute anchored VWAP from the lowest low of the past 52 weeks on daily data.
    Returns a Series of anchored VWAP values (NaN before the anchor date).
    """
    if df_1d.empty:
        return pd.Series(dtype=float)

    lookback_period = 252  # ~1 year
    recent_period = df_1d.tail(lookback_period)
    anchor_idx = recent_period['Low'].idxmin()
    if pd.isna(anchor_idx):
        return pd.Series(np.nan, index=df_1d.index)

    df_after = df_1d.loc[anchor_idx:].copy()
    cum_vol = df_after['Volume'].cumsum()
    cum_vol_price = (df_after['Close'] * df_after['Volume']).cumsum()
    vwap = cum_vol_price / cum_vol

    vwap_full = pd.Series(np.nan, index=df_1d.index)
    vwap_full.loc[vwap.index] = vwap
    vwap_full.name = 'AnchoredVWAP'
    return vwap_full

def to_daily(intra_df, label):
    """
    Convert intraday data to daily frequency using the last valid bar of each day.
    """
    if intra_df.empty:
        return pd.DataFrame()
    daily_data = intra_df.groupby(intra_df.index.date).tail(1)
    daily_data.index = pd.to_datetime(daily_data.index.date)
    daily_data.index.name = 'Date'
    return daily_data
