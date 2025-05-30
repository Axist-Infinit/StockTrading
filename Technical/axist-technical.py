#!/usr/bin/env python3
"""
Swing Trading Strategy Script with Feature Refinement and Backtesting

Usage examples:
  1) Normal signal generation (for current data):
     python my_trading_script.py AAPL MSFT

  2) Backtest a single ticker over a date range:
     python my_trading_script.py --backtest --start 2022-01-01 --end 2022-12-31 AAPL

This script can:
- Generate swing trade signals (1-5 day hold) for given stock tickers in "live" mode
- Perform a basic historical backtest over a given date range
- Log the signals/trades to a CSV file if desired
"""

import os
import sys
import argparse
import datetime
import json
import joblib, pandas as pd
import numpy as np
import yfinance as yf
import urwid
from pathlib import Path
from watchlist_utils import load_watchlist, save_watchlist, manage_watchlist
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from ta.volatility import AverageTrueRange
from ta.volume import MFIIndicator  # optional if you want more volume-based features
from ta.volatility import DonchianChannel  # optional, if needed


from colorama import init, Fore, Style
init(autoreset=True)

PREDICTIONS_FILE = "weekly_signals.json"

DEFAULT_INTERVALS_CONFIG = {
    '5m':  {'period': '14d',  'interval': '5m'},
    '30m': {'period': '60d',  'interval': '30m'},
    '1h':  {'period': '120d', 'interval': '1h'},
    '90m': {'period': '60d',  'interval': '90m'},
    '1d':  {'period': '380d', 'interval': '1d'}
}

DATA_CACHE: dict[tuple, tuple] = {}      # key → (df_5m, df_30m, df_1h, df_90m, df_1d)

def get_or_fetch(ticker: str, start=None, end=None):
    """
    Return the tuple from ``fetch_data`` but avoid duplicate calls
    within the *same* run.

    Usage
    -----
    df_5m, df_30m, df_1h, df_90m, df_1d = get_or_fetch(sym, start, end)
    """
    key = (ticker, start, end)
    if key not in DATA_CACHE:
        DATA_CACHE[key] = fetch_data(ticker, start=start, end=end)
    return DATA_CACHE[key]

def batch_fetch_data_for_tickers(
    tickers: list[str],
    requested_intervals: list[str] | None = None,
    cache_dir: str = ".ohlcv_cache",
    start_date_override_1d: str | None = None,
    end_date_override_1d: str | None = None
) -> dict:
    """
    Fetches data for multiple tickers and multiple intervals using batch yfinance calls.
    Returns a dictionary: {ticker: {interval_name: df, ...}, ...}
    Caches results per interval for the batch of tickers.
    """
    Path(cache_dir).mkdir(exist_ok=True)
    all_data = {ticker: {} for ticker in tickers}
    # Determine which intervals to fetch
    intervals_to_fetch = DEFAULT_INTERVALS_CONFIG.keys()
    if requested_intervals and len(requested_intervals) > 0:
        intervals_to_fetch = [i for i in requested_intervals if i in DEFAULT_INTERVALS_CONFIG]

    if not intervals_to_fetch:
        # This case should ideally not be reached if requested_intervals is validated or DEFAULT_INTERVALS_CONFIG is always non-empty
        print("Warning: No valid intervals to fetch specified.")
        return all_data

    processed_tickers = set() # To handle cases where a ticker might not return data for an interval

    for interval_name in intervals_to_fetch:
        params = DEFAULT_INTERVALS_CONFIG[interval_name]
        batch_cache_filename_parts = sorted(list(set(tickers))) # Ensure consistent naming

        use_override_for_1d = interval_name == '1d' and start_date_override_1d and end_date_override_1d

        if use_override_for_1d:
            cache_file_name = Path(cache_dir) / f"{'_'.join(batch_cache_filename_parts)}_1d_{start_date_override_1d}_{end_date_override_1d}.pkl"
        else:
            cache_file_name = Path(cache_dir) / f"{'_'.join(batch_cache_filename_parts)}_{interval_name}_{params['period']}_{params['interval']}.pkl"

        # Cache check
        # Max age for cache can be dynamic based on interval, similar to _download_cached
        # For simplicity, let's use a fixed max_age or adapt logic from _download_cached if necessary
        # This example uses a simple existence check; add freshness check for robustness
        if cache_file_name.exists():
            try:
                # This cache stores data for ALL tickers for THIS interval
                interval_batch_data = joblib.load(cache_file_name)
                # Distribute to individual tickers
                for ticker in tickers:
                    if ticker in interval_batch_data:
                         all_data[ticker][interval_name] = interval_batch_data[ticker]
                # If all tickers for this interval are loaded from cache, continue
                if all(interval_name in all_data[t] for t in tickers):
                    print(f"Loaded {interval_name} for {', '.join(tickers)} from cache.")
                    continue
            except Exception as e:
                print(f"Error loading cache file {cache_file_name}: {e}. Refetching.")
                cache_file_name.unlink(missing_ok=True)

        print(f"Fetching {interval_name} for {', '.join(tickers)}...")
        # Use threads=True for yfinance to handle multiple tickers efficiently
        # For '1d' interval, if start/end are needed, this function signature might need adjustment
        # or handle it specifically if params contain start/end.
        # Current DEFAULT_INTERVALS_CONFIG uses 'period'.
        try:
            if use_override_for_1d:
                bulk_data = safe_download(
                    tickers,
                    start=start_date_override_1d,
                    end=end_date_override_1d,
                    interval=params['interval'], # Should be '1d'
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
            else:
                bulk_data = safe_download(
                    tickers,
                    period=params['period'],
                    interval=params['interval'],
                    auto_adjust=True,
                    progress=False,
                    threads=True # yfinance handles threading for multiple tickers
                )
        except Exception as e:
            print(f"Error downloading {interval_name} for {tickers}: {e}")
            bulk_data = pd.DataFrame() # Ensure bulk_data is a DataFrame

        if bulk_data.empty:
            print(f"No data returned for {interval_name} for {', '.join(tickers)}.")
            # Assign empty DFs to all tickers for this interval if no data
            for ticker in tickers:
                all_data[ticker][interval_name] = pd.DataFrame()
            continue

        # Post-process and cache
        # This will store data for ALL tickers for THIS interval in one file
        current_interval_data_to_cache = {}
        for ticker in tickers:
            try:
                if not tickers: # Should not happen if loop is over tickers
                    continue

                if len(tickers) == 1 and ticker in bulk_data.columns: # Single ticker, direct access
                    df = bulk_data.copy()
                elif isinstance(bulk_data.columns, pd.MultiIndex): # Multi ticker
                    # Filter columns for the current ticker
                    ticker_df = bulk_data.xs(ticker, level=1, axis=1)
                    # Dropping the first level of column index which is now 'ticker'
                    # ticker_df.columns = ticker_df.columns.droplevel(0) # No, this is not needed, xs handles it.
                    df = ticker_df.copy()
                else: # Single ticker download might not have MultiIndex
                     df = bulk_data.copy()


                # Clean data - similar to _clean in fetch_data
                if 'Adj Close' in df.columns and 'Close' not in df.columns:
                    df.rename(columns={'Adj Close': 'Close'}, inplace=True)

                # Ensure standard OHLCV columns exist, even if empty after selection
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = np.nan # Add missing expected columns as NaN

                df.dropna(how='all', inplace=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                all_data[ticker][interval_name] = df
                current_interval_data_to_cache[ticker] = df
                processed_tickers.add(ticker)

            except KeyError: # Ticker might not be in results if yfinance returns no data for it
                print(f"No data for {ticker} in {interval_name} results. Assigning empty DataFrame.")
                all_data[ticker][interval_name] = pd.DataFrame()
                current_interval_data_to_cache[ticker] = pd.DataFrame()
            except Exception as e:
                print(f"Error processing {ticker} for {interval_name}: {e}")
                all_data[ticker][interval_name] = pd.DataFrame()
                current_interval_data_to_cache[ticker] = pd.DataFrame()

        # Cache the processed data for this interval batch
        if current_interval_data_to_cache: # Only save if there's something to cache
            try:
                joblib.dump(current_interval_data_to_cache, cache_file_name)
            except Exception as e:
                print(f"Error saving cache file {cache_file_name}: {e}")

    # Ensure all tickers have keys for the intervals that were attempted, even if some failed
    for ticker in tickers:
        for interval_name in intervals_to_fetch: # Use intervals_to_fetch here
            if interval_name not in all_data[ticker]:
                all_data[ticker][interval_name] = pd.DataFrame()

    return all_data

def load_predictions():
    """Return cached predictions; add default keys for older records."""
    if not os.path.isfile(PREDICTIONS_FILE):
        return []
    try:
        with open(PREDICTIONS_FILE, "r") as fh:
            data = json.load(fh)
    except Exception:
        return []

    for rec in data:
        rec.setdefault("status", "Open")
        rec.setdefault("entry_date",
                       datetime.datetime.today().strftime("%Y-%m-%d"))
    return data

def save_predictions(pred_list):
    """Overwrite the cache with *pred_list*."""
    with open(PREDICTIONS_FILE, "w") as fh:
        json.dump(pred_list, fh, indent=2)

#warnings.filterwarnings("ignore")

def _update_positions_status() -> None:
    """
    Walk every *Open* position in weekly_signals.json, check the latest close
    versus its stop / target, and convert it to Stop/Target status if hit.
    The file is rewritten only when at least one trade changes state.
    """
    import datetime
    import yfinance as yf

    preds = load_predictions()
    changed = False

    for p in preds:
        if p.get('status') != 'Open':
            continue

        sym   = p['symbol']
        try:
            now_price = yf.Ticker(sym).history(period='1d')['Close'].iloc[-1]
        except Exception:
            continue    # skip this symbol if Yahoo fails today

        direction = p['direction']
        hit_stop   = hit_target = False

        if direction == 'LONG':
            hit_stop   = now_price <= p['stop_loss']
            hit_target = now_price >= p['profit_target']
        elif direction == 'SHORT':
            hit_stop   = now_price >= p['stop_loss']
            hit_target = now_price <= p['profit_target']

        if hit_stop or hit_target:
            p['status']     = 'Stop' if hit_stop else 'Target'
            p['exit_date']  = datetime.date.today().isoformat()
            p['exit_price'] = round(float(now_price), 2)
            changed = True

    if changed:
        save_predictions(preds)

def safe_download(*args, **kwargs):

    import time, yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    for attempt in range(5):
        try:
            return yf.download(*args, **kwargs)
        except YFRateLimitError:
            wait = 2 ** attempt
            print(f"⚠️  Yahoo rate-limit – retrying in {wait}s …")
            time.sleep(wait)

    # last attempt without catch → propagate error
    return yf.download(*args, **kwargs)


def _download_cached(ticker: str,
                     period: str,
                     interval: str,
                     cache_dir: str = ".ohlcv_cache") -> pd.DataFrame:
    """
    Return a tz-naïve OHLCV frame, cached on disk.

    ▸ The cache is considered “fresh” until it is SIX bars old
      instead of just one.

    ▸ If a fresh copy is missing we fall back to `safe_download`
      (the exponential-back-off wrapper).
    """
    Path(cache_dir).mkdir(exist_ok=True)
    fname = Path(cache_dir) / f"{ticker}_{period}_{interval}.pkl"

    # ----   how old is too old?  ------------------------------------
    sec_per_bar = {"5m": 300, "30m": 1800, "1h": 3600,
                   "90m": 5400, "1d": 86400}
    max_age = sec_per_bar.get(interval, 3600) * 6      # << change

    if fname.exists():
        try:
            cached: pd.DataFrame = joblib.load(fname)
            if not cached.empty:
                age = (pd.Timestamp.utcnow() - cached.index[-1]).total_seconds()
                if age < max_age:
                    return cached
        except Exception:
            fname.unlink(missing_ok=True)              # corrupt → rebuild

    # ----   fresh pull with polite retries  -------------------------
    fresh = safe_download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False, threads=False)

    if isinstance(fresh.columns, pd.MultiIndex):
        fresh.columns = fresh.columns.droplevel(1)
    if 'Adj Close' in fresh.columns and 'Close' not in fresh.columns:
        fresh.rename(columns={'Adj Close': 'Close'}, inplace=True)

    if fname.exists():
        try:
            old = joblib.load(fname)
            fresh = pd.concat([old, fresh]).sort_index().drop_duplicates()
        except Exception:
            pass

    if fresh.index.tz is not None:
        fresh.index = fresh.index.tz_localize(None)

    joblib.dump(fresh, fname)
    return fresh

def preload_interval_cache(symbols: list[str],
                           period: str,
                           interval: str,
                           cache_dir: str = ".ohlcv_cache") -> None:
    symbols = sorted(set(symbols))
    if not symbols:
        return

    # one bulk call – threads=False keeps it to ONE HTTPS request
    bulk = safe_download(symbols, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
    if bulk.empty:
        return

    if isinstance(bulk.columns, pd.MultiIndex):
        bulk = bulk.swaplevel(axis=1)          # → symbol 1st level
        bulk.sort_index(axis=1, inplace=True)

    Path(cache_dir).mkdir(exist_ok=True)
    for sym in symbols:
        df = bulk[sym].dropna(how="all")
        if df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        joblib.dump(df,
                    Path(cache_dir) / f"{sym}_{period}_{interval}.pkl")


def fetch_data(ticker, start=None, end=None, intervals=None, warmup_days=300):
    """
    Returns (df_5m, df_30m, df_1h, df_90m, df_1d) – all timezone-naive.
    Each yfinance request is routed through _download_cached().
    """


    if intervals is None:
        intervals = {
            '5m':  ('14d',  '5m'),
            '30m': ('60d',  '30m'),
            '1h':  ('120d', '1h'),
            '90m': ('60d',  '90m'),
            '1d':  ('380d', '1d')
        }

    # intraday
    df_5m  = _download_cached(ticker, *intervals['5m'])
    df_30m = _download_cached(ticker, *intervals['30m'])
    df_1h  = _download_cached(ticker, *intervals['1h'])
    df_90m = _download_cached(ticker, *intervals['90m'])

    # daily
    if start and end:
        real_start = (pd.to_datetime(start) -
                      datetime.timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        df_1d = yf.download(ticker, start=real_start, end=end,
                            interval='1d', auto_adjust=True, progress=False)
    else:
        df_1d = _download_cached(ticker, *intervals['1d'])

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        df.dropna(how='all', inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    return tuple(map(_clean, (df_5m, df_30m, df_1h, df_90m, df_1d)))

def compute_indicators(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """
    Compute a standard indicator set on OHLCV data.
    If the DataFrame has fewer than `window` rows (default 14) the function
    exits early and adds stub columns filled with NaNs so downstream code
    never breaks.
    """
    df = df.copy()  # <-- ADDED to avoid SettingWithCopyWarning

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not required_cols.issubset(df.columns):
        return df

    window = 14
    if len(df) < window:
        # Stub columns if insufficient data
        stub_cols = [
            f"RSI_{timeframe}",
            f"ADX_{timeframe}", f"ADX_pos_{timeframe}", f"ADX_neg_{timeframe}",
            f"STOCHk_{timeframe}", f"STOCHd_{timeframe}",
            f"CCI_{timeframe}",
            f"MACD_{timeframe}", f"MACD_signal_{timeframe}", f"MACD_hist_{timeframe}",
            f"BB_upper_{timeframe}", f"BB_lower_{timeframe}", f"BB_middle_{timeframe}",
            f"ATR_{timeframe}",
        ]
        for col in stub_cols:
            df[col] = np.nan
        return df

    # RSI
    rsi = RSIIndicator(close=df["Close"], window=window).rsi()
    df[f"RSI_{timeframe}"] = rsi

    # ADX
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=window)
    df[f"ADX_{timeframe}"] = adx.adx()
    df[f"ADX_pos_{timeframe}"] = adx.adx_pos()
    df[f"ADX_neg_{timeframe}"] = adx.adx_neg()

    # Stochastic
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=window, smooth_window=3
    )
    df[f"STOCHk_{timeframe}"] = stoch.stoch()
    df[f"STOCHd_{timeframe}"] = stoch.stoch_signal()

    # CCI
    cci = CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=20, constant=0.015
    )
    df[f"CCI_{timeframe}"] = cci.cci()

    # MACD
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df[f"MACD_{timeframe}"] = macd.macd()
    df[f"MACD_signal_{timeframe}"] = macd.macd_signal()
    df[f"MACD_hist_{timeframe}"] = macd.macd_diff()

    # EMAs (only for daily/hourly)
    if timeframe in {"daily", "hourly"}:
        df[f"EMA50_{timeframe}"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
        df[f"EMA200_{timeframe}"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df[f"BB_upper_{timeframe}"] = bb.bollinger_hband()
    df[f"BB_lower_{timeframe}"] = bb.bollinger_lband()
    df[f"BB_middle_{timeframe}"] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=window
    )
    df[f"ATR_{timeframe}"] = atr.average_true_range()

    return df


def compute_anchored_vwap(
        df: pd.DataFrame,
        lookback_bars: int = 252
) -> pd.Series:
    if df.empty or not {"Close", "Low", "Volume"}.issubset(df.columns):
        return pd.Series(dtype=float, index=df.index)

    recent  = df.tail(lookback_bars)
    anchor  = recent['Low'].idxmin()
    if pd.isna(anchor):
        return pd.Series(np.nan, index=df.index)

    after        = df.loc[anchor:].copy()
    cum_vol      = after['Volume'].cumsum()
    cum_dollars  = (after['Close'] * after['Volume']).cumsum()
    vwap_series  = cum_dollars / cum_vol

    out = pd.Series(np.nan, index=df.index)
    out.loc[vwap_series.index] = vwap_series
    out.name = 'AnchoredVWAP'
    return out



def to_daily(intra_df, label):
    """
    Convert intraday data to daily frequency using the last valid bar of each day.
    """
    if intra_df.empty:
        #print(f"[DEBUG] to_daily: {label} is empty, skipping.")
        return pd.DataFrame()
    daily_data = intra_df.groupby(intra_df.index.date).tail(1)
    daily_data.index = pd.to_datetime(daily_data.index.date)
    daily_data.index.name = 'Date'
    return daily_data


def prepare_features(
    df_5m:  pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_90m: pd.DataFrame,
    df_1d:  pd.DataFrame,
    *,
    horizon: int = 10,
    drop_recent: bool = True
) -> pd.DataFrame:

    # ---------- indicators -------------------------------------------
    ind_5m  = compute_indicators(df_5m.copy(),  timeframe='5m')
    ind_30m = compute_indicators(df_30m.copy(), timeframe='30m')
    ind_1h  = compute_indicators(df_1h.copy(),  timeframe='hourly')
    ind_90m = compute_indicators(df_90m.copy(), timeframe='90m')
    ind_1d  = compute_indicators(df_1d.copy(),  timeframe='daily')
    ind_5m['AnchoredVWAP_5m']   = compute_anchored_vwap(ind_5m,  lookback_bars=2000)
    ind_30m['AnchoredVWAP_30m'] = compute_anchored_vwap(ind_30m, lookback_bars=200)
    ind_1h['AnchoredVWAP_1h']   = compute_anchored_vwap(ind_1h,  lookback_bars=120)
    ind_1d['AnchoredVWAP']      = compute_anchored_vwap(ind_1d,  lookback_bars=252)

    # ---------- collapse intraday to daily ---------------------------
    daily_5m  = to_daily(ind_5m,  "5m")
    daily_30m = to_daily(ind_30m, "30m")
    daily_1h  = to_daily(ind_1h,  "1h")
    daily_90m = to_daily(ind_90m, "90m")

    ind_1d.index.name = 'Date'
    features_df = (
        ind_1d
          .join(daily_5m,  rsuffix='_5m')
          .join(daily_30m, rsuffix='_30m')
          .join(daily_1h,  rsuffix='_1h')
          .join(daily_90m, rsuffix='_90m')
    )

    # ---------- label & post-process (unchanged) ---------------------
    features_df.dropna(subset=['Close'], inplace=True)
    if features_df.empty:
        return features_df

    atr = features_df['ATR_daily'].fillna(0)
    upper = features_df['Close'] + 2.0 * atr
    lower = features_df['Close'] - 2.0 * atr
    closes = features_df['Close'].values
    labels = np.ones(len(features_df), dtype=int)

    for i in range(len(features_df) - horizon):
        win = closes[i + 1 : i + 1 + horizon]
        up  = np.where(win >= upper.iloc[i])[0]
        dn  = np.where(win <= lower.iloc[i])[0]
        if up.size and dn.size:
            labels[i] = 2 if up[0] < dn[0] else 0
        elif up.size:
            labels[i] = 2
        elif dn.size:
            labels[i] = 0

    features_df['future_class'] = labels
    if drop_recent:
        features_df = features_df.iloc[:-horizon]
    else:
        features_df.loc[features_df.index[-horizon:], 'future_class'] = np.nan

    return features_df


def refine_features(features_df, importance_cutoff=0.0001, corr_threshold=0.9):
    """
    Simple feature-refinement approach:
    1) Train a quick baseline XGBoost model to get feature importances.
    2) Drop near-zero importance features.
    3) Drop one of any pair of features with high correlation (above corr_threshold).
    """
    if features_df.empty or 'future_class' not in features_df.columns:
        return features_df

    y = features_df['future_class']
    X = features_df.drop(columns=['future_class']).copy()

    # Fill missing
    X.ffill(inplace=True)
    X.bfill(inplace=True)

    # Quick baseline XGB
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',      # GPU
        device='cuda'    # GPU
    )

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(np.unique(y_train)) < 2:
        #print("[DEBUG] refine_features: Not enough classes to train. Skipping.")
        return features_df

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feat_importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    # Drop near-zero importance
    drop_by_importance = feat_importance_series[feat_importance_series < importance_cutoff].index.tolist()
    if drop_by_importance:
        X.drop(columns=drop_by_importance, inplace=True, errors='ignore')

    # Drop highly correlated features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numeric_cols].corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_by_corr = [col for col in upper.columns if any(upper[col].abs() > corr_threshold)]
    if drop_by_corr:
        X.drop(columns=drop_by_corr, inplace=True, errors='ignore')

    refined_df = X.join(y)
    return refined_df


def tune_threshold_and_train(features_df):
    """
    Tune the future return threshold for classification, then train the final XGBoost model.
    Returns (model, best_threshold).
    """
    if features_df.empty or 'future_class' not in features_df.columns:
        return None, None

    feature_cols = [c for c in features_df.columns if c != 'future_class']
    X_full = features_df[feature_cols].copy()
    X_full.ffill(inplace=True)
    X_full.bfill(inplace=True)

    y_full = features_df['future_class']
    split_idx = int(len(features_df) * 0.8)
    X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    train_data = pd.concat([X_train, y_train], axis=1)
    class_0 = train_data[train_data['future_class'] == 0]
    class_1 = train_data[train_data['future_class'] == 1]
    class_2 = train_data[train_data['future_class'] == 2]

    # If significantly fewer 0 or 2 than 1, oversample them
    max_count = max(len(class_0), len(class_1), len(class_2))
    class_0_up = resample(class_0, replace=True, n_samples=max_count, random_state=42)
    class_1_up = resample(class_1, replace=True, n_samples=max_count, random_state=42)
    class_2_up = resample(class_2, replace=True, n_samples=max_count, random_state=42)

    train_oversampled = pd.concat([class_0_up, class_1_up, class_2_up], axis=0)
    X_train = train_oversampled.drop(columns=['future_class'])
    y_train = train_oversampled['future_class']

    best_thr = None
    best_score = -np.inf

    # Simple threshold tuning in 1%-5% increments
    for thr in [0.01, 0.02, 0.03, 0.04, 0.05]:
        if 'Close' not in features_df.columns:
            continue
        future_ret_train = (features_df['Close'][:split_idx].shift(-5) / features_df['Close'][:split_idx] - 1.0)
        future_ret_test  = (features_df['Close'][split_idx:].shift(-5) / features_df['Close'][split_idx:] - 1.0)

        y_train_temp = y_train.copy()
        y_test_temp  = y_test.copy()

        # Re-label train
        y_train_temp[:] = 1
        y_train_temp[future_ret_train > thr] = 2
        y_train_temp[future_ret_train < -thr] = 0

        # Re-label test
        y_test_temp[:] = 1
        y_test_temp[future_ret_test > thr] = 2
        y_test_temp[future_ret_test < -thr] = 0

        unique_classes = np.unique(y_train_temp)
        if len(unique_classes) < 3:
            continue

        model = XGBClassifier(
#            objective='multi:softmax',
#            num_class=3,
#            use_label_encoder=False,
#            eval_metric='mlogloss',
#            verbosity=0,
#            tree_method='hist',      # GPU
#            device='cuda'   # GPU
            objective='multi:softprob',   # so we can use probabilities
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            tree_method='hist',
            device='cuda',
            max_depth=5,         # shallower trees
            min_child_weight=10, # require more samples per leaf
            gamma=1.0,           # min split loss
            subsample=0.8,       # row subsampling
            colsample_bytree=0.8,# feature subsampling
            learning_rate=0.05,  # slower learning
            n_estimators=500
        )
        model.fit(X_train, y_train_temp)
        y_pred = model.predict(X_test)
        report = classification_report(y_test_temp, y_pred, output_dict=True, zero_division=0)
        f1_long = report.get('2', {}).get('f1-score', 0.0)
        f1_short= report.get('0', {}).get('f1-score', 0.0)
        avg_f1  = (f1_long + f1_short)/2.0

        if avg_f1 > best_score:
            best_score = avg_f1
            best_thr   = thr

    if best_thr is None:
        return None, None

    # Rebuild final_y using best_thr
    future_ret_all = (features_df['Close'].shift(-5) / features_df['Close'] - 1.0)
    final_y = y_full.copy()
    final_y[:] = 1
    final_y[future_ret_all > best_thr] = 2
    final_y[future_ret_all < -best_thr] = 0

    if len(np.unique(final_y)) < 3:
        return None, best_thr

    final_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',      # GPU
        device='cuda'    # GPU
    )
    final_model.fit(X_full, final_y)
    return final_model, best_thr


def generate_signal_output(ticker, latest_row, model, thr):
    """
    Return a colourised LONG / SHORT string or None (skip bar).
    If ATR is missing/zero we fall back to 5 % of price instead of aborting.
    """
    latest_df = latest_row.to_frame().T
    probs = model.predict_proba(latest_df.values)[0]
    class_idx = int(np.argmax(probs))
    prob_score = probs[class_idx]

    # ignore low-confidence or neutral predictions
    if prob_score < 0.60 or class_idx == 1:
        return None

    direction = "LONG" if class_idx == 2 else "SHORT"
    price     = float(latest_row.get("Close", np.nan))

    # ─── safe ATR fallback ────────────────────────────────────────────
    atr = latest_row.get("ATR_daily", np.nan)
    if not atr or np.isnan(atr):
        atr = 0.05 * price          # ≈5 % of price as emergency width
        atr_note = " (est. ATR)"
    else:
        atr_note = ""

    stop   = price - atr if direction == "LONG" else price + atr
    target = price + 2*atr if direction == "LONG" else price - 2*atr

    # simple trend rationale
    ema50  = latest_row.get('EMA50_daily', np.nan)
    ema200 = latest_row.get('EMA200_daily', np.nan)
    trend  = ("up-trend (50>200 EMA)" if not np.isnan(ema50) and not np.isnan(ema200) and ema50 > ema200
              else "down-trend (50<200 EMA)")

    colour = Fore.GREEN if direction == "LONG" else Fore.RED
    return (f"{Fore.CYAN}{ticker}{Style.RESET_ALL}: {colour}{direction}{Style.RESET_ALL} "
            f"at ${price:.2f}, Stop ${stop:.2f}, Target ${target:.2f}, "
            f"Prob {prob_score:.2f} – {trend}{atr_note}")


def get_macro_data(start, end, fred_api_key=None):
    """
    Placeholder for macro data fetching.
    Returns an empty DataFrame with a DatetimeIndex or None.
    For now, it returns pd.DataFrame().
    """
    # Ensure pandas is imported. It's imported as `joblib, pandas as pd`
    # so direct use of pd is fine.
    return pd.DataFrame()


def backtest_strategy(ticker, start_date, end_date, macro_df: pd.DataFrame | None = None, log_file=None):
    """
    Daily-lumped back-test with Sharpe, max-DD, time-in-market and ATR guard.
    """
    # ── data & features ──────────────────────────────────────────────────
    df_5m, df_30m, df_1h, df_90m, df_1d = fetch_data(
        ticker, start=start_date, end=end_date
    )
    feat = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d)
    feat = refine_features(feat)
    model, _ = tune_threshold_and_train(feat)
    if model is None:
        print(Fore.YELLOW + f"Model training failed for {ticker}." + Style.RESET_ALL)
        return

    X_all   = feat.drop(columns=['future_class']).ffill().bfill()
    preds   = model.predict(X_all)
    probas  = model.predict_proba(X_all)
    df_pred = feat.copy()
    df_pred['prediction'] = preds

    # ── trade simulation ────────────────────────────────────────────────
    trades = []
    in_trade = False
    trade_dir = None
    entry_price = stop_price = target_price = None
    entry_time  = None

    for i in range(len(df_pred)):
        ts        = df_pred.index[i]
        row       = df_pred.iloc[i]
        price     = row['Close']
        atr       = row.get('ATR_daily', np.nan)
        if np.isnan(atr) or atr == 0:
            raise ValueError(f"[{ticker}] ATR NaN/zero at {ts}")

        if in_trade:
            if trade_dir == "LONG":
                if price >= target_price or price <= stop_price:
                    pnl = (price - entry_price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time, 'exit_timestamp': ts,
                        'direction': 'LONG',
                        'entry_price': round(entry_price, 2),
                        'exit_price':  round(price, 2),
                        'pnl_pct':     round(pnl, 4),
                        'stop_price':  round(stop_price, 2),
                        'target_price':round(target_price, 2)
                    })
                    in_trade = False
            else:  # SHORT
                if price <= target_price or price >= stop_price:
                    pnl = (entry_price - price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time, 'exit_timestamp': ts,
                        'direction': 'SHORT',
                        'entry_price': round(entry_price, 2),
                        'exit_price':  round(price, 2),
                        'pnl_pct':     round(pnl, 4),
                        'stop_price':  round(stop_price, 2),
                        'target_price':round(target_price, 2)
                    })
                    in_trade = False
            if in_trade:
                continue

        # flat – consider new entry
        class_idx  = int(row['prediction'])
        prob_score = probas[i][class_idx]
        if prob_score < 0.60:
            continue

        if class_idx == 2:      # LONG
            in_trade   = True
            trade_dir  = "LONG"
            entry_time = ts
            entry_price= price
            stop_price = price - atr
            target_price = price + 2 * atr
        elif class_idx == 0:    # SHORT
            in_trade   = True
            trade_dir  = "SHORT"
            entry_time = ts
            entry_price= price
            stop_price = price + atr
            target_price = price - 2 * atr

    if not trades:
        print(Fore.YELLOW + f"No trades for {ticker}." + Style.RESET_ALL)
        return

    # ── performance summary ─────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    summary   = _summarise_performance(trades_df, len(feat))

    pct = lambda x: f"{x*100:.2f}%"
    print(Fore.BLUE + f"\nBacktest Results for {ticker}" + Style.RESET_ALL +
          f" ({start_date} → {end_date}):")
    print(f"  Total trades        : {Fore.CYAN}{summary['total']}{Style.RESET_ALL}")
    print(f"  Win rate            : {Fore.CYAN}{pct(summary['win_rate'])}{Style.RESET_ALL}")
    print(f"  Avg P/L per trade   : {Fore.CYAN}{pct(summary['avg_pnl'])}{Style.RESET_ALL}")
    print(f"  Sharpe (trade)      : {Fore.CYAN}{summary['sharpe']:.2f}{Style.RESET_ALL}")
    print(f"  Max drawdown        : {Fore.CYAN}{pct(summary['max_dd'])}{Style.RESET_ALL}")
    print(f"  Time-in-market      : {Fore.CYAN}{pct(summary['tim'])}{Style.RESET_ALL}")

    # (CSV logging stays the same as your original implementation)

def prepare_features_intraday(
    df_30m: pd.DataFrame
) -> pd.DataFrame:
    """
    30-minute feature set with its own Anchored VWAP_30m plus daily context.
    """
    df_30m = df_30m.copy()
    df_30m = compute_indicators(df_30m, timeframe='intraday')
    df_30m['AnchoredVWAP_30m'] = compute_anchored_vwap(df_30m, lookback_bars=200)

    if df_30m.empty or 'Close' not in df_30m.columns:
        return pd.DataFrame()

    daily = to_daily(df_30m, "intraday")
    daily = compute_indicators(daily, timeframe='daily')
    daily['AnchoredVWAP'] = compute_anchored_vwap(daily, lookback_bars=252)

    # drop tz for safe join
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    if df_30m.index.tz is not None:
        df_30m.index = df_30m.index.tz_localize(None)

    df_30m = df_30m.join(daily.reindex(df_30m.index, method='ffill'), rsuffix='_daily')

    # ---------- triple-barrier labelling on 30-min bars --------------
    horizon_bars = 16
    atr_col = 'ATR_intraday'
    if atr_col not in df_30m.columns:
        df_30m[atr_col] = df_30m['Close'].rolling(14).std().fillna(0)

    up = df_30m['Close'] + 2 * df_30m[atr_col]
    dn = df_30m['Close'] - 2 * df_30m[atr_col]
    cls = df_30m['Close'].values
    lbl = np.ones(len(df_30m), int)

    for i in range(len(df_30m) - horizon_bars):
        win = cls[i+1 : i+1+horizon_bars]
        a   = np.where(win >= up.iloc[i])[0]
        b   = np.where(win <= dn.iloc[i])[0]
        if a.size and b.size:
            lbl[i] = 2 if a[0] < b[0] else 0
        elif a.size:
            lbl[i] = 2
        elif b.size:
            lbl[i] = 0

    df_30m['future_class'] = lbl
    df_30m = df_30m.iloc[:-horizon_bars]

    return df_30m

def backtest_strategy_intraday(ticker, start_date, end_date, macro_df: pd.DataFrame | None = None,
                               log_file=None):
    """
    30-minute intraday back-test with Sharpe, max-DD, time-in-market and ATR guard.
    """
    _, df_30m, _, _, _ = fetch_data(ticker, start=start_date, end=end_date)

    feat = prepare_features_intraday(df_30m)
    feat = refine_features(feat)
    model, _ = tune_threshold_and_train(feat)
    if model is None:
        print(Fore.YELLOW + f"Model training failed for {ticker}." + Style.RESET_ALL)
        return

    X_all   = feat.drop(columns=['future_class']).ffill().bfill()
    preds   = model.predict(X_all)
    probas  = model.predict_proba(X_all)
    df_pred = feat.copy()
    df_pred['prediction'] = preds

    trades = []
    in_trade = False
    trade_dir = None
    entry_price = stop_price = target_price = None
    entry_time  = None

    for i in range(len(df_pred)):
        ts        = df_pred.index[i]
        row       = df_pred.iloc[i]
        price     = row['Close']
        atr       = row.get('ATR_intraday', np.nan)
        if np.isnan(atr) or atr == 0:
            raise ValueError(f"[{ticker}] ATR NaN/zero at {ts}")

        if in_trade:
            if trade_dir == "LONG":
                if price >= target_price or price <= stop_price:
                    pnl = (price - entry_price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time, 'exit_timestamp': ts,
                        'direction': 'LONG',
                        'entry_price': round(entry_price, 2),
                        'exit_price':  round(price, 2),
                        'pnl_pct':     round(pnl, 4),
                        'stop_price':  round(stop_price, 2),
                        'target_price':round(target_price, 2)
                    })
                    in_trade = False
            else:  # SHORT
                if price <= target_price or price >= stop_price:
                    pnl = (entry_price - price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time, 'exit_timestamp': ts,
                        'direction': 'SHORT',
                        'entry_price': round(entry_price, 2),
                        'exit_price':  round(price, 2),
                        'pnl_pct':     round(pnl, 4),
                        'stop_price':  round(stop_price, 2),
                        'target_price':round(target_price, 2)
                    })
                    in_trade = False
            if in_trade:
                continue

        # flat – potential new entry
        class_idx  = int(row['prediction'])
        prob_score = probas[i][class_idx]
        if prob_score < 0.60:
            continue

        if class_idx == 2:      # LONG
            in_trade   = True
            trade_dir  = "LONG"
            entry_time = ts
            entry_price= price
            stop_price = price - atr
            target_price = price + 2 * atr
        elif class_idx == 0:    # SHORT
            in_trade   = True
            trade_dir  = "SHORT"
            entry_time = ts
            entry_price= price
            stop_price = price + atr
            target_price = price - 2 * atr

    if not trades:
        print(Fore.YELLOW + f"No trades for {ticker} intraday." + Style.RESET_ALL)
        return

    trades_df = pd.DataFrame(trades)
    summary   = _summarise_performance(trades_df, len(feat))

    pct = lambda x: f"{x*100:.2f}%"
    print(Fore.BLUE + f"\nBacktest Results for {ticker}" + Style.RESET_ALL +
          f" ({start_date} → {end_date}):")
    print(f"  Total trades        : {Fore.CYAN}{summary['total']}{Style.RESET_ALL}")
    print(f"  Win rate            : {Fore.CYAN}{pct(summary['win_rate'])}{Style.RESET_ALL}")
    print(f"  Avg P/L per trade   : {Fore.CYAN}{pct(summary['avg_pnl'])}{Style.RESET_ALL}")
    print(f"  Sharpe (trade)      : {Fore.CYAN}{summary['sharpe']:.2f}{Style.RESET_ALL}")
    print(f"  Max drawdown        : {Fore.CYAN}{pct(summary['max_dd'])}{Style.RESET_ALL}")
    print(f"  Time-in-market      : {Fore.CYAN}{pct(summary['tim'])}{Style.RESET_ALL}")

    # (CSV logging – keep your original code here if you log trades)

def display_main_menu():
    print("\nMain Menu:")
    print("1. Manage Watchlist")
    print("2. Run Signals on Watchlist (Live Mode)")
    print("3. Backtest All Watchlist Tickers")
    print("0. Exit")

def run_signals_on_watchlist(use_intraday=True):
    """
    Fetches watchlist tickers and generates "live" signals for each ticker,
    using either intraday (30m) or daily-lumped features.
    """
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty. Add tickers first.")
        return

    intervals_needed = None
    if use_intraday:
        intervals_needed = ['30m'] # Only fetch 30m for intraday mode

    # Batch fetch data for all tickers
    all_ticker_data = batch_fetch_data_for_tickers(
        tickers,
        requested_intervals=intervals_needed
        # No date overrides needed here as we use DEFAULT_INTERVALS_CONFIG periods
    )

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live) ===")
        ticker_data = all_ticker_data.get(ticker, {})

        if use_intraday:
            df_30m = ticker_data.get('30m', pd.DataFrame())
            # Other df_ an be omitted or set to empty if prepare_features_intraday doesn't need them
            features_df = prepare_features_intraday(df_30m)
        else:
            df_5m  = ticker_data.get('5m', pd.DataFrame())
            df_30m = ticker_data.get('30m', pd.DataFrame())
            df_1h  = ticker_data.get('1h', pd.DataFrame())
            df_90m = ticker_data.get('90m', pd.DataFrame())
            df_1d  = ticker_data.get('1d', pd.DataFrame())
            features_df = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d)

        features_df = refine_features(features_df)
        if features_df.empty or 'future_class' not in features_df.columns:
            print("No valid feature rows.")
            continue

        model, best_thr = tune_threshold_and_train(features_df)
        if model is None:
            print("Model training failed.")
            continue

        # Generate final signal
        latest_row = features_df.drop(columns='future_class').iloc[-1]

        print(generate_signal_output(ticker, latest_row, model, best_thr))

def backtest_watchlist():
    """Runs your existing backtest logic on each watchlist ticker."""
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty. Add tickers first.")
        return

    start_arg = input("Enter backtest start date (YYYY-MM-DD): ").strip()
    end_arg   = input("Enter backtest end date (YYYY-MM-DD): ").strip()
    if not start_arg or not end_arg:
        print("Invalid date range.")
        return

    for ticker in tickers:
        print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
        backtest_strategy(ticker, start_arg, end_arg, macro_df=pd.DataFrame())
        # Or if you want to log to a file, pass `log_file=...`

def show_signals_for_current_week() -> None:
    """
    Prints one actionable signal per symbol until it is closed.  Duplicate
    same-direction signals during the week are suppressed.  Opposite-direction
    signals close the old position and open a new one.  Skips rows where ATR
    is missing/zero.
    """
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty. Add tickers first.")
        return

    today   = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    monday  = today - datetime.timedelta(days=7) if today.weekday() == 0 \
             else today - datetime.timedelta(days=today.weekday())
    lookback = 180
    train_start = monday - datetime.timedelta(days=lookback)
    start_s, end_s = train_start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')

    # Batch fetch data for all tickers, using start_s and end_s for the '1d' interval
    all_ticker_data = batch_fetch_data_for_tickers(
        tickers,
        requested_intervals=None, # Fetch all default intervals
        start_date_override_1d=start_s,
        end_date_override_1d=end_s
    )

    cache            = load_predictions()
    open_by_symbol   = {p['symbol']: p for p in cache if p['status'] == 'Open'}

    for ticker in tickers:
        print(f"\n=== {ticker}: Signals {monday:%Y-%m-%d} → {today:%Y-%m-%d} ===")
        ticker_data = all_ticker_data.get(ticker, {})
        df_5m  = ticker_data.get('5m', pd.DataFrame())
        df_30m = ticker_data.get('30m', pd.DataFrame())
        df_1h  = ticker_data.get('1h', pd.DataFrame())
        df_90m = ticker_data.get('90m', pd.DataFrame())
        df_1d  = ticker_data.get('1d', pd.DataFrame())

        # Error handling for missing data can be done here if necessary,
        # e.g. if df_1d is empty, continue

        feats_all = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, drop_recent=False)
        if feats_all.empty:
            print("No data.")
            continue

        train_df = refine_features(feats_all.dropna(subset=['future_class']))
        if train_df.empty or 'future_class' not in train_df.columns:
            print("No labelled features.")
            continue

        model, thr = tune_threshold_and_train(train_df)
        if model is None:
            continue

        feats_cols = [c for c in train_df.columns if c != 'future_class']
        week_feat  = feats_all.loc[monday:today][feats_cols].ffill().bfill()

        # already in an open trade?
        open_dir = open_by_symbol.get(ticker, {}).get('direction')

        for dt, row in week_feat.iterrows():
            sig = generate_signal_output(ticker, row, model, thr)
            if sig is None:
                continue

            direction = "LONG" if "LONG" in sig else "SHORT"

            # --- suppress duplicate same-direction signals ---------------
            if open_dir == direction:
                break   # skip rest of week for this ticker

            print(f"{dt.date()}: {sig}")

            price = float(row['Close'])
            atr   = float(row.get('ATR_daily', 0))
            stop  = price - atr if direction == 'LONG' else price + atr
            target= price + 2*atr if direction == 'LONG' else price - 2*atr

            new_rec = {
                "symbol": ticker,
                "entry_date": dt.strftime("%Y-%m-%d"),
                "entry_price": round(price, 2),
                "direction": direction,
                "stop_loss": round(stop, 2),
                "profit_target": round(target, 2),
                "status": "Open"
            }

            if open_dir:                       # opposite side → close old
                old = open_by_symbol[ticker]
                old.update({
                    "status": "Closed",
                    "exit_date": dt.strftime("%Y-%m-%d"),
                    "exit_price": price
                })

            open_by_symbol[ticker] = new_rec
            open_dir = direction
            break      # after first actionable signal, stop for week

    # ---------- persist cache -------------------------------------------
    closed = [p for p in cache if p['status'] != 'Open']
    save_predictions(closed + list(open_by_symbol.values()))

def signals_performance_cli():
    """Dashboard of OPEN trades; uses one batch download to avoid YF rate-limit."""
    from yfinance.exceptions import YFRateLimitError

    all_recs = load_predictions()
    open_tr  = [p for p in all_recs if p['status'] == 'Open']
    if not open_tr:
        print("No open positions. Run option 5 first.")
        input("\nPress Enter to return …"); return

    palette = [
        ('title','white,bold',''), ('headers','light blue,bold',''),
        ('positive','dark green',''), ('negative','dark red',''),
        ('hit','white','dark cyan'), ('footer','white,bold','')
    ]

    header = urwid.AttrMap(urwid.Text(" Weekly Signals – Open Trades", 'center'),'title')
    footer = urwid.AttrMap(urwid.Text(" (R)efresh  (D)eject hit trades  (Q)uit "),'footer')
    txt    = urwid.Text("")
    lay    = urwid.Frame(header=header,
                         body=urwid.AttrMap(urwid.Filler(txt,'top'),'body'),
                         footer=footer)

    # ── helper: fetch last close prices in **one** request ──────────────
    def get_prices():
        syms = sorted({p['symbol'] for p in open_tr})
        try:
            df = yf.download(syms, period='1d', auto_adjust=True, progress=False)['Close']
            if isinstance(df, pd.Series): df = df.to_frame(syms[0])  # single sym
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            return df.iloc[-1].to_dict()   # {sym: price}
        except YFRateLimitError:
            print(Fore.YELLOW+"Yahoo rate-limit hit; using previous prices."+Style.RESET_ALL)
            # fallback: use entry_price so P/L = 0
            return {p['symbol']: p['entry_price'] for p in open_tr}

    # ── table builder ──────────────────────────────────────────────────
    # ── table builder ──────────────────────────────────────────────────
    def build_table(prices):
        rows=[('headers',f"{'Symbol':8}{'Dir':6}{'Entry':>10}{'Now':>10}{'P/L%':>8}"
                          f"{'Stop':>10}{'Target':>10}{'Status':>12}{'Date':>12}\n")]
        rec_info=[]
        today=datetime.datetime.today().strftime('%Y-%m-%d')

        for p in open_tr:
            sym   = p['symbol']
            now   = float(prices.get(sym, p['entry_price']))

            # -------- correct P/L sign for SHORT vs LONG ----------------
            if p['direction'] == 'LONG':
                pnl_pct = (now - p['entry_price']) / p['entry_price'] * 100
            else:  # SHORT: profit if price DOWN
                pnl_pct = (p['entry_price'] - now) / p['entry_price'] * 100
            # ----------------------------------------------------------------

            status, hit = "Open", False
            if p['direction']=='LONG':
                if now <= p['stop_loss']:     status, hit = "Stop", True
                elif now >= p['profit_target']: status, hit = "Target", True
            else:  # SHORT
                if now >= p['stop_loss']:     status, hit = "Stop", True
                elif now <= p['profit_target']: status, hit = "Target", True

            attr = 'hit' if hit else ('positive' if pnl_pct >= 0 else 'negative')
            rows.append((attr,
                f"{sym:8}{p['direction']:6}{p['entry_price']:>10.2f}"
                f"{now:>10.2f}{pnl_pct:>8.2f}%{p['stop_loss']:>10.2f}"
                f"{p['profit_target']:>10.2f}{status:>12}{p['entry_date']:>12}\n"))

            rec_info.append((p, hit, status, now, pnl_pct, today))
        return rows, rec_info


    # ── first render ───────────────────────────────────────────────────
    rec_cache=[]
    def refresh(*_):
        nonlocal rec_cache
        prices = get_prices()
        lines, rec_cache = build_table(prices)
        txt.set_text(lines)

    refresh()

    # ── key-handler ────────────────────────────────────────────────────
    def unhandled(k):
        nonlocal open_tr, all_recs
        if k.lower()=='q': raise urwid.ExitMainLoop()
        if k.lower()=='r': refresh()
        if k.lower()=='d':
            changed=False
            for p,hit,stat,now,pnl,today in rec_cache:
                if hit:
                    p.update({'status':stat,'exit_price':round(now,2),
                              'exit_date':today,'pnl_pct':round(pnl,2)})
                    changed=True
            if changed:
                open_tr=[p for p in open_tr if p['status']=='Open']
                all_recs=[p for p in all_recs if p['status']!='Open']+open_tr
                save_predictions(all_recs)
            refresh()

    urwid.MainLoop(lay, palette, unhandled_input=unhandled).run()


def closed_stats_cli():
    """
    Color-enhanced statistics for CLOSED positions in weekly_signals.json.
    Green  = favourable numbers, Red = unfavourable.
    """
    recs = [p for p in load_predictions() if p['status'] != 'Open']
    if not recs:
        print("No closed trades recorded.")
        input("\nPress Enter to return …")
        return

    # ── compute P/L % for every record (in case older JSON lacks it) ──
    for r in recs:
        if 'pnl_pct' not in r or r['pnl_pct'] is None:
            ep, xp = r['entry_price'], r.get('exit_price', r['entry_price'])
            if r['direction'] == 'LONG':
                r['pnl_pct'] = (xp - ep) / ep * 100
            else:                              # SHORT
                r['pnl_pct'] = (ep - xp) / ep * 100
            r['pnl_pct'] = round(r['pnl_pct'], 2)

    wins   = [r for r in recs if r['status'] == 'Target']
    losses = [r for r in recs if r['status'] in ('Stop', 'Closed')]
    total  = len(recs)
    win_rt = len(wins) / total * 100

    avg_win = np.mean([w['pnl_pct'] for w in wins])   if wins   else 0.0
    avg_los = np.mean([l['pnl_pct'] for l in losses]) if losses else 0.0

    # compounded equity curve
    equity = 1.0
    for r in recs: equity *= 1 + r['pnl_pct'] / 100
    tot_ret = (equity - 1) * 100

    # ── colour helpers ──
    g = lambda s: Fore.GREEN + s + Style.RESET_ALL
    r = lambda s: Fore.RED   + s + Style.RESET_ALL
    b = lambda s: Fore.CYAN  + s + Style.RESET_ALL   # headings

    print(b("\nClosed-Position Statistics"))
    print(b("--------------------------------"))

    print(f"Total closed trades   : {b(str(total))}")
    print(f"Wins (hit target)     : {g(str(len(wins)) if wins else '0')}"
          f"  |  Avg gain : {g(f'{avg_win:+.2f}%') if wins else '--'}")
    print(f"Losses / switches     : {r(str(len(losses)) if losses else '0')}"
          f"  |  Avg loss : {r(f'{avg_los:+.2f}%') if losses else '--'}")

    win_color = g if win_rt >= 50 else r
    print(f"Win rate              : {win_color(f'{win_rt:.2f}%')}")

    tot_color = g if tot_ret >= 0 else r
    print(f"Compounded return     : {tot_color(f'{tot_ret:+.2f}%')}")

    # ── last 10 rows table ──
    print(b("\nMost recent 10 closed trades:"))
    for rcd in recs[-10:]:
        pl_col = g if rcd['pnl_pct'] >= 0 else r
        print(f"{rcd['exit_date']}  {rcd['symbol']:5}  {rcd['direction']:5} "
              f"{rcd['status']:6}  PnL {pl_col(f'{rcd['pnl_pct']:+6.2f}%')}")

    # ── clear option ──
    if input(Fore.YELLOW + "\n(C)lear stats or Enter to return: " + Style.RESET_ALL).lower() == 'c':
        save_predictions([p for p in load_predictions() if p['status'] == 'Open'])
        print(Fore.YELLOW + "History cleared." + Style.RESET_ALL)
        input("\nPress Enter to return …")



def interactive_menu():
    while True:
        print("\nMain Menu:")
        print("1. Manage Watchlist")
        print("2. Run Signals on Watchlist (Live Mode)")
        print("3. Backtest All Watchlist Tickers")
        print("4. Schedule Signals (9:30 to 16:00 EST via Cron)")
        print("5. Show This Weeks Signals")
        print("6. Show Latest Signals Performance")
        print("7. Closed-Trades Statistics")        # NEW
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            manage_watchlist()
        elif choice == '2':
            run_signals_on_watchlist()
        elif choice == '3':
            backtest_watchlist()
        elif choice == '4':
            schedule_signals_instructions()
        elif choice == '5':
            show_signals_for_current_week()
        elif choice == '6':
            signals_performance_cli()
        elif choice == '7':
            closed_stats_cli()
        else:
            print("Invalid option. Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Swing trading signal generator/backtester")
    parser.add_argument('tickers', nargs='*', help="List of stock ticker symbols to analyze")
    parser.add_argument('--log', action='store_true', help="Optionally log selected trades to a file")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode instead of live signal mode")
    parser.add_argument('--start', default=None, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument('--real', action='store_true', help="Use 30-min intraday backtest for max realism")
    parser.add_argument('--live-real', action='store_true',
                        help="Use 30‑min intraday pipeline in live mode")
    args = parser.parse_args()

    # If no positional tickers and no flags set, go interactive:
    if not args.tickers and not any(vars(args).values()):
        interactive_menu()
        return

    tickers = args.tickers
    log_trades = args.log
    run_backtest = args.backtest
    start_arg = args.start
    end_arg = args.end
    use_intraday = args.real

    # Open log file if needed
    log_file = None
    if log_trades:
        log_file = open("trades_log.csv", "a")

    # ---------------------------------------------------------------------
    # 1) If in BACKTEST mode, we handle either intraday or daily-lumped
    # ---------------------------------------------------------------------
    if run_backtest and start_arg and end_arg:
        # (a) Intraday approach if -real is set
        if use_intraday:
            for ticker in tickers:
                log_filename = f"{ticker}_30m_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Intraday Backtesting {ticker} (30m) from {start_arg} to {end_arg} ===")
                    backtest_strategy_intraday(ticker, start_arg, end_arg, macro_df=pd.DataFrame(), log_file=lf)

        # (b) Otherwise daily-lumped approach
        else:
            for ticker in tickers:
                log_filename = f"{ticker}_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
                    backtest_strategy(ticker, start_arg, end_arg, macro_df=pd.DataFrame(), log_file=lf)

        # Once backtest is done, we can safely close the main log file (if any) and return
        if log_file:
            log_file.close()
        return

    # ---------------------------------------------------------------------
    # 2) Otherwise do the "LIVE" signal generation mode
    # ---------------------------------------------------------------------
    # (No start/end provided or no --backtest)
    # Batch fetch data for all tickers in live mode
    all_ticker_data = batch_fetch_data_for_tickers(
        tickers,
        requested_intervals=None # Fetch all default intervals
    )

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live signal mode) ===")
        ticker_data = all_ticker_data.get(ticker, {})
        df_5m  = ticker_data.get('5m', pd.DataFrame())
        df_30m = ticker_data.get('30m', pd.DataFrame())
        df_1h  = ticker_data.get('1h', pd.DataFrame())
        df_90m = ticker_data.get('90m', pd.DataFrame())
        df_1d  = ticker_data.get('1d', pd.DataFrame())

        if df_1d.empty or 'Close' not in df_1d.columns: # Still check df_1d as it's crucial
            print(f"No usable daily data for {ticker}, skipping.")
            continue

        features_df = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d)
        if features_df.empty:
            print(f"Insufficient data for {ticker}, skipping.")
            continue

        features_df = refine_features(features_df)
        if features_df.empty or 'future_class' not in features_df.columns:
            continue

        model, best_thr = tune_threshold_and_train(features_df)
        if model is None:
            print(f"Model training failed or no valid data for {ticker}.")
            continue

        # Generate signal using the last row
        latest_features = features_df.iloc[-1]
        X_latest = latest_features.drop(labels='future_class', errors='ignore').to_frame().T
        X_latest.ffill(inplace=True)
        X_latest.bfill(inplace=True)
        latest_series = X_latest.iloc[0]

        signal_output = generate_signal_output(ticker, latest_series, model, best_thr)
        print(signal_output)

        # Optionally log this single signal
        if log_file and ("LONG" in signal_output or "SHORT" in signal_output):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{now_str},{ticker},{signal_output}\n")
            log_file.flush()

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
