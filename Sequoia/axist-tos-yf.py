#!/usr/bin/env python3


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
from ta.trend import CCIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from ta.trend import ADXIndicator, CCIIndicator, MACD, EMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import MFIIndicator  # optional if you want more volume-based features
from ta.volatility import DonchianChannel  # optional, if needed
from colorama import init, Fore, Style
init(autoreset=True)
from fredapi import Fred 

import configparser
import warnings
from pandas.errors import PerformanceWarning      # already triggered by pandas
warnings.filterwarnings(
    "ignore",
    message=r"The behavior of array concatenation with empty entries is deprecated",
    category=FutureWarning,
)

_tf_map = {
    "5m":  "5m",
    "30m": "30m",
    "1h":  "1h",
    "90m": "90m", # yfinance supports 90m
    "1d":  "1d",
}

PREDICTIONS_FILE = "weekly_signals.json"
YF_CALLS = 0          # total HTTPS requests sent to Yahoo Finance
DATA_CACHE: dict[tuple, tuple] = {}      # key → (df_5m, df_30m, df_1h, df_90m, df_1d)

def load_config(section: str | None = None) -> dict | str | None:
    cfg_path = Path(__file__).with_name("config.ini")

    # read() with encoding="utf-8-sig" automatically strips the BOM
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8-sig")

    if section is None:
        # default: just hand back the FRED key (or None if missing)
        return cfg["FRED"].get("api_key") if cfg.has_section("FRED") else None

    # missing section ⇒ empty dict keeps callers happy
    if not cfg.has_section(section):
        return {}

    # return all options in that section as a plain dict
    return {k: v for k, v in cfg.items(section)}

# Alpaca client related lines removed.

def yf_download_wrapper(tickers, period=None, interval="1d", start=None, end=None, progress=False) -> pd.DataFrame:
    """
    A wrapper for yf.download that handles common tasks like:
    - Ensuring Open, High, Low, Close, Volume columns.
    - Making the index timezone-naive.
    - Handling empty responses.
    """
    global YF_CALLS
    YF_CALLS += 1
    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            start=start,
            end=end,
            progress=progress,
            auto_adjust=False, # Keep True if you prefer adjusted prices directly
            threads=True # Use threads for multiple tickers
        )
        if df.empty:
            # For single ticker, yf.download returns empty df with no columns
            # For multiple tickers, it might have structure but all NaNs
            # We ensure a consistent empty DataFrame format
            if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
                 return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            # For multiple tickers, yfinance might return a multi-index frame
            # If all data is NaN (e.g. bad tickers), it could be empty in content
            # It's safer to let it return its structure and let caller handle it if multi-indexed
            # or check if all values are NaN. For now, if df.empty is true, assume this check is sufficient.

        # Ensure standard column names (yfinance usually provides them like this)
        # If auto_adjust=False, columns are 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
        # If auto_adjust=True, columns are 'Open', 'High', 'Low', 'Close', 'Volume'
        # We will select specific columns to ensure consistency
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                # For multi-ticker download, columns are like [('Open', 'AAPL'), ('Close', 'AAPL'), ...]
                # We want to ensure the top level has the OHLCV names
                # This is generally fine, but we'll ensure the final df passed around
                # in single-ticker functions has simple column names.
                # safe_download will return this multi-index if multiple tickers are requested.
                # Other functions like _download_cached typically deal with single tickers.
                pass # Let it be multi-indexed for now.
            else: # Single ticker
                df = df[["Open", "High", "Low", "Close", "Volume"]]


            # Make index timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"yfinance download error for {tickers} ({interval}, {period}, {start}-{end}): {e}")
        # Return an empty DataFrame with standard columns in case of error
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        else: # For multiple tickers, an empty multi-index df might be more complex to create here
              # For now, return simple empty df, or consider raising to let caller handle
            return pd.DataFrame()


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

def _get_latest_price_yf(symbol: str) -> float | None:
    """
    Fetches the latest available closing price for a symbol using yfinance.
    Tries 1-minute data for the current day first, then falls back to the last daily close.
    """
    try:
        # Try to get the latest 1-minute data for today
        # Fetches data for the last day, with 1m interval
        data = yf.download(tickers=symbol, period="1d", interval="1m", progress=False, auto_adjust=False)
        if not data.empty and 'Close' in data.columns:
            return float(data['Close'].iloc[-1])
        else:
            # Fallback: try to get the last daily close from the last 2 days
            data_daily = yf.download(tickers=symbol, period="2d", interval="1d", progress=False, auto_adjust=False)
            if not data_daily.empty and 'Close' in data_daily.columns:
                return float(data_daily['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"Error fetching latest price for {symbol} with yfinance: {e}", file=sys.stderr)
        return None

def _update_positions_status() -> None:
    """
    Refresh open-trade status using the latest price from yfinance.
    """
    preds = load_predictions()
    open_pos = [p for p in preds if p.get("status") == "Open"]
    if not open_pos:
        return

    changed = False
    today_iso = datetime.date.today().isoformat()

    for rec in open_pos:
        sym = rec["symbol"]
        latest_price = _get_latest_price_yf(sym)

        if latest_price is None:
            print(f"Could not fetch latest price for {sym} in _update_positions_status. Skipping update for this position.", file=sys.stderr)
            continue # Use last known price or skip? Original used entry_price as fallback in Alpaca version. Here we skip if no price.

        now_price = latest_price # If we fetched successfully

        hit_stop = False
        hit_target = False

        if rec["direction"] == "LONG":
            if now_price <= rec["stop_loss"]:
                hit_stop = True
            elif now_price >= rec["profit_target"]:
                hit_target = True
        elif rec["direction"] == "SHORT": # Assuming "SHORT"
            if now_price >= rec["stop_loss"]:
                hit_stop = True
            elif now_price <= rec["profit_target"]:
                hit_target = True

        if hit_stop or hit_target:
            rec["status"]     = "Stop" if hit_stop else "Target"
            rec["exit_date"]  = today_iso
            rec["exit_price"] = round(now_price, 2)
            changed = True
            print(f"Position updated: {sym} hit {rec['status']} at {now_price}")

    if changed:
        save_predictions(preds)

def safe_download(tickers, *_, period=None, interval="1d",
                  start=None, end=None, **__):
    """
    Thin wrapper so existing calls (period/interval OR start/end) still work.
    Uses yf_download_wrapper for data fetching.
    """
    # yfinance's yf.download can handle single strings or lists of strings for tickers.
    # It also manages period/interval and start/end logic internally.
    
    # Determine if 'interval' from _tf_map needs to be used or if it's already a yf string
    tf = _tf_map.get(interval, interval) # Use mapped value if available, else assume interval is yf compatible

    data = yf_download_wrapper(
        tickers=tickers,
        period=period,
        interval=tf,
        start=start,
        end=end,
        progress=False # Usually set to False for automated scripts
    )
    
    # yf.download returns a DataFrame.
    # If multiple tickers are given, it's a multi-index DataFrame.
    # This is handled by preload_interval_cache, so no change needed here for that.
    # For single ticker, it's a simple DataFrame.
    # yf_download_wrapper already handles timezone localization and column selection.
    return data


def _download_cached(ticker: str,
                     period: str,
                     interval: str,
                     cache_dir: str = ".ohlcv_cache") -> pd.DataFrame:
    """
    Exactly the same disk-cache logic as before, but the network fetch
    now goes through yfinance instead of Alpaca.
    """
    Path(cache_dir).mkdir(exist_ok=True)
    fname = Path(cache_dir) / f"{ticker}_{period}_{interval}.pkl"

    sec_per_bar = {"5m":300,"30m":1800,"1h":3600,"90m":5400,"1d":86400}
    factor  = 48 if interval != "1d" else 6
    max_age = sec_per_bar.get(interval,3600)*factor

    if fname.exists():
        try:
            cached: pd.DataFrame = joblib.load(fname)
            if not cached.empty:
                age = (pd.Timestamp.utcnow() - cached.index[-1]).total_seconds()
                if age < max_age:
                    return cached
        except Exception:
            fname.unlink(missing_ok=True)

    
    tf_interval = _tf_map.get(interval, interval) # Get yfinance compatible interval string

    # yfinance typically uses 'period' OR 'start'/'end'.
    # The original alpaca_download had a 'limit' parameter.
    # yf.download doesn't have a direct 'limit' for number of bars with 'period'.
    # If 'period' is given (e.g., "60d"), yf.download fetches data for that duration.
    # If 'limit' was essential, one might fetch more data and then df.tail(limit).
    # However, looking at fetch_data, limit=None is passed when period is used.
    fresh = yf_download_wrapper(
        tickers=ticker, # _download_cached is for single ticker
        period=period,  # yfinance uses period e.g. "60d"
        interval=tf_interval, # yfinance uses interval e.g. "30m"
        # start and end are not typically used with period in yf.download
    )

    if fname.exists():
        try:
            old = joblib.load(fname)
            fresh = pd.concat([old, fresh]).sort_index().drop_duplicates()
        except Exception:
            pass

    joblib.dump(fresh, fname)
    return fresh

def preload_interval_cache(symbols: list[str],
                           period: str,
                           interval: str,
                           cache_dir: str = ".ohlcv_cache") -> None:
    symbols = sorted(set(symbols))
    if not symbols:
        return

    # one bulk call
    # auto_adjust, progress, threads are handled by yf_download_wrapper or safe_download
    bulk = safe_download(symbols, period=period, interval=interval)
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
        # Timezone localization is now handled by yf_download_wrapper
        # if df.index.tz is not None:
        #     df.index = df.index.tz_localize(None)
        joblib.dump(df,
                    Path(cache_dir) / f"{sym}_{period}_{interval}.pkl")


def fetch_data(ticker, start=None, end=None, intervals=None, warmup_days=300):
    """
    Returns (df_5m, df_30m, df_1h, df_90m, df_1d) – now entirely from yfinance.
    """
    if intervals is None:
        intervals = {
            "5m":  ("14d",  "5m"),
            "30m": ("60d",  "30m"),
            "1h":  ("120d", "1h"),
            "90m": ("60d",  "90m"),
            "1d":  ("380d", "1d"),
        }

    # intraday (disk-cached)
    df_5m  = _download_cached(ticker, *intervals["5m"])
    df_30m = _download_cached(ticker, *intervals["30m"])
    df_1h  = _download_cached(ticker, *intervals["1h"])
    df_90m = _download_cached(ticker, *intervals["90m"])

    # daily
    if start and end:
        real_start = (pd.to_datetime(start) -
                      datetime.timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        df_1d = yf_download_wrapper(
            tickers=ticker,
            interval="1d", # Fetched as daily
            start=real_start,
            end=end
        )
    else:
        # Here intervals["1d"] is like ("380d", "1d")
        # So period = "380d", interval = "1d"
        period_val, interval_val = intervals["1d"]
        df_1d = _download_cached(ticker, period_val, interval_val) # Use mapped interval if necessary

    # clean helper (modified to handle potential MultiIndex columns for single tickers)
    def _clean(df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
        df.dropna(how="all", inplace=True)
        if df.empty:
            return df

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if isinstance(df.columns, pd.MultiIndex):
            if ticker_symbol in df.columns.get_level_values(1):
                df = df.xs(ticker_symbol, axis=1, level=1)
            # Add other multi-index handling logic if necessary, but keep it concise for now
            # Fallback or warning if structure is not as expected:
            elif not all(isinstance(col, str) for col in df.columns): # if not already flattened
                 print(f"[WARN] In _clean for {ticker_symbol}, could not fully flatten MultiIndex columns: {df.columns.tolist()}", file=sys.stderr)


        # Ensure 'Close' column from 'Adj Close' if necessary, and OHLCV presence
        if not df.empty:
            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
            
            # It's better if yf_download_wrapper strictly enforces OHLCV presence.
            # This check is a fallback.
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                 print(f"[WARN] In _clean for {ticker_symbol}, DataFrame is missing expected columns: {missing_cols}. Current columns: {df.columns.tolist()}", file=sys.stderr)
        return df

    # Pass ticker to _clean for each DataFrame
    cleaned_dfs = []
    for frame in (df_5m, df_30m, df_1h, df_90m, df_1d):
        cleaned_dfs.append(_clean(frame, ticker)) # ticker is available in fetch_data's scope
    
    return tuple(cleaned_dfs)


def compute_indicators(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """
    Compute the core indicator set *plus* the exact “Sequoia” flags and
    previous-day high/low needed for stop/target logic.
    """
    df = df.copy()                       # avoid SettingWithCopyWarning
    req = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not req.issubset(df.columns):
        return df

    window = 14
    # ===== standard indicators (unchanged) ==========================
    rsi  = RSIIndicator(close=df["Close"], window=window).rsi()
    adx  = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=window)
    stoch= StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"],
                                window=window, smooth_window=3)
    cci  = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"],
                        window=20, constant=0.015)
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    bb   = BollingerBands(close=df["Close"], window=20, window_dev=2)
    atr  = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"],
                            window=window)

    df[f"RSI_{timeframe}"]          = rsi
    df[f"ADX_{timeframe}"]          = adx.adx()
    df[f"ADX_pos_{timeframe}"]      = adx.adx_pos()
    df[f"ADX_neg_{timeframe}"]      = adx.adx_neg()
    df[f"STOCHk_{timeframe}"]       = stoch.stoch()
    df[f"STOCHd_{timeframe}"]       = stoch.stoch_signal()
    df[f"CCI_{timeframe}"]          = cci.cci()
    df[f"MACD_{timeframe}"]         = macd.macd()
    df[f"MACD_signal_{timeframe}"]  = macd.macd_signal()
    df[f"MACD_hist_{timeframe}"]    = macd.macd_diff()
    df[f"BB_upper_{timeframe}"]     = bb.bollinger_hband()
    df[f"BB_lower_{timeframe}"]     = bb.bollinger_lband()
    df[f"BB_middle_{timeframe}"]    = bb.bollinger_mavg()
    df[f"ATR_{timeframe}"]          = atr.average_true_range()

    if timeframe in {"daily", "hourly"}:
        df[f"EMA50_{timeframe}"]  = EMAIndicator(close=df["Close"], window=50).ema_indicator()
        df[f"EMA200_{timeframe}"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()

    # ===== extra Sequoia pieces (only once on DAILY bars) ===========
    if timeframe == "daily":
        # --- Hull MA & slope ---------------------------------------
        def _hull(close, n):
            wma_half = WMAIndicator(close, window=n//2).wma()
            wma_root = WMAIndicator(close, window=int(np.sqrt(n))).wma()
            return 2 * wma_half - wma_root

        hma = _hull(df["Close"], 20)
        df["HMA_daily"]       = hma
        df["HMA_slope_daily"] = hma.diff()

        # --- DI difference -----------------------------------------
        df["DI_diff_daily"] = df["ADX_pos_daily"] - df["ADX_neg_daily"]

        # --- yesterday’s high/low (for stop/target calc) -----------
        df["PrevHigh_daily"] = df["High"].shift(1)
        df["PrevLow_daily"]  = df["Low"].shift(1)

        # --- scanner boolean flags ---------------------------------
        macd_green  = df["MACD_hist_daily"] > 0
        hull_green  = df["HMA_slope_daily"] > 0
        dmi_green   = df["DI_diff_daily"]   > 0
        atr_ok      = df["ATR_daily"]       > 0.50
        vol_ok      = df["Volume"]          > df["Volume"].shift(1) * 0.8
        hollow      = df["Close"] > df["Open"]
        solid       = df["Close"] < df["Open"]
        ushadow_ok  = (df["High"] - df["Close"]) / (df["High"] - df["Low"] + 1e-9) < 0.2
        lshadow_ok  = (df["Close"] - df["Low"])  / (df["High"] - df["Low"] + 1e-9) < 0.2

        df["Sequoia_long"]  = (macd_green & hull_green & dmi_green &
                               hollow & ushadow_ok & vol_ok & atr_ok)
        df["Sequoia_short"] = (~macd_green & ~hull_green & ~dmi_green &
                               solid  & lshadow_ok & vol_ok & atr_ok)

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

    # existing daily vwap
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
    
    # DEBUG prints removed here
    # Explicit check removed as the KeyError was due to MultiIndex, now handled in _clean
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
    Turn the latest feature row into a LONG/SHORT text summary
    – honour Sequoia filters
    – stop 0.5 % beyond yesterday’s low/high
    – profit ≥ max( 2R , 10 % ).
    """
    # -------- probability / class -----------------------------------
    probs      = model.predict_proba(latest_row.to_frame().T.values)[0]
    class_idx  = int(np.argmax(probs))
    prob_score = probs[class_idx]

    if prob_score < 0.60 or class_idx == 1:
        return None                    # neutral/low-confidence

    direction = "LONG" if class_idx == 2 else "SHORT"
    if direction == "LONG" and not latest_row.get("Sequoia_long", False):
        return None
    if direction == "SHORT" and not latest_row.get("Sequoia_short", False):
        return None

    price     = float(latest_row["Close"])
    prev_low  = float(latest_row.get("PrevLow_daily", np.nan))
    prev_high = float(latest_row.get("PrevHigh_daily", np.nan))

    if direction == "LONG":
        stop    = prev_low * 0.995 if not np.isnan(prev_low) else price * 0.94
        risk    = price - stop
        min_trg = price * 1.10
        target  = price + max(2 * risk, min_trg - price)
    else:  # SHORT
        stop    = prev_high * 1.005 if not np.isnan(prev_high) else price * 1.06
        risk    = stop - price
        min_trg = price * 0.90
        target  = price - max(2 * risk, price - min_trg)

    colour = Fore.GREEN if direction == "LONG" else Fore.RED
    return (f"{Fore.CYAN}{ticker}{Style.RESET_ALL}: {colour}{direction}{Style.RESET_ALL} "
            f"@ ${price:.2f}  Stop ${stop:.2f}  Target ${target:.2f}  "
            f"P={prob_score:.2f}")


def _build_entry(row, direction):
    """
    Return (stop_price, target_price) applying the 0.5 % prev-day rule
    and ≥2R / ≥10 % profit logic.  `row` must be a DAILY feature row.
    """
    price = float(row["Close"])
    if direction == "LONG":
        prev_low = float(row.get("PrevLow_daily", np.nan))
        stop     = prev_low * 0.995 if not np.isnan(prev_low) else price * 0.94
        risk     = price - stop
        tgt_abs  = price * 1.10                      # +10 %
        target   = price + max(2 * risk, tgt_abs - price)
    else:  # SHORT
        prev_hi  = float(row.get("PrevHigh_daily", np.nan))
        stop     = prev_hi * 1.005 if not np.isnan(prev_hi) else price * 1.06
        risk     = stop - price
        tgt_abs  = price * 0.90                      # –10 %
        target   = price - max(2 * risk, price - tgt_abs)
    return stop, target


def backtest_strategy(ticker, start_date, end_date, log_file=None):
    """
    Daily-bars swing-trade back-test:
      • uses Sequoia_long / Sequoia_short flags
      • stop = 0.5 % beyond yesterday’s low (long) or high (short)
      • target = max(2 × risk , 10 % absolute)
    """
    # ── data & feature prep ───────────────────────────────────────────
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

    # ── trade simulation ─────────────────────────────────────────────
    trades  = []
    in_pos  = False
    dir_    = None
    entry_price = stop_price = target_price = None
    entry_ts    = None

    for idx, row in df_pred.iterrows():
        price = row["Close"]

        # ---- manage open position ----------------------------------
        if in_pos:
            hit_stop   = (price <= stop_price) if dir_ == "LONG" else (price >= stop_price)
            hit_target = (price >= target_price) if dir_ == "LONG" else (price <= target_price)
            if hit_stop or hit_target:
                pnl = (price - entry_price) / entry_price if dir_ == "LONG" \
                      else (entry_price - price) / entry_price
                trades.append({
                    "entry_timestamp": entry_ts, "exit_timestamp": idx,
                    "direction": dir_,
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(price, 2),
                    "pnl_pct":     round(pnl, 4),
                    "stop_price":  round(stop_price, 2),
                    "target_price":round(target_price, 2)
                })
                in_pos = False
            if in_pos:
                continue            # still in trade – skip new entry logic

        # ---- flat: evaluate new entry ------------------------------
        cls       = int(row["prediction"])
        prob      = probas[df_pred.index.get_loc(idx)][cls]
        if prob < 0.60:
            continue

        if cls == 2 and row.get("Sequoia_long", False):
            dir_ = "LONG"
        elif cls == 0 and row.get("Sequoia_short", False):
            dir_ = "SHORT"
        else:
            continue               # not a valid Sequoia setup

        entry_price            = price
        stop_price, target_price = _build_entry(row, dir_)
        entry_ts               = idx
        in_pos                 = True

    if not trades:
        print(Fore.YELLOW + f"No trades for {ticker}." + Style.RESET_ALL)
        return

    # ── performance summary (unchanged) ─────────────────────────────
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

    # (CSV logging logic – keep your existing code here if needed)


def prepare_features_intraday(
    df_30m: pd.DataFrame | None = None
) -> pd.DataFrame: # Assuming it should always return a DataFrame
    """
    30-minute feature set with its own Anchored VWAP_30m plus daily context.
    """
    if df_30m is None:
        return pd.DataFrame()

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
    return df_30m # Added missing return statement


def backtest_strategy_intraday(ticker, start_date, end_date,  log_file=None):
    """
    30-minute intraday back-test with Sequoia filtering and the same
    stop/target mechanics as the daily strategy.
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

    trades  = []
    in_pos  = False
    dir_    = None
    entry_price = stop_price = target_price = None
    entry_ts    = None

    for idx, row in df_pred.iterrows():
        price = row["Close"]

        # ---- manage open position ----------------------------------
        if in_pos:
            hit_stop   = (price <= stop_price) if dir_ == "LONG" else (price >= stop_price)
            hit_target = (price >= target_price) if dir_ == "LONG" else (price <= target_price)
            if hit_stop or hit_target:
                pnl = (price - entry_price) / entry_price if dir_ == "LONG" \
                      else (entry_price - price) / entry_price
                trades.append({
                    "entry_timestamp": entry_ts, "exit_timestamp": idx,
                    "direction": dir_,
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(price, 2),
                    "pnl_pct":     round(pnl, 4),
                    "stop_price":  round(stop_price, 2),
                    "target_price":round(target_price, 2)
                })
                in_pos = False
            if in_pos:
                continue

        # ---- flat: evaluate new entry ------------------------------
        cls  = int(row["prediction"])
        prob = probas[df_pred.index.get_loc(idx)][cls]
        if prob < 0.60:
            continue

        if cls == 2 and row.get("Sequoia_long", False):
            dir_ = "LONG"
        elif cls == 0 and row.get("Sequoia_short", False):
            dir_ = "SHORT"
        else:
            continue                # not a scanner-approved bar

        entry_price            = price
        stop_price, target_price = _build_entry(row, dir_)
        entry_ts               = idx
        in_pos                 = True

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

    # (CSV logging logic – keep your existing code here if needed)


def run_signals_on_watchlists(use_intraday: bool = True):

    # ------------- one bulk fetch populates the on-disk cache ---------
    all_syms = load_watchlist("long") + load_watchlist("short")
    preload_interval_cache(all_syms, period="60d",  interval="30m")   # intraday bars
    preload_interval_cache(all_syms, period="380d", interval="1d")    # daily bars
    # ------------------------------------------------------------------

    fred_api_key = load_config()
    today        = datetime.datetime.today()


    for side in ("long", "short"):
        tickers = load_watchlist(side)
        if not tickers:
            continue
        print(f"\n==========   {side.upper()} WATCH-LIST   ==========")
        want_dir = "LONG" if side == "long" else "SHORT"

        for ticker in tickers:
            try:
                df_5m, df_30m, df_1h, df_90m, df_1d = get_or_fetch(ticker)
            except Exception as e:
                print(f"{ticker}: data error → {e}")
                continue

            feats = (prepare_features_intraday(df_30m) if use_intraday
                     else prepare_features(df_5m, df_30m, df_1h,
                                           df_90m, df_1d))
            feats = refine_features(feats)
            if feats.empty or 'future_class' not in feats.columns:
                print(f"{ticker}: no usable rows.")
                continue

            model, thr = tune_threshold_and_train(feats)
            if model is None:
                print(f"{ticker}: model training failed.")
                continue

            latest = feats.drop(columns='future_class').iloc[-1]
            sig    = generate_signal_output(ticker, latest, model, thr)

            if sig and (want_dir in sig):
                print(sig)

def show_signals_since_start_of_week() -> None:
    """
    Print every actionable Sequoia signal generated between Monday and today,
    re-using the on-disk cache so that **no new Yahoo requests** are made.
    """
    # 1) Refresh the cache once for every interval (five HTTPS requests total)
    all_syms = load_watchlist("long") + load_watchlist("short")
    for period, interval in [
        ("14d",  "5m"),
        ("60d",  "30m"),
        ("120d", "1h"),
        ("60d",  "90m"),
        ("380d", "1d")
    ]:
        preload_interval_cache(all_syms, period=period, interval=interval)

    # 2) Define date boundaries
    today   = datetime.datetime.today().replace(hour=0, minute=0,
                                                second=0, microsecond=0)
    monday  = today - datetime.timedelta(days=today.weekday())
    lookback_days = 180
    train_start = monday - datetime.timedelta(days=lookback_days)
    start_s, end_s = train_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # 3) Load the existing trade log
    cache            = load_predictions()
    open_by_symbol   = {p["symbol"]: p for p in cache if p["status"] == "Open"}

    # 4) Scan both watch-lists
    for side in ("long", "short"):
        tickers = load_watchlist(side)
        if not tickers:
            continue
        print(f"\n==========   {side.upper()} WEEKLY SIGNALS   ==========")
        want_dir = "LONG" if side == "long" else "SHORT"

        for ticker in tickers:
            print(f"\n--- {ticker} ---")
            try:
                df_5m, df_30m, df_1h, df_90m, df_1d = get_or_fetch(ticker)
            except Exception as e:
                print(f"Fetch error {e}")
                continue

            # Slice the daily frame locally – **no new download**
            df_1d = df_1d.loc[start_s:end_s]

            feats_all = prepare_features(df_5m, df_30m, df_1h,
                                         df_90m, df_1d, drop_recent=False)
            if feats_all.empty:
                print("No data.")
                continue

            train_df = refine_features(feats_all.dropna(subset=["future_class"]))
            if train_df.empty:
                print("No labels.")
                continue

            model, thr = tune_threshold_and_train(train_df)
            if model is None:
                print("Model training failed.")
                continue

            feat_cols = [c for c in train_df.columns if c != "future_class"]
            week_feat = feats_all.loc[monday:today][feat_cols].ffill().bfill()

            open_dir = open_by_symbol.get(ticker, {}).get("direction")

            for dt, row in week_feat.iterrows():
                sig = generate_signal_output(ticker, row, model, thr)
                if not sig or (want_dir not in sig):
                    continue

                # Skip duplicates pointing the same way
                if open_dir == want_dir:
                    break

                print(f"{dt.date()}: {sig}")

                price = float(row["Close"])
                stop, target = _build_entry(row, want_dir)

                new_rec = {
                    "symbol": ticker,
                    "entry_date": dt.strftime("%Y-%m-%d"),
                    "entry_price": round(price, 2),
                    "direction": want_dir,
                    "stop_loss": round(stop, 2),
                    "profit_target": round(target, 2),
                    "status": "Open"
                }

                if open_dir:                          # reverse the old trade
                    old = open_by_symbol[ticker]
                    old.update({
                        "status": "Closed",
                        "exit_date": dt.strftime("%Y-%m-%d"),
                        "exit_price": price
                    })

                open_by_symbol[ticker] = new_rec
                open_dir = want_dir
                break                                 # only first signal per week

    closed = [p for p in cache if p["status"] != "Open"]
    save_predictions(closed + list(open_by_symbol.values()))

def signals_performance_cli():
    """Dashboard of OPEN trades – price refresh via yfinance."""

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

    def get_prices():
        """Fetch the latest price for each open position's symbol using _get_latest_price_yf."""
        result = {}
        symbols_in_open_trades = {p['symbol'] for p in open_tr} # Use open_tr defined in signals_performance_cli

        for sym in symbols_in_open_trades:
            latest_price = _get_latest_price_yf(sym)
            if latest_price is not None:
                result[sym] = latest_price
            else:
                # Fallback to entry price if live price fetch fails
                result[sym] = next(p['entry_price'] for p in open_tr if p['symbol'] == sym)
                print(f"Using entry price as fallback for {sym} in get_prices due to fetch error.", file=sys.stderr)
        return result

    def build_table(prices):
        rows=[('headers',f"{'Symbol':8}{'Dir':6}{'Entry':>10}{'Now':>10}{'P/L%':>8}"
                          f"{'Stop':>10}{'Target':>10}{'Status':>12}{'Date':>12}\n")]
        rec_info=[]
        today=datetime.datetime.today().strftime('%Y-%m-%d')

        for p in open_tr:
            sym   = p['symbol']
            now   = float(prices.get(sym, p['entry_price']))

            # correct P/L sign for SHORT vs LONG
            pnl_pct = ((now - p['entry_price']) if p['direction'] == 'LONG' else
                       (p['entry_price'] - now)) / p['entry_price'] * 100

            status, hit = "Open", False
            if p['direction']=='LONG':
                if now <= p['stop_loss']:        status, hit = "Stop", True
                elif now >= p['profit_target']:  status, hit = "Target", True
            else:
                if now >= p['stop_loss']:        status, hit = "Stop", True
                elif now <= p['profit_target']:  status, hit = "Target", True

            attr = 'hit' if hit else ('positive' if pnl_pct >= 0 else 'negative')
            rows.append((attr,
                f"{sym:8}{p['direction']:6}{p['entry_price']:>10.2f}"
                f"{now:>10.2f}{pnl_pct:>8.2f}%{p['stop_loss']:>10.2f}"
                f"{p['profit_target']:>10.2f}{status:>12}{p['entry_date']:>12}\n"))

            rec_info.append((p, hit, status, now, pnl_pct, today))
        return rows, rec_info

    # first render
    rec_cache=[]
    def refresh(*_):
        nonlocal rec_cache
        prices = get_prices()
        lines, rec_cache = build_table(prices)
        txt.set_text(lines)

    refresh()

    # key‑handler
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
        pnl_str = f"{rcd['pnl_pct']:+6.2f}%" # Use double quotes for dict key access
        print(f"{rcd['exit_date']}  {rcd['symbol']:5}  {rcd['direction']:5} "
              f"{rcd['status']:6}  PnL {pl_col(pnl_str)}")

    # ── clear option ──
    if input(Fore.YELLOW + "\n(C)lear stats or Enter to return: " + Style.RESET_ALL).lower() == 'c':
        save_predictions([p for p in load_predictions() if p['status'] == 'Open'])
        print(Fore.YELLOW + "History cleared." + Style.RESET_ALL)
        input("\nPress Enter to return …")


def run_signals_for_watchlist(side: str, use_intraday: bool = True):
    """
    Generate signals **only** of the correct direction for the requested list.
      side = 'long'  → look for LONG signals
      side = 'short' → look for SHORT signals
    """
    fred_api_key = load_config()
    tickers = load_watchlist(side)
    if not tickers:
        print(f"{side.capitalize()} watch-list is empty.")
        return

    today       = datetime.datetime.today()
    for ticker in tickers:
        print(f"\n=== {ticker} ({side}) ===")
        df_5m, df_30m, df_1h, df_90m, df_1d = get_or_fetch(ticker)

        # build feature set ------------------------------------------------
        feats = (prepare_features_intraday(df_30m) if use_intraday
                 else prepare_features(df_5m, df_30m, df_1h,
                                       df_90m, df_1d))
        feats = refine_features(feats)
        if feats.empty or 'future_class' not in feats.columns:
            print("No valid data.")
            continue

        model, thr = tune_threshold_and_train(feats)
        if model is None:
            print("Model training failed.")
            continue

        latest = feats.drop(columns='future_class').iloc[-1]
        sig    = generate_signal_output(ticker, latest, model, thr)

        # ensure direction matches list -----------------------------------
        if sig and side.upper() in sig:
            print(sig)

def interactive_menu():
    while True:
        print("\nMain Menu:")
        print("1. Manage Watch-lists")
        print("2. Run Signals on BOTH Watch-lists (live)")
        print("3. Show Signals Since Start of Week")
        print("4. Backtest ALL Watch-lists")
        print("5. Show Latest Signals Performance")   
        print("6. Closed-Trades Statistics")          
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == '0':
            print("Exiting."); break

        elif choice == '1':
            manage_watchlist()

        elif choice == '2':
            run_signals_on_watchlists(use_intraday=True)

        elif choice == '3':
            show_signals_since_start_of_week()

        elif choice == '4':
            backtest_watchlist()

        elif choice == '5':
            _update_positions_status() # Call the reimplemented function
            print("Position statuses updated (if any hits).")
            signals_performance_cli()

        elif choice == '6':                     
            closed_stats_cli()

        else:
            print("Invalid option.")


def main():
    parser = argparse.ArgumentParser(description="Swing trading signal generator/backtester")
    parser.add_argument('tickers', nargs='*', help="List of stock ticker symbols to analyze")
    parser.add_argument('--log', action='store_true', help="Optionally log selected trades to a file")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode instead of live signal mode")
    parser.add_argument('--start', default=None, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument('--real', action='store_true', help="Use 30-min intraday backtest for max realism")
    parser.add_argument('--live-real', action='store_true',
                        help="Use 30-min intraday pipeline in live mode")
    args = parser.parse_args()

    # If no positional tickers and no flags set, go interactive:
    if not args.tickers and not any(vars(args).values()):
        interactive_menu()
        return

    fred_api_key = load_config()
    tickers      = args.tickers
    log_trades   = args.log
    run_backtest = args.backtest
    start_arg    = args.start
    end_arg      = args.end
    use_intraday = args.real

    log_file = open("trades_log.csv", "a") if log_trades else None

    # --------------------------- BACK-TEST MODE ---------------------------
    if run_backtest and start_arg and end_arg:

        # ---- NEW: one bulk cache warm-up for all requested symbols ----
        preload_interval_cache(tickers, period="60d",  interval="30m")
        preload_interval_cache(tickers, period="380d", interval="1d")
        # ----------------------------------------------------------------

        if use_intraday:
            for ticker in tickers:
                fname = f"{ticker}_30m_{start_arg}_{end_arg}.csv"
                with open(fname, "a") as lf:
                    print(f"\n=== Intraday Backtesting {ticker} (30m) {start_arg} → {end_arg} ===")
                    backtest_strategy_intraday(ticker, start_arg, end_arg, log_file=lf)
        else:
            for ticker in tickers:
                fname = f"{ticker}_{start_arg}_{end_arg}.csv"
                with open(fname, "a") as lf:
                    print(f"\n=== Backtesting {ticker} {start_arg} → {end_arg} ===")
                    backtest_strategy(ticker, start_arg, end_arg,  log_file=lf)

        if log_file: log_file.close()
        return

    # --------------------------- LIVE MODE (unchanged) -------------------
    today       = datetime.datetime.today()

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live signal mode) ===")
        try:
            df_5m, df_30m, df_1h, df_90m, df_1d = get_or_fetch(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        if df_1d.empty or 'Close' not in df_1d.columns:
            print("No usable daily data, skipping.")
            continue

        feat = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d)
        if feat.empty:
            print("Insufficient data, skipping.")
            continue

        feat = refine_features(feat)
        if feat.empty or 'future_class' not in feat.columns:
            continue

        model, thr = tune_threshold_and_train(feat)
        if model is None:
            print("Model training failed.")
            continue

        latest_row = feat.drop(columns='future_class').iloc[-1]
        sig = generate_signal_output(ticker, latest_row, model, thr)
        print(sig)

        if log_file and sig:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{now},{ticker},{sig}\n")
            log_file.flush()

    if log_file: log_file.close()



if __name__ == "__main__":
    main()
