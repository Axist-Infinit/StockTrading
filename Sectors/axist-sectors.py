#!/usr/bin/env python3
import sys
user_site_packages = "/home/jules/.local/lib/python3.10/site-packages"
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

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
# NOTE: sys is already imported above for path modification
import argparse
import datetime
import json
import pandas as pd
import numpy as np
import requests
# import urwid # Commented out to avoid TUI issues       
from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest # Added imports

from watchlist_utils import load_watchlist, save_watchlist, manage_watchlist
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator, ADXIndicator, MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum  import (RSIIndicator, StochasticOscillator, ROCIndicator,
                          WilliamsRIndicator)
from ta.volatility import (BollingerBands, AverageTrueRange,
                           KeltnerChannel, DonchianChannel)
from ta.volume   import (MFIIndicator, OnBalanceVolumeIndicator,
                         ChaikinMoneyFlowIndicator)
from ta.volatility import AverageTrueRange
from ta.volume import MFIIndicator  # optional if you want more volume-based features
from ta.volatility import DonchianChannel  # optional, if needed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from xgboost import XGBClassifier
import time, threading
from pathlib import Path



from colorama import init, Fore, Style
init(autoreset=True)

# Helper functions for colored output
def g(text): return Fore.GREEN + str(text) + Style.RESET_ALL
def r(text): return Fore.RED + str(text) + Style.RESET_ALL
def b(text): return Fore.BLUE + str(text) + Style.RESET_ALL

from fredapi import Fred 
import configparser
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def _init_alpaca_client(keys: dict) -> StockHistoricalDataClient:
    """
    Create one StockHistoricalDataClient.

    • If the [ALPACA] section is present → use those creds.
    • If not, fall back to ALPACA_API_KEY / ALPACA_SECRET_KEY
      environment variables (exactly how alpaca-py does it).
    """
    if not keys:                         # no INI creds → rely on env-vars
        return StockHistoricalDataClient()

    return StockHistoricalDataClient(
        api_key    = keys.get("api_key") or keys.get("key_id"),
        secret_key = keys.get("secret_key") or keys.get("api_secret") # Check for secret_key first
    )

# Corrected _tf_map to use TimeFrame constructor
_tf_map = {
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "30m": TimeFrame(30, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4h":  TimeFrame(4,  TimeFrameUnit.Hour),   
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
    "1wk": TimeFrame(1,  TimeFrameUnit.Week),
}

PREDICTIONS_FILE = "weekly_signals.json"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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

# ── Alpaca client (global singleton) ─────────────────────────────────
_ALP_KEYS   = load_config("ALPACA")           # may be {} if section absent
_ALP_CLIENT = _init_alpaca_client(_ALP_KEYS)
# ─────────────────────────────────────────────────────────────────────


def alpaca_download(symbol: str, *,
                    start: str | None = None,
                    end:   str | None = None,
                    timeframe: str = "1d", # This is a string e.g. "1d"
                    limit: int | None = None) -> pd.DataFrame:
    """Return a tz‑naïve OHLCV DataFrame from Alpaca."""
    
    alpaca_timeframe_obj = _tf_map.get(timeframe) # Changed variable name for clarity
    if alpaca_timeframe_obj is None:
        raise ValueError(f"Unsupported timeframe string: {timeframe}")

    start_dt  = pd.to_datetime(start) if start else None
    end_dt    = pd.to_datetime(end)   if end   else None
    
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_timeframe_obj, # Use the TimeFrame object
        start=start_dt,
        end=end_dt,
        limit=limit
    )

    try:
        _inc_api_calls()
        bars = _ALP_CLIENT.get_stock_bars(request_params).df
    except APIError as e:
        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
            print(Fore.YELLOW + "Alpaca API rate limit (429) encountered in alpaca_download. Please wait before trying again." + Style.RESET_ALL)
        raise RuntimeError(f"Alpaca download error: {e}")

    if bars.empty:
        return bars
    
    # Ensure consistent index handling for single vs multi-symbol results from .df
    # If 'symbol' column exists (from multi-symbol request), set it as index then droplevel
    if 'symbol' in bars.columns:
        bars = bars.set_index('symbol', append=True).swaplevel(0, 1)

    if isinstance(bars.index, pd.MultiIndex):
        try: # Attempt to select current symbol if present in MultiIndex
            bars = bars.loc[symbol]
        except KeyError: # If symbol not found (e.g. empty df for one symbol in multi-request)
             return pd.DataFrame() # Return empty DF for this symbol


    bars.rename(columns={"open":"Open","high":"High","low":"Low",
                         "close":"Close","volume":"Volume"}, inplace=True)
    bars.index = bars.index.tz_localize(None) 
    return bars[["Open","High","Low","Close","Volume"]]

_API_CALL_COUNT = 0

def _inc_api_calls(n: int = 1) -> None:
    """Increment the global counter each time we hit the Alpaca REST API."""
    global _API_CALL_COUNT
    _API_CALL_COUNT += n

def reset_api_call_count() -> None:
    global _API_CALL_COUNT
    _API_CALL_COUNT = 0

def get_api_call_count() -> int:
    return _API_CALL_COUNT

def _wait_for_token():
    """
    Simple token bucket: allows up to _MAX_QPS calls per second total.
    """
    global _TOKENS, _LAST_REFILL
    with _LOCK:
        now = time.time()
        # Refill tokens based on how many seconds have passed:
        elapsed = now - _LAST_REFILL
        new_tokens = int(elapsed * _MAX_QPS)
        if new_tokens > 0:
            _TOKENS = min(_TOKENS + new_tokens, _MAX_QPS)
            _LAST_REFILL = now

        # If no tokens, wait long enough for 1 token to appear:
        if _TOKENS == 0:
            time.sleep(1.0 / _MAX_QPS)
            _TOKENS = 1
            _LAST_REFILL = time.time()
        else:
            _TOKENS -= 1

def _cache_path(ticker: str, interval: str) -> str:
    """Parquet file location for one ticker/interval."""
    return os.path.join(CACHE_DIR, f"{ticker.replace('/','_')}_{interval}.parquet")

def _load_cache(ticker: str, interval: str) -> pd.DataFrame:
    path = _cache_path(ticker, interval)
    return pd.read_parquet(path) if os.path.isfile(path) else pd.DataFrame()

def _save_cache(df: pd.DataFrame, ticker: str, interval: str):

    if df.empty:
        return

    # Convert index to tz-naive if required
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    df.sort_index(inplace=True)
    df.to_parquet(_cache_path(ticker, interval))

def _safe_download(ticker, *, period=None, start=None, end=None,
                   interval="1d", **__):
    """Maintains original signature but proxies to Alpaca."""
    if period and not start:
        # translate Yahoo‑style period into limit (rough heuristic)
        days = int(period.rstrip("d")) if period.endswith("d") else None
        limit = None if days is None else max(5, days)
        return alpaca_download(ticker, timeframe=interval, limit=limit)

    return alpaca_download(ticker, timeframe=interval, start=start, end=end)


def _interval_to_polygon(interval: str) -> tuple[int, str]:
    """
    Translate a yfinance interval into Polygon’s /v2/aggs parameters.
    """
    if interval == "1d":   return 1, "day"
    if interval == "1wk":  return 1, "week"          # NEW ➜ fixes crash
    if interval.endswith("m"):
        return int(interval.rstrip("m")), "minute"   # 15m, 30m …
    if interval.endswith("h"):
        return int(interval.rstrip("h")), "hour"     # 1h, 2h …
    raise ValueError(f"Unsupported interval: {interval}")


def _polygon_download(ticker: str,
                      start: str,
                      end:   str,
                      interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV from Polygon.io’s /v2/aggs endpoint and return a
    yfinance-style DataFrame (tz-naive index, column names Open…Close).
    """
    api_key = _load_polygon_key()
    if not api_key:
        print("[ERROR] Polygon API key not found – returning empty frame.")
        return pd.DataFrame()

    mult, span = _interval_to_polygon(interval)
    poly_ticker = ticker.lstrip("^")               # Polygon dislikes “^”

    url = (f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/"
           f"{mult}/{span}/{start}/{end}")
    params = {"adjusted": "true", "sort": "asc",
              "limit": 50000, "apiKey": api_key}

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        if js.get("status") != "OK" or not js.get("results"):
            print(f"[WARN] Polygon returned no data for {ticker}.")
            return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Polygon download failed for {ticker}: {e}")
        return pd.DataFrame()

    rows = [{
        "Date":   datetime.datetime.fromtimestamp(bar["t"] / 1000,
                                                  tz=datetime.timezone.utc),
        "Open":   bar["o"],
        "High":   bar["h"],
        "Low":    bar["l"],
        "Close":  bar["c"],
        "Volume": bar.get("v", np.nan)
    } for bar in js["results"]]

    df = (pd.DataFrame(rows)
            .set_index("Date")
            .sort_index())

    # remove timezone so it matches Yahoo frames
    df.index = df.index.tz_localize(None)
    return df

def preload_interval_cache(symbols: list[str],
                           *,
                           batch_size: int = 4,    # kept for signature parity
                           pause_sec: float = 0.25) -> None:
    """
    Warm the on-disk cache with recent bars – **Alpaca only**.

    Intervals:
        15-min & 30-min : last 60 days
        1-hour          : last 120 days
        4-hour          : last 380 days   (≈ 18 m)
        1-day           : last 380 days
    """
    combos = [
        (60,  '15m'),
        (60,  '30m'),
        (120, '1h'),
        (380, '4h'),
        (380, '1d'),
    ]
    symbols = sorted(set(symbols))
    if not symbols:
        return

    today = pd.Timestamp.utcnow().normalize()

    for days, ivl in combos:
        start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        for sym in symbols:
            try:
                df = alpaca_download(sym, timeframe=ivl, start=start)
                _save_cache(df, sym, ivl)
            except Exception as e:
                print(f"[WARN] Preload {sym}/{ivl} failed: {e}")
            time.sleep(pause_sec)   # gentle pause for rate-limits

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


def _update_positions_status() -> None:
    preds = load_predictions()
    open_pos = [p for p in preds if p.get("status") == "Open"]
    if not open_pos:
        return

    price_map = {}
    unique_symbols = sorted(list({p["symbol"] for p in open_pos}))

    if unique_symbols:
        try:
            latest_bars_req = StockLatestBarRequest(symbol_or_symbols=unique_symbols)
            _inc_api_calls()
            latest_bars_data = _ALP_CLIENT.get_stock_latest_bar(latest_bars_req)
            for sym, bar_data in latest_bars_data.items():
                if bar_data: # Check if bar_data is not None
                    price_map[sym] = float(bar_data.close)
        except APIError as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                print(Fore.YELLOW + "Alpaca API rate limit (429) encountered in _update_positions_status. Please wait before trying again." + Style.RESET_ALL)
            print(f"APIError in _update_positions_status: {e}")
            for sym_needed in unique_symbols:
                if sym_needed not in price_map:
                    price_map[sym_needed] = next(p["entry_price"] for p in open_pos if p["symbol"] == sym_needed)
    
    today = datetime.date.today().isoformat(); changed = False
    for rec in open_pos:
        now = price_map.get(rec["symbol"], rec["entry_price"]) 
        if rec["direction"] == "LONG":
            hit_stop   = now <= rec["stop_loss"]
            hit_target = now >= rec["profit_target"]
        else:
            hit_stop   = now >= rec["stop_loss"]
            hit_target = now <= rec["profit_target"]
        if hit_stop or hit_target:
            rec.update({"status":"Stop" if hit_stop else "Target", "exit_date":today, "exit_price":round(now,2)})
            changed = True
    if changed:
        save_predictions(preds)

def cached_download(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Disk cache identical to before, but network fetch is Alpaca‑only."""
    cached = _load_cache(ticker, interval)
    if not cached.empty and cached.index.tz is not None:
        cached.index = cached.index.tz_localize(None)

    need = cached.empty or pd.to_datetime(start) < cached.index.min() or \
           pd.to_datetime(end)   > cached.index.max()

    if need:
        new = alpaca_download(ticker, timeframe=interval, start=start, end=end)
        if not new.empty and new.index.tz is not None:
            new.index = new.index.tz_localize(None)
        df = pd.concat([cached, new]).sort_index().drop_duplicates()
        _save_cache(df, ticker, interval)
    else:
        df = cached

    return df.loc[start:end]

def fetch_data(
    ticker: str,
    start: str | None = None,
    end:   str | None = None,
    *,
    warmup_days: int = 300
) -> tuple[pd.DataFrame, ...]:
    """
    Retrieve six time-frames for *ticker* **with enough history** that all
    indicator look-backs (up to 21) are always satisfied.

    • When *start*/*end* are given (historical mode), the call *automatically*
      pulls an extra cushion of data **behind** the requested start date:
         – intraday (≤4 h) : +30 trading days
         – daily           : +`warmup_days`  (default 300)
         – weekly          : +120 weeks  (≈ 2.3 years)
    • Live mode (no *start/end*) keeps the original limits.
    """
    def _get(ivl_str: str, default_limit: int) -> pd.DataFrame:
        if start and end:                                  # ─ historical ─
            s_dt = pd.to_datetime(start)
            e_dt = pd.to_datetime(end)

            if ivl_str in {"15m", "30m", "1h", "4h"}:
                pad = s_dt - pd.Timedelta(days=30)          # +30 trading days
                fetch_start = pad.strftime("%Y-%m-%d")

            elif ivl_str == "1d":
                pad = s_dt - pd.Timedelta(days=warmup_days) # +warm-up
                fetch_start = pad.strftime("%Y-%m-%d")

            elif ivl_str == "1wk":
                pad = s_dt - pd.Timedelta(weeks=120)        # +120 weeks
                fetch_start = pad.strftime("%Y-%m-%d")

            else:                                           # fallback
                fetch_start = start

            return cached_download(
                ticker,
                start=fetch_start,
                end=e_dt.strftime("%Y-%m-%d"),
                interval=ivl_str
            ).loc[start:end]    # slice back to the exact user range

        # ─ live mode ───────────────────────────────────────────────────
        return alpaca_download(ticker, timeframe=ivl_str, limit=default_limit)

    # actual pulls ------------------------------------------------------
    df_15  = _get("15m", 60)
    df_30  = _get("30m", 60)
    df_1h  = _get("1h",  120)
    df_4h  = _get("4h",  120)
    df_1d  = _get("1d",  380)
    df_1wk = _get("1wk", 520)   # 520-bar cap in live mode (≈10 yrs)

    # tidy up -----------------------------------------------------------
    for df in (df_15, df_30, df_1h, df_4h, df_1d, df_1wk):
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)

    return df_15, df_30, df_1h, df_4h, df_1d, df_1wk

def compute_indicators(df: pd.DataFrame,
                       timeframe: str = "daily") -> pd.DataFrame:
    """
    Compute a broad indicator set with robust fall-backs for short or patchy
    price series.  All TA-Lib / ta library calls are *guarded* so we never hit
    IndexError when the look-back window is longer than the data we have.
    """
    df = df.copy()
    req_cols = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not req_cols.issubset(df.columns) or len(df) < 2:
        return df

    # ── adaptive look-back window ------------------------------------
    pct_vol = (
        df["Close"].pct_change().rolling(20).std().iloc[-1]
        if len(df) >= 21 else
        df["Close"].pct_change().std()
    ) or 0.0

    window = 21 if pct_vol >= 0.06 else 10 if pct_vol <= 0.01 else 14

    # If the series is shorter than the desired window, down-size safely
    if len(df) <= window:
        window = max(5, len(df) - 1)           # keep at least 5-bar context

    need_rows = window + 1                    # TA-lib needs ≥ window+1 rows
    non_na    = df[["High", "Low", "Close"]].dropna().shape[0]
    # -----------------------------------------------------------------

    # quick helpers ---------------------------------------------------
    def _safe_rsi(series):            # always works (short series OK)
        return RSIIndicator(series, window).rsi()

    def _manual_atr(high, low, close, win):
        """
        True-range computed entirely with pandas, so the result is a
        Series and `.rolling()` works.  (The old version built a raw
        ndarray, which had no `rolling` attribute.)
        """
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(win, min_periods=1).mean()

    # base indicators (never raise) -----------------------------------
    rsi  = _safe_rsi(df["Close"])
    sto  = StochasticOscillator(df["High"], df["Low"], df["Close"],
                                window, smooth_window=3)
    cci  = CCIIndicator(df["High"], df["Low"], df["Close"], window, 0.015)
    macd = MACD(df["Close"], 26, 12, 9)
    bb   = BollingerBands(df["Close"], 20, 2)
    roc  = ROCIndicator(df["Close"], window)
    wlr  = WilliamsRIndicator(df["High"], df["Low"], df["Close"], window)
    mfi  = MFIIndicator(df["High"], df["Low"], df["Close"],
                        df["Volume"], window)
    obv  = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    cmf  = ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"],
                                     df["Volume"], window)
    # -----------------------------------------------------------------

    # ATR / Keltner ----------------------------------------------------
    if len(df) >= need_rows and non_na >= need_rows:
        atr_ta = AverageTrueRange(df["High"], df["Low"], df["Close"], window)
        df[f"ATR_{timeframe}"]     = atr_ta.average_true_range()
        df[f"ATR_pct_{timeframe}"] = df[f"ATR_{timeframe}"] / df["Close"]

        kc = KeltnerChannel(df["High"], df["Low"], df["Close"],
                            window, original_version=False)
        df[f"KC_upper_{timeframe}"]  = kc.keltner_channel_hband()
        df[f"KC_lower_{timeframe}"]  = kc.keltner_channel_lband()
        df[f"KC_middle_{timeframe}"] = (
            kc.keltner_channel_mavg() if hasattr(kc, "keltner_channel_mavg")
            else kc.keltner_channel_mband()
        )
    else:
        atr = _manual_atr(df["High"], df["Low"], df["Close"])
        ema = df["Close"].ewm(span=window, adjust=False).mean()
        rng = atr * 1.5
        df[f"ATR_{timeframe}"]        = atr
        df[f"ATR_pct_{timeframe}"]    = atr / df["Close"]
        df[f"KC_middle_{timeframe}"]  = ema
        df[f"KC_upper_{timeframe}"]   = ema + rng
        df[f"KC_lower_{timeframe}"]   = ema - rng
    # -----------------------------------------------------------------

    # DI± / DX / ADX ---------------------------------------------------
    if len(df) >= need_rows and non_na >= need_rows:
        try:
            adx_ta = ADXIndicator(df["High"], df["Low"], df["Close"], window)
            df[f"ADX_{timeframe}"]     = adx_ta.adx()
            df[f"ADX_pos_{timeframe}"] = adx_ta.adx_pos()
            df[f"ADX_neg_{timeframe}"] = adx_ta.adx_neg()
        except IndexError:
            # Fall back to manual method if ta-lib still balks
            up  = df["High"].diff()
            dn  = -(df["Low"].diff())
            pos_dm = np.where((up > dn) & (up > 0), up, 0.0)
            neg_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

            atr_sum = _manual_atr(df["High"], df["Low"], df["Close"]) * window
            pos_di  = 100 * pd.Series(pos_dm, index=df.index).rolling(
                          window, min_periods=1).sum() / atr_sum
            neg_di  = 100 * pd.Series(neg_dm, index=df.index).rolling(
                          window, min_periods=1).sum() / atr_sum
            dx      = 100 * (pos_di - neg_di).abs() / (
                          pos_di + neg_di).replace(0, np.nan)
            adx     = dx.rolling(window, min_periods=1).mean()

            df[f"ADX_{timeframe}"]     = adx
            df[f"ADX_pos_{timeframe}"] = pos_di
            df[f"ADX_neg_{timeframe}"] = neg_di
    else:
        # manual-only branch
        up  = df["High"].diff()
        dn  = -(df["Low"].diff())
        pos_dm = np.where((up > dn) & (up > 0), up, 0.0)
        neg_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

        atr_sum = _manual_atr(df["High"], df["Low"], df["Close"]) * window
        pos_di  = 100 * pd.Series(pos_dm, index=df.index).rolling(
                      window, min_periods=1).sum() / atr_sum
        neg_di  = 100 * pd.Series(neg_dm, index=df.index).rolling(
                      window, min_periods=1).sum() / atr_sum
        dx      = 100 * (pos_di - neg_di).abs() / (
                      pos_di + neg_di).replace(0, np.nan)
        adx     = dx.rolling(window, min_periods=1).mean()

        df[f"ADX_{timeframe}"]     = adx
        df[f"ADX_pos_{timeframe}"] = pos_di
        df[f"ADX_neg_{timeframe}"] = neg_di
    # -----------------------------------------------------------------

    # Donchian (always safe) ------------------------------------------
    dc = DonchianChannel(df["High"], df["Low"], df["Close"], window)
    df[f"DC_upper_{timeframe}"]  = dc.donchian_channel_hband()
    df[f"DC_lower_{timeframe}"]  = dc.donchian_channel_lband()
    df[f"DC_middle_{timeframe}"] = (
        dc.donchian_channel_mavg() if hasattr(dc, "donchian_channel_mavg")
        else dc.donchian_channel_mband()
    )
    # -----------------------------------------------------------------

    # Attach remaining series -----------------------------------------
    df[f"RSI_{timeframe}"]          = rsi
    df[f"STOCHk_{timeframe}"]       = sto.stoch()
    df[f"STOCHd_{timeframe}"]       = sto.stoch_signal()
    df[f"CCI_{timeframe}"]          = cci.cci()
    df[f"MACD_{timeframe}"]         = macd.macd()
    df[f"MACD_signal_{timeframe}"]  = macd.macd_signal()
    df[f"MACD_hist_{timeframe}"]    = macd.macd_diff()
    df[f"BB_upper_{timeframe}"]     = bb.bollinger_hband()
    df[f"BB_lower_{timeframe}"]     = bb.bollinger_lband()
    df[f"BB_middle_{timeframe}"]    = bb.bollinger_mavg()
    df[f"ROC_{timeframe}"]          = roc.roc()
    df[f"WR_{timeframe}"]           = wlr.williams_r()
    df[f"MFI_{timeframe}"]          = mfi.money_flow_index()
    df[f"OBV_{timeframe}"]          = obv.on_balance_volume()
    df[f"CMF_{timeframe}"]          = cmf.chaikin_money_flow()

    # Higher-timeframe moving averages (if requested) -----------------
    if timeframe in {"daily", "hourly", "weekly", "1wk"}:
        df[f"SMA20_{timeframe}"]  = SMAIndicator(df["Close"], 20).sma_indicator()
        df[f"SMA50_{timeframe}"]  = SMAIndicator(df["Close"], 50).sma_indicator()
        df[f"EMA50_{timeframe}"]  = EMAIndicator(df["Close"], 50).ema_indicator()
        df[f"EMA200_{timeframe}"] = EMAIndicator(df["Close"], 200).ema_indicator()

    return df



def triple_barrier_labels(close: pd.Series,
                          atr:   pd.Series,
                          horizon: int = 10,
                          stop_mult: float = 2.0,
                          tgt_mult:  float = 2.0) -> np.ndarray:
    """
    Return 0 = hit lower barrier first
           1 = stayed in the middle (noise / flat)
           2 = hit upper barrier first
    """
    upper = close + tgt_mult  * atr
    lower = close - stop_mult * atr
    closes = close.values
    label  = np.ones(len(close), dtype=int)          

    for i in range(len(close) - horizon):
        win = closes[i + 1 : i + 1 + horizon]
        hit_up = np.where(win >= upper.iloc[i])[0]
        hit_dn = np.where(win <= lower.iloc[i])[0]
        if hit_up.size and hit_dn.size:
            label[i] = 2 if hit_up[0] < hit_dn[0] else 0
        elif hit_up.size:
            label[i] = 2
        elif hit_dn.size:
            label[i] = 0
    return label

def tune_xgb_hyperparams(X: pd.DataFrame,
                         y: np.ndarray,
                         *, n_iter: int = 25,
                         random_state: int = 42) -> dict:
    """
    Light-weight random search with time-series CV → returns best_params.
    """
    param_dist = {
        "max_depth":        np.arange(2, 9),
        "learning_rate":    np.linspace(0.01, 0.2, 20),
        "subsample":        np.linspace(0.6, 1.0, 5),
        "colsample_bytree": np.linspace(0.5, 1.0, 6),
        "gamma":            [0, 0.5, 1, 2],
        "min_child_weight": [1, 5, 10],
        "n_estimators":     np.arange(100, 501, 50),
        "reg_lambda":       np.linspace(0, 5, 6),
        "reg_alpha":        np.linspace(0, 2, 5),
    }
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1
    )
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=tscv,
        scoring="neg_log_loss", random_state=random_state,
        verbose=0, n_jobs=-1
    )
    search.fit(X, y)
    return search.best_params_



def compute_anchored_vwap(df_1d):
    """
    Compute anchored VWAP from the lowest low of the past 52 weeks on daily data.
    Returns a Series of anchored VWAP values (NaN before the anchor date).
    """
    if df_1d.empty:
        return pd.Series(dtype=float)

    lookback_period = 252 
    recent_period = df_1d.tail(lookback_period)
    
    # Check if 'Low' column exists and is not all NaN
    if 'Low' not in recent_period.columns or recent_period['Low'].isnull().all():
        return pd.Series(np.nan, index=df_1d.index)
        
    anchor_idx = recent_period['Low'].idxmin()
    
    if pd.isna(anchor_idx): # Should be caught by isnull().all() if 'Low' was all NaN
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

def prepare_features(
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_4h: pd.DataFrame, 
    df_1d:  pd.DataFrame,
    df_1wk: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    horizon: int = 10,
    drop_recent: bool = True,
    ticker: str | None = None
) -> pd.DataFrame:

    ind_15m = compute_indicators(df_15m.copy(), timeframe='15m')
    ind_30m = compute_indicators(df_30m.copy(), timeframe='30m')
    ind_1h  = compute_indicators(df_1h.copy(),  timeframe='hourly') 
    ind_4h  = compute_indicators(df_4h.copy(),  timeframe='4h') 
    
    ind_1d  = compute_indicators(df_1d.copy(),  timeframe='daily')

    # Early exit if ind_1d is not viable for further processing (Fix 1)
    if ind_1d.empty or 'Close' not in ind_1d.columns:
        ticker_name = ticker if ticker is not None else "N/A"
        print(Fore.YELLOW + f"Warning: Daily data (ind_1d) for ticker {ticker_name} is empty or missing 'Close' column after compute_indicators. Returning empty DataFrame from prepare_features." + Style.RESET_ALL)
        return pd.DataFrame()

    ind_1d['AnchoredVWAP'] = compute_anchored_vwap(ind_1d)
    ind_1wk = compute_indicators(df_1wk.copy(), timeframe='1wk')

    ind_htf = enrich_higher_timeframes(ind_1d)

    daily_15m = to_daily(ind_15m, "15m").add_suffix('_15m')
    daily_30m = to_daily(ind_30m, "30m").add_suffix('_30m')
    daily_1h  = to_daily(ind_1h,  "1h").add_suffix('_1h')
    daily_4h  = to_daily(ind_4h, "4h").add_suffix('_4h') 

    ind_1wk = ind_1wk.reindex(ind_1d.index, method='ffill')

    ind_1d.index.name = 'Date'
    features_df = (
        ind_1d
          .join(ind_htf)          
          .join(ind_1wk)          
          .join(daily_15m)
          .join(daily_30m)
          .join(daily_1h)
          .join(daily_4h) 
    )

    if macro_df is not None and not macro_df.empty:
        features_df = features_df.join(macro_df, on='Date')

    bench = {
        'XLE': 'CRUDE', 'XLF': 'YIELD10', 'XLK': 'SPY',
        'XLU': 'YIELD10', 'XLRE': 'YIELD10',
        'GLD': 'USD',   'DBC': 'CRUDE'
    }
    bcol = bench.get(ticker, 'SPY')
    if bcol in features_df.columns:
        features_df[f'RS_{ticker}'] = (
            features_df['Close'].pct_change(10) -
            features_df[bcol].pct_change(10)
        )

    features_df.dropna(subset=['Close'], inplace=True)
    if features_df.empty:
        return features_df

    labels = triple_barrier_labels(
        features_df['Close'],
        features_df['ATR_daily'].fillna(0),
        horizon=horizon,
        stop_mult=2.0, tgt_mult=2.0
    )
    features_df['future_class'] = labels

    if drop_recent:
        features_df = features_df.iloc[:-horizon]
    else:
        features_df.loc[features_df.index[-horizon:], 'future_class'] = np.nan

    return features_df

def refine_features(features_df,
                    importance_cutoff: float = 1e-4,
                    corr_threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop low-information and highly-correlated columns using a quick
    XGBoost probe – the probe runs entirely on the GPU.
    """
    if features_df.empty or 'future_class' not in features_df.columns:
        return features_df

    y = features_df['future_class']
    X = features_df.drop(columns=['future_class']).ffill().bfill()

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        tree_method='hist',      
        device='cuda'            
    )

    split = int(0.8 * len(X))
    if len(np.unique(y[:split])) < 2:       
        return features_df

    model.fit(X.iloc[:split], y.iloc[:split])
    imp = pd.Series(model.feature_importances_, index=X.columns)

    X = X.drop(columns=imp[imp < importance_cutoff].index, errors='ignore')

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper.columns if any(upper[c] > corr_threshold)],
               errors='ignore')

    # Ensure 'Close' column is preserved
    if 'Close' not in X.columns and 'Close' in features_df.columns:
        X = X.join(features_df[['Close']])
    elif 'Close' not in X.columns and 'Close' not in features_df.columns:
        # This case should ideally not happen if prepare_features works as expected
        print(Fore.RED + "[Error] 'Close' column missing from input to refine_features and could not be restored." + Style.RESET_ALL)

    return X.join(y)


def _summarise_performance(trades: pd.DataFrame, total_bars: int) -> dict:
    """
    Return a dict with win-rate, average P/L, trade-level Sharpe,
    max draw-down and time-in-market.
    """
    if trades.empty:
        return dict(total=0, win_rate=0.0, avg_pnl=0.0,
                    sharpe=0.0, max_dd=0.0, tim=0.0)

    pnl = trades["pnl_pct"].values
    win = pnl > 0
    win_rate = win.mean() if pnl.size else 0.0
    avg_pnl  = pnl.mean() if pnl.size else 0.0
    sharpe   = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252) if pnl.std(ddof=1) else 0.0

    equity = (1 + pnl).cumprod()
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    max_dd = abs(dd.min()) if dd.size else 0.0

    tim = (trades["exit_timestamp"] - trades["entry_timestamp"]) \
            .dt.total_seconds().sum() / (total_bars * 60 * 30) 
    tim = min(max(tim, 0), 1)            

    return dict(total=len(trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                sharpe=sharpe,
                max_dd=max_dd,
                tim=tim)


def train_stacked_ensemble(features_df: pd.DataFrame,
                           *, random_state: int = 42):

    if features_df.empty or "future_class" not in features_df.columns:
        return None, None

    y = features_df["future_class"].values
    X = features_df.drop(columns=["future_class"]).ffill().bfill()

    best_xgb_params = tune_xgb_hyperparams(X, y, n_iter=20,
                                           random_state=random_state)
    xgb_clf = XGBClassifier(**best_xgb_params,
                            objective="multi:softprob",
                            num_class=3,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            n_jobs=-1,
                            random_state=random_state)

    rf_clf  = RandomForestClassifier(
        n_estimators=400, max_depth=None,
        class_weight="balanced", n_jobs=-1,
        random_state=random_state
    )
    log_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        n_jobs=-1, solver="saga", random_state=random_state
    )

    stack = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
        final_estimator=log_clf,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False
    )

    tscv = TimeSeriesSplit(n_splits=4)
    best_thr, best_f1 = 0.60, -np.inf       

    for train_idx, val_idx in tscv.split(X):
        stack.fit(X.iloc[train_idx], y[train_idx])
        proba = stack.predict_proba(X.iloc[val_idx])

        for thr in np.arange(0.55, 0.91, 0.05):
            pred = np.where(
                proba[:, 2] >= thr, 2,    
                np.where(proba[:, 0] >= thr, 0, 1)  
            )
            mask = pred != 1
            if mask.sum() == 0:
                continue
            f1 = f1_score(y[val_idx][mask], pred[mask],
                          labels=[0, 2], average="macro")
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

    stack.fit(X, y)
    return stack, best_thr

def generate_signal_output(ticker, latest_row, model, thr, macro_latest):
    latest_df = latest_row.to_frame().T
    probs = model.predict_proba(latest_df.values)[0]
    class_idx = int(np.argmax(probs))
    prob_score = probs[class_idx]

    if prob_score < 0.60 or class_idx == 1:
        return None

    direction = "LONG" if class_idx == 2 else "SHORT"
    price = float(latest_row.get("Close", np.nan))

    reg = latest_row.get('REGIME', 0)
    if reg == -1 and direction == "LONG":
        return None
    if reg == 1 and direction == "SHORT":
        return None

    atr = latest_row.get("ATR_daily", np.nan)
    if not atr or np.isnan(atr):
        atr = 0.05 * price
        atr_note = " (est. ATR)"
    else:
        atr_note = ""

    stop   = price - atr if direction == "LONG" else price + atr
    target = price + 2 * atr if direction == "LONG" else price - 2 * atr

    ema50  = latest_row.get('EMA50_daily', np.nan)
    ema200 = latest_row.get('EMA200_daily', np.nan)
    trend  = "up" if ema50 > ema200 else "down"

    colour = Fore.GREEN if direction == "LONG" else Fore.RED
    return (f"{Fore.CYAN}{ticker}{Style.RESET_ALL}: {colour}{direction}{Style.RESET_ALL} "
            f"${price:.2f}  Stop {stop:.2f}  Target {target:.2f}  "
            f"Prob {prob_score:.2f}  Trend {trend}{atr_note}")

def get_macro_data(start: str,
                   end:   str,
                   fred_api_key: str | None = None,
                   *,
                   cache_dir: str = ".macro_cache") -> pd.DataFrame:
    """
    Build a daily macro / cross-asset DataFrame **without yfinance**.

    • Uses Alpaca for liquid ETF proxies (SPY, TIP, UUP, USO, HYG, TLT, etc.).
    • Optional FRED add-ons (10-yr yield, 2-10 spread) still supported.
    • Caches the merged result per-day on disk.
    """
    from pathlib import Path
    import joblib, datetime as _dt

    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"macro_{end}.pkl"
    if cache_file.exists():
        try:
            return joblib.load(cache_file)
        except Exception:
            cache_file.unlink(missing_ok=True)

    etf_map = {             
        "SPY":  "SPY",      
        "TIP":  "TIP",      
        "UUP":  "USD",      
        "USO":  "CRUDE",    
        "HYG":  "HYG",      
        "TLT":  "TLT",      
    }
    price_df = pd.DataFrame()
    for etf, col in etf_map.items():
        try:
            df = alpaca_download(etf, timeframe="1d", start=start, end=end)
            price_df[col] = df["Close"]
        except Exception as e:
            print(f"[WARN] Macro fetch failed for {etf}: {e}")

    fred_df = pd.DataFrame(index=price_df.index)
    if fred_api_key:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)

        def _fred(code):
            return (fred.get_series(code,
                                     observation_start='1970-01-01',
                                     observation_end=end)
                        .resample('D').ffill())

        try:
            fred_df['YIELD10']      = _fred('DGS10')
            fred_df['YIELD_SPREAD'] = _fred('T10Y2Y')
        except Exception as e:
            print(f"[WARN] FRED fetch failed: {e}")

    macro = pd.concat([price_df, fred_df], axis=1)
    if macro.empty:
        macro = pd.DataFrame(index=pd.date_range(start, end, freq='D'))

    macro = macro.asfreq('D').ffill()

    reg = pd.Series(0, index=macro.index, dtype=int)
    if 'VIX' in macro:          reg += (macro['VIX'] > 25).astype(int) * -1
    if 'YIELD_SPREAD' in macro: reg += (macro['YIELD_SPREAD'] > 0).astype(int)
    if 'USD' in macro:
        reg += (macro['USD'].pct_change(20) < 0).fillna(False).astype(int)
    macro['REGIME'] = reg.clip(-1, 1)

    try:
        joblib.dump(macro, cache_file)
    except Exception:
        pass
    return macro.loc[start:end]

def backtest_strategy(ticker: str,
                      start_date: str,
                      end_date:   str,
                      macro_df:  pd.DataFrame,
                      log_file=None):
    """
    Daily-lumped back-test using the six-frame fetch & stacked ensemble.
    """
    df_15m, df_30m, df_1h, df_4h, df_1d, df_1wk = fetch_data(
        ticker, start=start_date, end=end_date
    )

    feat = prepare_features(df_15m, df_30m, df_1h, df_4h,
                            df_1d,  df_1wk, macro_df, ticker=ticker) 
    feat = refine_features(feat)

    model, _ = train_stacked_ensemble(feat)          
    if model is None:
        print(Fore.YELLOW + f"Model training failed for {ticker}." + Style.RESET_ALL)
        return

    feat = feat.dropna(subset=["ATR_daily"])
    X_all = feat.drop(columns=["future_class"]).ffill().bfill()
    preds = model.predict(X_all)
    prob  = model.predict_proba(X_all)

    df_pred = feat.assign(prediction=preds)

    trades, in_trade = [], False
    # Fix 2: Changed loop to use enumerate for integer index `i`
    for i, (ts, row) in enumerate(df_pred.iterrows()):
        price, atr = row["Close"], row["ATR_daily"]
        if np.isnan(atr) or atr == 0:
            continue
        # Fix 2: Use `prob[i]` instead of `prob[row.name]`
        if not in_trade and prob[i][row.prediction] >= 0.6:
            dir_ = "LONG" if row.prediction == 2 else "SHORT"
            entry, stop = price, price - atr if dir_ == "LONG" else price + atr
            target = price + 2 * atr if dir_ == "LONG" else price - 2 * atr
            trades.append({"entry": ts, "dir": dir_, "entry_px": entry,
                           "stop": stop, "target": target})
            in_trade = True
        elif in_trade:
            t = trades[-1]
            if (t["dir"] == "LONG"  and (price <= t["stop"] or price >= t["target"])) or \
               (t["dir"] == "SHORT" and (price >= t["stop"] or price <= t["target"])):
                t.update({"exit": ts, "exit_px": price})
                in_trade = False

    print(f"{ticker}: completed back-test from {start_date} to {end_date}.  "
          f"{len(trades)} trades executed.")

def prepare_features_intraday(df_30m, macro_df=None):
    """
    Computes technical indicators and triple-barrier labels directly on 30-minute data.
    Joins daily context (EMA, Anchored VWAP) for each 30-minute bar, then merges macro.
    """
    df_30m = df_30m.copy()  

    df_30m = compute_indicators(df_30m, timeframe='intraday')
    if df_30m.empty or 'Close' not in df_30m.columns:
        return pd.DataFrame()

    daily = to_daily(df_30m, "intraday")
    daily = compute_indicators(daily, timeframe='daily')
    daily['AnchoredVWAP'] = compute_anchored_vwap(daily)

    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)  
    if df_30m.index.tz is not None:
        df_30m.index = df_30m.index.tz_localize(None)  

    daily_filled = daily.reindex(df_30m.index, method='ffill')
    df_30m = df_30m.join(daily_filled, rsuffix='_daily')

    horizon_bars = 16
    multiplier = 2.0
    atr_col = 'ATR_intraday'
    if atr_col not in df_30m.columns:
        df_30m[atr_col] = df_30m['Close'].rolling(14).std().fillna(0)

    up_barrier = df_30m['Close'] + multiplier * df_30m[atr_col]
    down_barrier = df_30m['Close'] - multiplier * df_30m[atr_col]
    future_class = np.ones(len(df_30m), dtype=int)
    closes = df_30m['Close'].values

    for i in range(len(df_30m) - horizon_bars):
        upper_lvl = up_barrier.iloc[i]
        lower_lvl = down_barrier.iloc[i]
        window_prices = closes[i + 1 : i + 1 + horizon_bars]
        above_idx = np.where(window_prices >= upper_lvl)[0]
        below_idx = np.where(window_prices <= lower_lvl)[0]
        if len(above_idx) == 0 and len(below_idx) == 0:
            continue
        elif len(above_idx) > 0 and len(below_idx) > 0:
            if above_idx[0] < below_idx[0]:
                future_class[i] = 2
            else:
                future_class[i] = 0
        elif len(above_idx) > 0:
            future_class[i] = 2
        elif len(below_idx) > 0:
            future_class[i] = 0

    df_30m['future_class'] = future_class
    df_30m = df_30m.iloc[:-horizon_bars]

    if macro_df is not None and not macro_df.empty:
        if macro_df.index.tz is not None:
            macro_df = macro_df.copy()
            macro_df.index = macro_df.index.tz_localize(None)  

        macro_resampled = macro_df.reindex(df_30m.index, method='ffill')
        df_30m = df_30m.join(macro_resampled, how='left')

    return df_30m

def backtest_strategy_intraday(ticker, start_date, end_date, macro_df,
                               log_file=None):
    """
    30-minute intraday back-test with Sharpe, max-DD, time-in-market and ATR guard.
    """
    _, df_30m, _, _, _, _ = fetch_data(ticker, start=start_date, end=end_date)

    feat = prepare_features_intraday(df_30m, macro_df)
    feat = refine_features(feat)
    model, _ = train_stacked_ensemble(feat)
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
        
        class_idx  = int(row['prediction'])
        prob_score = probas[i][class_idx]
        if prob_score < 0.60:
            continue

        if class_idx == 2:      
            in_trade   = True
            trade_dir  = "LONG"
            entry_time = ts
            entry_price= price
            stop_price = price - atr
            target_price = price + 2 * atr
        elif class_idx == 0:    
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


def display_main_menu():
    print("\nMain Menu:")
    print("1. Manage Watchlist")
    print("2. Run Signals on Watchlist (Live Mode)")
    print("3. Backtest All Watchlist Tickers")
    print("0. Exit")

def run_signals_on_watchlist(use_intraday: bool = True):
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    preload_interval_cache(tickers)

    today = datetime.date.today()
    macro_df = get_macro_data(
        (today - datetime.timedelta(days=380)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
        fred_api_key=fred_api_key
    )

    for tkr in tickers:
        print(f"\n=== {tkr} (live) ===")
        try:
            df15, df30, df1h, df4h, df1d, df1w = fetch_data(tkr) 
        except Exception as e:
            print(f"Fetch error: {e}"); continue

        feats = (prepare_features_intraday(df30, macro_df)
                 if use_intraday else
                 prepare_features(df15, df30, df1h, df4h, df1d, df1w, macro_df, ticker=tkr)) 
        feats = refine_features(feats)
        model, thr = train_stacked_ensemble(feats)
        if model is None:
            print("Model training failed."); continue

        sig = generate_signal_output(
            tkr,
            feats.drop(columns='future_class').iloc[-1],
            model, thr,
            macro_df.iloc[-1].to_dict()
        )
        print(sig or "No actionable signal.")

def backtest_watchlist():
    """Runs back-tests on each watch-list ticker with a pre-warmed cache."""
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    start_arg = input("Enter backtest start date (YYYY-MM-DD): ").strip()
    end_arg   = input("Enter backtest end date   (YYYY-MM-DD): ").strip()
    if not start_arg or not end_arg:
        print("Invalid date range."); return

    preload_interval_cache(tickers)      

    macro_df = get_macro_data(start_arg, end_arg)
    for ticker in tickers:
        print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
        backtest_strategy(ticker, start_arg, end_arg, macro_df) 

def show_signals_for_current_week():
    """Print first actionable signal (if any) for each watch-list name."""
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty.")
        return

    # ── define the 180-day look-back used for features ────────────────
    today   = datetime.date.today()
    monday  = today - datetime.timedelta(days=today.weekday())          # this week’s Monday
    start_s = (monday - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    end_s   = today.strftime("%Y-%m-%d")

    # pull macro once (covers full range) -- no yfinance
    macro_df = get_macro_data(start_s, end_s, fred_api_key=fred_api_key)

    for tkr in tickers:
        print(f"\n=== {tkr}: {monday} → {today} ===")

        # key change: supply *start* & *end* so every interval (esp. 4 h)
        # contains the full history needed for 14-/21-bar indicators
        try:
            df15, df30, df1h, df4h, df1d, df1w = fetch_data(
                tkr, start=start_s, end=end_s
            )
        except Exception as e:
            print(f"Fetch error: {e}")
            continue

        # no slicing required – frames already span the desired range
        all_feat = prepare_features(
            df15, df30, df1h, df4h, df1d, df1w,
            macro_df, drop_recent=False, ticker=tkr
        )
        if all_feat.empty:
            print(Fore.YELLOW +
                  f"Insufficient data to process signals for {tkr}. Skipping."
                  + Style.RESET_ALL)
            continue

        lbl_df = refine_features(all_feat.dropna(subset=['future_class']))
        model, thr = train_stacked_ensemble(lbl_df)
        if model is None:
            continue

        # walk forward through this week and print the first actionable signal
        for dt, row in all_feat.loc[monday:end_s].iterrows():
            sig = generate_signal_output(tkr, row, model, thr, {})
            if sig:
                print(f"{dt.date()}: {sig}")
                break

def signals_performance_cli():
    """
    Dashboard of OPEN trades – price updates fetched from Alpaca in one
    multi-symbol request (→ no Yahoo dependency).
    """
    open_recs = [p for p in load_predictions() if p["status"] == "Open"]
    if not open_recs:
        print("No open positions.  Run option 5 first."); return

    palette = [
        ("title",    "white,bold", ""),
        ("headers",  "light blue,bold", ""),
        ("positive", "dark green", ""),
        ("negative", "dark red",   ""),
        ("hit",      "white",      "dark cyan"),
        ("footer",   "white,bold", "")
    ]
    
    def _get_last_prices() -> dict[str, float]:
        syms = sorted({p["symbol"] for p in open_recs})
        if not syms:
            return {}

        try:
            latest_bars_req = StockLatestBarRequest(symbol_or_symbols=syms)
            _inc_api_calls() # Added API call increment
            latest_bars = _ALP_CLIENT.get_stock_latest_bar(latest_bars_req) 
            
            price_dict = {}
            if isinstance(latest_bars, dict): 
                for symbol, bar_data in latest_bars.items():
                    if bar_data: 
                        price_dict[symbol] = bar_data.close 
            return price_dict

        except APIError as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                print(Fore.YELLOW + "Alpaca API rate limit (429) encountered in _get_last_prices. Please wait before trying again." + Style.RESET_ALL)
            print(Fore.YELLOW + f"Alpaca error in _get_last_prices: {e}" + Style.RESET_ALL)
            return {p["symbol"]: p["entry_price"] for p in open_recs}

    def _build_table(px: dict[str, float]):
        rows = [("headers",
                 f"{'Symbol':8}{'Dir':6}{'Entry':>10}{'Now':>10}{'P/L%':>8}"
                 f"{'Stop':>10}{'Target':>10}{'Status':>12}{'Date':>12}\n")]
        cache = []
        today = datetime.date.today().strftime("%Y-%m-%d")

        for rec in open_recs:
            sym, ep = rec["symbol"], rec["entry_price"]
            now     = float(px.get(sym, ep))

            pnl_pct = (now - ep) / ep * 100 if rec["direction"] == "LONG" \
                     else (ep - now) / ep * 100

            hit_stop   = (now <= rec["stop_loss"])   if rec["direction"] == "LONG" else (now >= rec["stop_loss"])
            hit_target = (now >= rec["profit_target"]) if rec["direction"] == "LONG" else (now <= rec["profit_target"])
            status = "Stop" if hit_stop else "Target" if hit_target else "Open"
            hit = status != "Open"

            attr = "hit" if hit else ("positive" if pnl_pct >= 0 else "negative")
            rows.append((attr,
                f"{sym:8}{rec['direction']:6}{ep:>10.2f}{now:>10.2f}"
                f"{pnl_pct:>8.2f}%{rec['stop_loss']:>10.2f}"
                f"{rec['profit_target']:>10.2f}{status:>12}{rec['entry_date']:>12}\n"))

            cache.append((rec, hit, status, now, pnl_pct, today))
        return rows, cache

    _table_cache = []
    def _refresh(*_):
        nonlocal _table_cache
        prices = _get_last_prices()
        # lines, _table_cache = _build_table(prices) # urwid related
        # w_body.set_text(lines) # Requires urwid
        # Instead of urwid, just print to console:
        print("\n--- Open Trades ---")
        table_rows, _table_cache = _build_table(prices)
        for _, line in table_rows: # Assuming second element is the string
            print(line.strip())
        print("-------------------\n")


    _refresh()
    print("signals_performance_cli is currently using basic print. (D)eject/(Q)uit/(R)efresh in input below.")
    
    # Simplified key handling for non-urwid version
    while True:
        action = input("(R)efresh, (D)eject hit, (Q)uit: ").lower()
        if action == 'q':
            break
        elif action == 'r':
            _refresh()
        elif action == 'd':
            changed = False
            for rec, hit, status, now, pnl, today_str in _table_cache: # Ensure today_str is used if it's just string
                if hit:
                    rec.update({"status": status,
                                "exit_price": round(now, 2),
                                "exit_date":  today_str, # Use the string date
                                "pnl_pct":    round(pnl, 2)})
                    changed = True
            if changed:
                all_preds = load_predictions() # Load all predictions
                # Update open_recs in all_preds before saving
                # This requires matching records, e.g. by symbol and entry_date if unique
                # For simplicity here, assuming open_recs is a subset of all_preds and modifications are reflected
                # A more robust way would be to create a new list of all predictions with updated items.
                updated_all_preds = []
                open_rec_dict = {(p['symbol'], p['entry_date']): p for p in open_recs}
                for p_all in all_preds:
                    key = (p_all['symbol'], p_all['entry_date'])
                    if p_all['status'] == 'Open' and key in open_rec_dict:
                         updated_all_preds.append(open_rec_dict[key]) # Add the modified record
                    else:
                        updated_all_preds.append(p_all) # Add unchanged record
                save_predictions(updated_all_preds)
                # Refresh open_recs for the current view
                open_recs = [p for p in updated_all_preds if p["status"] == "Open"]

            _refresh()


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

    for r_item in recs: 
        if 'pnl_pct' not in r_item or r_item['pnl_pct'] is None:
            ep, xp = r_item['entry_price'], r_item.get('exit_price', r_item['entry_price'])
            if r_item['direction'] == 'LONG':
                r_item['pnl_pct'] = (xp - ep) / ep * 100
            else:                              
                r_item['pnl_pct'] = (ep - xp) / ep * 100
            r_item['pnl_pct'] = round(r_item['pnl_pct'], 2)

    wins   = [r_item for r_item in recs if r_item['status'] == 'Target'] 
    losses = [r_item for r_item in recs if r_item['status'] in ('Stop', 'Closed')] 
    total  = len(recs)
    win_rt = (len(wins) / total * 100) if total > 0 else 0


    avg_win = np.mean([w['pnl_pct'] for w in wins])   if wins   else 0.0
    avg_los = np.mean([l['pnl_pct'] for l in losses]) if losses else 0.0

    equity = 1.0
    for r_item in recs: equity *= 1 + r_item['pnl_pct'] / 100 
    tot_ret = (equity - 1) * 100


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

    print(b("\nMost recent 10 closed trades:"))
    for rcd in recs[-10:]:
        print(f"{rcd.get('exit_date','N/A')} {rcd.get('symbol','N/A')} {rcd.get('direction','N/A')} {rcd.get('status','N/A')} PnL {rcd.get('pnl_pct', 0.0):+.2f}%")

    if input(Fore.YELLOW + "\n(C)lear stats or Enter to return: " + Style.RESET_ALL).lower() == 'c':
        save_predictions([p for p in load_predictions() if p['status'] == 'Open'])
        print(Fore.YELLOW + "History cleared." + Style.RESET_ALL)
        input("\nPress Enter to return …")



def interactive_menu():
    while True:
        print(f"\nAPI Calls Made: {get_api_call_count()}") # Added API call count print
        print("\nMain Menu:")
        print("1. Manage Watchlist")
        print("2. Run Signals on Watchlist (Live Mode)")
        print("3. Backtest All Watchlist Tickers")
        print("4. Schedule Signals (9:30 to 16:00 EST via Cron)")
        print("5. Show This Weeks Signals")
        print("6. Show Latest Signals Performance")
        print("7. Closed-Trades Statistics")        
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

    if not args.tickers and not any(vars(args).values()):
        interactive_menu()
        return

    fred_api_key = load_config()
    tickers = args.tickers
    log_trades = args.log
    run_backtest = args.backtest
    start_arg = args.start
    end_arg = args.end
    use_intraday = args.real

    log_file = None
    if log_trades:
        log_file = open("trades_log.csv", "a")

    if run_backtest and start_arg and end_arg:
        macro_df = get_macro_data(start_arg, end_arg, fred_api_key=fred_api_key)

        if use_intraday:
            for ticker in tickers:
                log_filename = f"{ticker}_30m_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Intraday Backtesting {ticker} (30m) from {start_arg} to {end_arg} ===")
                    backtest_strategy_intraday(ticker, start_arg, end_arg, macro_df, log_file=lf)
        else:
            for ticker in tickers:
                log_filename = f"{ticker}_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
                    backtest_strategy(ticker, start_arg, end_arg, macro_df, log_file=lf)
        if log_file:
            log_file.close()
        return

    today = datetime.datetime.today()
    macro_start = today - datetime.timedelta(days=380)
    macro_df = get_macro_data(
        macro_start.strftime('%Y-%m-%d'),
        today.strftime('%Y-%m-%d'),
        fred_api_key=fred_api_key
    )

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live signal mode) ===")
        try:
            df15, df30, df1h, df4h, df1d, df1w = fetch_data(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        if df1d.empty or 'Close' not in df1d.columns:
            print(f"No usable daily data for {ticker}, skipping.")
            continue

        features_df = prepare_features(df15, df30, df1h, df4h, df1d, df1w, macro_df, ticker=ticker) 
        if features_df.empty:
            print(f"Insufficient data for {ticker}, skipping.")
            continue

        features_df = refine_features(features_df)
        if features_df.empty or 'future_class' not in features_df.columns:
            continue

        model, best_thr = train_stacked_ensemble(features_df)
        if model is None:
            print(f"Model training failed or no valid data for {ticker}.")
            continue

        latest_features = features_df.iloc[-1]
        X_latest = latest_features.drop(labels='future_class', errors='ignore').to_frame().T
        X_latest.ffill(inplace=True)
        X_latest.bfill(inplace=True)
        latest_series = X_latest.iloc[0]

        latest_date = features_df.index[-1]
        if latest_date in macro_df.index:
            macro_latest = macro_df.loc[latest_date].to_dict()
        else:
            macro_latest = macro_df.iloc[-1].to_dict()

        signal_output = generate_signal_output(ticker, latest_series, model, best_thr, macro_latest)
        print(signal_output)

        if log_file and ("LONG" in signal_output or "SHORT" in signal_output):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{now_str},{ticker},{signal_output}\n")
            log_file.flush()

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
