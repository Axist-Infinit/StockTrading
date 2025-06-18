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
from requests_oauthlib import OAuth1Session
from pathlib import Path
# configparser is likely already imported or will be by load_config, ensure it's available
# import configparser
from typing import Sequence
# import urwid # Commented out to avoid TUI issues

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
# from pathlib import Path # Already imported above


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

class ETradeClient:
    _BASE = "https://apisb.etrade.com"

    def __init__(self, creds: dict):
        self.ck = creds["consumer_key"]
        self.cs = creds["consumer_secret"]
        self.tok = creds["access_token"]
        self.tok_sec = creds["access_token_secret"]
        self.env = creds.get("env", "sandbox")
        self._BASE = "https://api.etrade.com" if self.env == "prod" else "https://apisb.etrade.com"

    def _req(self, path: str, params: dict | None = None) -> dict:
        sess = OAuth1Session(self.ck, self.cs, self.tok, self.tok_sec)
        url = f"{self._BASE}{path}"
        filtered_params = {k: v for k, v in (params or {}).items() if v is not None}
        resp = sess.get(url, params=filtered_params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_quotes(self, symbols: Sequence[str]) -> dict[str, float]:
        if not symbols: return {}
        # Note: E*TRADE API quote limit is 25 symbols. Chunking needed for more.
        path = f"/v1/market/quote/{','.join(symbols)}.json"
        api_params = {"detailFlag": "INTRADAY"}
        try:
            js = self._req(path, params=api_params)
        except requests.exceptions.HTTPError as e:
            print(r(f"ETrade API HTTPError for quotes {symbols}: {e.response.status_code} {e.response.text if e.response else 'No response text'}"))
            return {s: np.nan for s in symbols}

        results = {s: np.nan for s in symbols}
        if "QuoteResponse" not in js or "QuoteData" not in js["QuoteResponse"]:
            print(r(f"Unexpected ETrade quote response for {symbols}: 'QuoteData' missing. Response: {js}"))
            return results

        q_root = js["QuoteResponse"]["QuoteData"]
        if not isinstance(q_root, list):
            if isinstance(q_root, dict) and "symbol" in q_root and q_root.get("all") and "lastTrade" in q_root["all"]:
                try: results[q_root["symbol"]] = float(q_root["all"]["lastTrade"])
                except: pass
            else: print(r(f"ETrade QuoteData is not a list for {symbols}. Response: {q_root}"))
            return results

        for q_data in q_root:
            if isinstance(q_data, dict) and "symbol" in q_data and q_data.get("all") and "lastTrade" in q_data["all"]:
                try: results[q_data["symbol"]] = float(q_data["all"]["lastTrade"])
                except (ValueError, TypeError): pass
        return results

    _ivl_map = {
        "1d": ("daily", 1), "1wk": ("weekly", 1),
        "15m": ("minute", 15), "30m": ("minute", 30),
        "1h": ("minute", 60), "4h": ("minute", 60),
    }

    def get_bars(self, symbol: str, *, start: str | None, end: str | None,
                 interval: str, limit: int | None = None) -> pd.DataFrame:
        if interval not in self._ivl_map:
            raise ValueError(f"Interval '{interval}' not supported. Supported: {list(self._ivl_map.keys())}")

        orig_interval_req = interval
        freq_type, freq_val = self._ivl_map[interval]

        if orig_interval_req == "4h" and freq_val == 60:
            # Using existing Fore and Style for color, assuming init(autoreset=True) is called
            print(Fore.YELLOW + f"[WARN] ETradeClient: Interval '4h' mapped to '1h' (60 min) data for {symbol}." + Style.RESET_ALL)

        api_params = {
            "periodType": "custom",
            "startDate": pd.Timestamp(start).strftime("%m%d%Y") if start else None,
            "endDate": pd.Timestamp(end).strftime("%m%d%Y") if end else None,
            "frequencyType": freq_type, "frequency": freq_val, "sortOrder": "ASC",
        }
        api_params = {k: v for k, v in api_params.items() if v is not None}
        path = f"/v1/market/quote/{symbol}/historical.json"

        try:
            js = self._req(path, params=api_params)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                alt_path = f"/v1/market/quote/{symbol}/timeseries.json"
                try:
                    js = self._req(alt_path, params=api_params)
                except requests.exceptions.HTTPError as e2:
                    print(r(f"ETrade API HTTPError for historical {symbol} {interval}: {e2.response.status_code} {e2.response.text if e2.response else 'No response text'}"))
                    return pd.DataFrame()
            else:
                print(r(f"ETrade API HTTPError for historical {symbol} {interval}: {e.response.status_code} {e.response.text if e.response else 'No response text'}"))
                return pd.DataFrame()

        rows_data = []
        if "QuoteData" in js and js["QuoteData"]:
            if freq_type == "minute": # Check Intraday path
                if "Intraday" in js["QuoteData"] and js["QuoteData"]["Intraday"] and "Row" in js["QuoteData"]["Intraday"]:
                    rows_data = js["QuoteData"]["Intraday"]["Row"]
            elif freq_type != "minute": # Check All path for daily/weekly
                if "All" in js["QuoteData"] and js["QuoteData"]["All"] and "Row" in js["QuoteData"]["All"]:
                    rows_data = js["QuoteData"]["All"]["Row"]

        if not rows_data: return pd.DataFrame()

        try:
            df = pd.DataFrame([{
                "Date": pd.Timestamp(r["dateTimeUTC"] // 1000, unit='s', tz='UTC'),
                "Open": float(r["open"]), "High": float(r["high"]),
                "Low": float(r["low"]), "Close": float(r["close"]),
                "Volume": int(r["volume"])
            } for r in rows_data])
        except (TypeError, KeyError) as e: # Catch errors if row structure is unexpected
            print(r(f"Error parsing ETrade historical data for {symbol} {interval}: {e}. Sample: {rows_data[0] if rows_data else 'no rows'}"))
            return pd.DataFrame()

        if df.empty: return df
        df.set_index("Date", inplace=True)
        df.index = df.index.tz_convert(None) # Convert to local, naive
        if freq_type in ("daily", "weekly"): df.index = df.index.normalize() # Normalize to midnight for daily/weekly
        df = df.sort_index() # Ensure data is sorted by date
        return df.head(limit) if limit and not df.empty else df

# ── E*TRADE client (global singleton) ─────────────────────────────────
_ET_CLIENT = None # Initialize to None
try:
    # Ensure config.ini has [ETRADE] section alongside axist-sectors.py
    _ET_SECRETS = load_config("ETRADE")
    if _ET_SECRETS: # Check if load_config returned a non-empty dict
        _ET_CLIENT = ETradeClient(_ET_SECRETS)
        # Use existing g() for green color, assuming colorama is initialized
        print(g("ETradeClient initialized successfully from config.ini."))
    else:
        # Use existing r() for red color
        print(r("[CRITICAL] E*TRADE credentials from config.ini [ETRADE] section were not loaded properly (empty or False). ETradeClient not initialized."))
except FileNotFoundError as e: # This exception might not be directly raised by load_config if config.ini is missing, but kept for safety.
    print(r(f"[CRITICAL] E*TRADE configuration file ('Sectors/config.ini') not found: {e}."))
    print(r("Please ensure 'Sectors/config.ini' exists and has an [ETRADE] section with your API credentials."))
except KeyError as e: # Specific error for missing keys within [ETRADE] section
    print(r(f"[CRITICAL] Error in E*TRADE configuration ('Sectors/config.ini' [ETRADE] section): Missing key {e}."))
except Exception as e: # Catch any other unexpected errors during init
    print(r(f"[CRITICAL] Failed to initialize ETradeClient due to an unexpected error with config.ini: {e}"))
# ─────────────────────────────────────────────────────────────────────


def alpaca_download(symbol: str, *,
                    start: str | None = None,
                    end:   str | None = None,
                    timeframe: str = "1d", # This is a string e.g. "1d"
                    limit: int | None = None) -> pd.DataFrame:
    """Return a tz‑naïve OHLCV DataFrame from E*TRADE.""" # MODIFIED

    if _ET_CLIENT is None:
        print(r("ETradeClient not initialized. Cannot fetch E*TRADE data."))
        return pd.DataFrame()

    _inc_api_calls() # Increment API call counter
    try:
        # Assuming get_bars is compatible or will be adapted
        bars = _ET_CLIENT.get_bars(symbol, start=start, end=end, interval=timeframe, limit=limit)
    except Exception as e:
        print(r(f"ETrade download error for {symbol} {timeframe}: {e}"))
        return pd.DataFrame()

    if bars.empty:
        return bars

    # Assuming ETradeClient returns data in a compatible format or it's handled in ETradeClient
    # If 'symbol' column exists (from multi-symbol request), set it as index then droplevel
    # This part might need adjustment based on ETradeClient's actual output
    if 'symbol' in bars.columns: # This check might be E*TRADE specific or not needed
        bars = bars.set_index('symbol', append=True).swaplevel(0, 1)
        if isinstance(bars.index, pd.MultiIndex):
            try:
                bars = bars.loc[symbol]
            except KeyError:
                 return pd.DataFrame()

    # Standardize column names if necessary (ETradeClient might do this already)
    # bars.rename(columns={"open":"Open","high":"High","low":"Low",
    #                      "close":"Close","volume":"Volume"}, inplace=True, errors='ignore')

    # Ensure index is tz-naive (ETradeClient might do this already)
    # if bars.index.tz is not None:
    #    bars.index = bars.index.tz_localize(None)

    # Ensure required columns are present, or return empty if not (adjust as per ETradeClient output)
    # required_cols = ["Open", "High", "Low", "Close", "Volume"]
    # if not all(col in bars.columns for col in required_cols):
    #    print(r(f"Missing required OHLCV columns for {symbol} from E*TRADE. Got: {bars.columns.tolist()}"))
    #    return pd.DataFrame()

    return bars # Return as is, assuming ETradeClient.get_bars returns compatible DataFrame

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
    """Warm the on-disk cache with recent bars using the E*TRADE client (via alpaca_download wrapper).

    Intervals:
        15-min & 30-min : last 60 days
        1-hour          : last 120 days
        4-hour          : last 380 days   (approx 18 m)
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
        # MODIFIED: Use _ET_CLIENT for quotes
        if _ET_CLIENT:
            try:
                _inc_api_calls() # Increment API call counter
                price_map = _ET_CLIENT.get_quotes(unique_symbols)
                # Handle NaN values if get_quotes might return them for some symbols
                for sym in unique_symbols:
                    if sym not in price_map or pd.isna(price_map[sym]):
                        # Fallback or error handling for symbols not found or NaN price
                        print(r(f"Warning: Could not retrieve E*TRADE price for {sym}. Using entry price as fallback."))
                        price_map[sym] = next(p["entry_price"] for p in open_pos if p["symbol"] == sym)
            except Exception as e: # Broad exception for ETradeClient issues
                print(r(f"ETradeClient error in _update_positions_status: {e}"))
                # Fallback to entry price for all symbols if E*TRADE call fails
                for sym_needed in unique_symbols:
                    price_map[sym_needed] = next(p["entry_price"] for p in open_pos if p["symbol"] == sym_needed)
        else:
            print(r("ETradeClient not initialized. Falling back to entry prices for open positions."))
            for sym_needed in unique_symbols:
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
    """Disk cache for OHLCV data; network fetch now uses E*TRADE client via alpaca_download wrapper."""
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

    def _manual_atr(high, low, close, win): # Changed to accept win parameter
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
                            window, original_version=False) # Pass window to KeltnerChannel
        df[f"KC_upper_{timeframe}"]  = kc.keltner_channel_hband()
        df[f"KC_lower_{timeframe}"]  = kc.keltner_channel_lband()
        df[f"KC_middle_{timeframe}"] = (
            kc.keltner_channel_mavg() if hasattr(kc, "keltner_channel_mavg")
            else kc.keltner_channel_mband()
        )
    else:
        atr = _manual_atr(df["High"], df["Low"], df["Close"], window) # Pass window to _manual_atr
        ema = df["Close"].ewm(span=window, adjust=False).mean()
        rng = atr * 1.5 # atr here is now a Series
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

            atr_sum = _manual_atr(df["High"], df["Low"], df["Close"], window) * window # Pass window
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

        atr_sum = _manual_atr(df["High"], df["Low"], df["Close"], window) * window # Pass window
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
    if timeframe in {"daily", "hourly", "weekly", "1wk"}: # Corrected timeframe check
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
        objective="multi:softprob", # Corrected from multi:softmax to multi:softprob
        num_class=3,
        tree_method="hist", # Ensure hist is used if device is not explicitly cuda
        eval_metric="mlogloss", # Corrected from mlogloss to mlogloss
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

    # Deduplicate index to avoid reindex errors
    if df_1d.index.duplicated().any():
        df_1d = df_1d.loc[~df_1d.index.duplicated(keep='first')].copy()

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
    ind_1wk = compute_indicators(df_1wk.copy(), timeframe='1wk') # Corrected from 'weekly' to '1wk'

    # This function enrich_higher_timeframes was not defined in the original script.
    # Assuming it's meant to be a placeholder or defined elsewhere.
    # For now, let's create a passthrough or simple version to avoid NameError.
    def enrich_higher_timeframes(df): return df # Placeholder
    ind_htf = enrich_higher_timeframes(ind_1d.copy()) # Pass copy to avoid modifying ind_1d

    daily_15m = to_daily(ind_15m, "15m").add_suffix('_15m')
    daily_30m = to_daily(ind_30m, "30m").add_suffix('_30m')
    daily_1h  = to_daily(ind_1h,  "1h").add_suffix('_1h') # Corrected from "hourly"
    daily_4h  = to_daily(ind_4h, "4h").add_suffix('_4h')

    ind_1wk = ind_1wk.reindex(ind_1d.index, method='ffill')

    ind_1d.index.name = 'Date' # Ensure index name is set for joins
    features_df = (
        ind_1d
          .join(ind_htf, rsuffix='_htf') # Added rsuffix to avoid column name clashes
          .join(ind_1wk, rsuffix='_1wk') # Added rsuffix
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
    # Ensure ticker is not None before using it as a key
    bcol_key = ticker if ticker else "" # Default to empty string if ticker is None
    bcol = bench.get(bcol_key, 'SPY') # Use bcol_key
    if bcol in features_df.columns and 'Close' in features_df.columns: # Ensure 'Close' is present
        features_df[f'RS_{ticker if ticker else "TICKER"}'] = ( # Handle None ticker for column name
            features_df['Close'].pct_change(10) -
            features_df[bcol].pct_change(10)
        )

    features_df.dropna(subset=['Close'], inplace=True)
    if features_df.empty:
        return features_df

    # Ensure 'ATR_daily' exists before using it for labels
    if 'ATR_daily' not in features_df.columns:
        # Fallback: calculate ATR if missing, or use a default small value
        if 'High' in features_df and 'Low' in features_df and 'Close' in features_df:
             features_df['ATR_daily'] = AverageTrueRange(features_df['High'], features_df['Low'], features_df['Close'], window=14).average_true_range().fillna(0)
        else:
             features_df['ATR_daily'] = 0.001 # A small non-zero default if columns missing

    labels = triple_barrier_labels(
        features_df['Close'],
        features_df['ATR_daily'].fillna(0), # Ensure NaNs are filled
        horizon=horizon,
        stop_mult=2.0, tgt_mult=2.0
    )
    features_df['future_class'] = labels

    if drop_recent:
        features_df = features_df.iloc[:-horizon]
    else:
        # Ensure index exists before trying to assign NaN using .loc
        if len(features_df) >= horizon:
            features_df.loc[features_df.index[-horizon:], 'future_class'] = np.nan
        elif len(features_df) > 0 : # if less than horizon rows exist, set all to nan
            features_df.loc[:, 'future_class'] = np.nan


    return features_df

def refine_features(features_df,
                    importance_cutoff: float = 1e-4,
                    corr_threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop low-information and highly-correlated columns using a quick
    XGBoost probe – the probe runs entirely on the GPU.
    """
    if features_df.empty or 'future_class' not in features_df.columns or features_df['future_class'].isnull().all(): # Added isnull check
        return features_df

    y = features_df['future_class'].dropna() # Drop NaN labels for training
    X = features_df.loc[y.index].drop(columns=['future_class']).ffill().bfill()
    y = y.astype(int) # Ensure labels are integers

    # Fallback to CPU if CUDA is not available or causes issues
    try:
        model = XGBClassifier(
            objective='multi:softmax', # Corrected from multi:softprob
            num_class=3,
            eval_metric='mlogloss', # Corrected from mlogloss
            tree_method='hist',
            device='cuda' # Specify CUDA device
        )
        model.fit(X.iloc[:1], y.iloc[:1]) # Try fitting with a tiny bit of data to check CUDA
    except Exception: # Fallback to CPU
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            tree_method='hist',
            n_jobs=-1 # Use all available CPU cores
        )

    split = int(0.8 * len(X))
    if len(np.unique(y[:split])) < 2 or split == 0 : # Added split == 0 check
        return features_df.drop(columns=['future_class']).join(y, how='right')


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
        print(Fore.RED + "[Error] 'Close' column missing from input to refine_features and could not be restored." + Style.RESET_ALL)

    return X.join(y, how='right') # Use how='right' to keep all labels from y


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
    # Ensure pnl.std(ddof=1) is not zero before division
    sharpe   = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252) if pnl.std(ddof=1) and pnl.std(ddof=1) != 0 else 0.0


    equity = (1 + pnl).cumprod()
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    max_dd = abs(dd.min()) if dd.size else 0.0

    # Ensure total_bars is not zero
    tim = (trades["exit_timestamp"] - trades["entry_timestamp"]) \
            .dt.total_seconds().sum() / (total_bars * 60 * 30) if total_bars > 0 else 0.0
    tim = min(max(tim, 0), 1)

    return dict(total=len(trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                sharpe=sharpe,
                max_dd=max_dd,
                tim=tim)


def train_stacked_ensemble(features_df: pd.DataFrame,
                           *, random_state: int = 42):

    if features_df.empty or "future_class" not in features_df.columns or features_df['future_class'].isnull().all():
        return None, None

    y = features_df["future_class"].dropna().astype(int) # Ensure y is 1D and integer, drop NaNs
    X = features_df.loc[y.index].drop(columns=["future_class"]).ffill().bfill()

    if X.empty or len(X) < 5: # Need enough samples for TimeSeriesSplit
        print(r("Not enough data to train stacked ensemble after pre-processing."))
        return None, None


    best_xgb_params = tune_xgb_hyperparams(X, y, n_iter=20, # Reduced n_iter for speed
                                           random_state=random_state)
    xgb_clf = XGBClassifier(**best_xgb_params,
                            objective="multi:softprob", # Ensure this is softprob for predict_proba
                            num_class=3,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            n_jobs=-1,
                            random_state=random_state)

    rf_clf  = RandomForestClassifier(
        n_estimators=200, max_depth=10, # Reduced complexity
        class_weight="balanced", n_jobs=-1,
        random_state=random_state
    )
    log_clf = LogisticRegression(
        max_iter=500, class_weight="balanced", # Reduced max_iter
        n_jobs=-1, solver="saga", random_state=random_state
    )

    stack = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
        final_estimator=log_clf,
        stack_method="predict_proba",
        n_jobs=-1, # Can be parallelized
        passthrough=False # False is generally better unless features are very diverse
    )

    tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for speed
    best_thr, best_f1 = 0.60, -np.inf

    for train_idx, val_idx in tscv.split(X):
        if len(train_idx) < 2 or len(val_idx) < 2: continue # Skip if splits are too small
        stack.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = stack.predict_proba(X.iloc[val_idx])

        for thr_val in np.arange(0.55, 0.91, 0.05): # Renamed thr to thr_val
            pred = np.where(
                proba[:, 2] >= thr_val, 2,
                np.where(proba[:, 0] >= thr_val, 0, 1)
            )
            mask = pred != 1
            if mask.sum() == 0:
                continue
            # Ensure y.iloc[val_idx][mask] and pred[mask] are not empty and have compatible types/labels
            f1 = f1_score(y.iloc[val_idx][mask], pred[mask],
                          labels=[0, 2], average="macro", zero_division=0) # Added zero_division
            if f1 > best_f1:
                best_f1, best_thr = f1, thr_val

    stack.fit(X, y) # Fit on full data before returning
    return stack, best_thr

def generate_signal_output(ticker, latest_row, model, thr, macro_latest):
    # Ensure latest_row is a DataFrame for predict_proba
    if isinstance(latest_row, pd.Series):
        latest_df = latest_row.to_frame().T
    else:
        latest_df = latest_row

    # Ensure columns are in the same order as training
    if hasattr(model, 'feature_names_in_'):
        latest_df = latest_df[model.feature_names_in_]

    probs = model.predict_proba(latest_df.values)[0] # .values to ensure numpy array
    class_idx = int(np.argmax(probs)) # Ensure class_idx is int
    prob_score = probs[class_idx]

    if prob_score < thr or class_idx == 1: # Use thr passed to function
        return None

    direction = "LONG" if class_idx == 2 else "SHORT"
    price = float(latest_row.get("Close", np.nan)) # Ensure price is float

    reg = latest_row.get('REGIME', 0) # Default to 0 if REGIME is not present
    if reg == -1 and direction == "LONG":
        return None
    if reg == 1 and direction == "SHORT":
        return None

    atr = latest_row.get("ATR_daily", np.nan) # Ensure atr is float
    if not atr or np.isnan(atr) or atr == 0: # Added atr == 0 check
        atr = 0.05 * price # Default ATR
        atr_note = " (est. ATR)"
    else:
        atr_note = ""

    stop   = price - atr if direction == "LONG" else price + atr
    target = price + 2 * atr if direction == "LONG" else price - 2 * atr # Ensure target is float

    ema50  = latest_row.get('EMA50_daily', np.nan)
    ema200 = latest_row.get('EMA200_daily', np.nan)

    # Handle NaN in EMAs for trend calculation
    if pd.isna(ema50) or pd.isna(ema200):
        trend = "unknown"
    else:
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

    • Uses E*TRADE (via alpaca_download wrapper) for liquid ETF proxies (SPY, TIP, UUP, USO, HYG, TLT, etc.).
    • Optional FRED add-ons (10-yr yield, 2-10 spread) still supported.
    • Caches the merged result per-day on disk.
    """
    from pathlib import Path
    import joblib, datetime as _dt # _dt to avoid conflict

    Path(cache_dir).mkdir(exist_ok=True)
    # Use pd.Timestamp to ensure consistent date format for cache file name
    end_date_str = pd.Timestamp(end).strftime('%Y-%m-%d')
    cache_file = Path(cache_dir) / f"macro_{end_date_str}.pkl"

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
    # Ensure start and end are valid date strings for alpaca_download
    start_str = pd.Timestamp(start).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(end).strftime('%Y-%m-%d')

    for etf, col in etf_map.items():
        try:
            # Pass timeframe="1d" explicitly
            df = alpaca_download(etf, timeframe="1d", start=start_str, end=end_str)
            if not df.empty and 'Close' in df.columns:
                 price_df[col] = df["Close"]
            else:
                print(f"[WARN] No 'Close' data for {etf} in macro fetch.")
        except Exception as e:
            print(f"[WARN] Macro fetch failed for {etf}: {e}")

    fred_df = pd.DataFrame(index=price_df.index if not price_df.empty else None) # Handle empty price_df
    if fred_api_key:
        from fredapi import Fred # Ensure Fred is imported if used
        fred = Fred(api_key=fred_api_key)

        def _fred(code):
            # Ensure observation_end is a string
            return (fred.get_series(code,
                                     observation_start=start_str, # Use consistent start string
                                     observation_end=end_str) # Use consistent end string
                        .resample('D').ffill())

        try:
            fred_df['YIELD10']      = _fred('DGS10')
            fred_df['YIELD_SPREAD'] = _fred('T10Y2Y')
        except Exception as e:
            print(f"[WARN] FRED fetch failed: {e}")

    if price_df.empty and fred_df.empty:
         macro = pd.DataFrame(index=pd.date_range(start_str, end_str, freq='D'))
    elif fred_df.empty:
        macro = price_df
    elif price_df.empty:
        macro = fred_df
    else:
        macro = pd.concat([price_df, fred_df], axis=1)


    macro = macro.asfreq('D').ffill() # Fill missing daily data

    reg = pd.Series(0, index=macro.index, dtype=int)
    if 'VIX' in macro.columns: # Check if VIX column exists
        reg += (macro['VIX'] > 25).astype(int) * -1
    if 'YIELD_SPREAD' in macro.columns: # Check if YIELD_SPREAD column exists
        reg += (macro['YIELD_SPREAD'] > 0).astype(int)
    if 'USD' in macro.columns and not macro['USD'].empty: # Check if USD column exists and is not empty
        reg += (macro['USD'].pct_change(20) < 0).fillna(False).astype(int)
    macro['REGIME'] = reg.clip(-1, 1)

    try:
        joblib.dump(macro, cache_file)
    except Exception:
        pass # Avoid crashing if cache dump fails
    return macro.loc[start_str:end_str] # Return data within the requested range

def backtest_strategy(ticker: str,
                      start_date: str,
                      end_date:   str,
                      macro_df:  pd.DataFrame,
                      log_file=None): # log_file is passed but not used in this function
    """
    Daily-lumped back-test using the six-frame fetch & stacked ensemble.
    """
    df_15m, df_30m, df_1h, df_4h, df_1d, df_1wk = fetch_data(
        ticker, start=start_date, end=end_date
    )

    feat = prepare_features(df_15m, df_30m, df_1h, df_4h,
                            df_1d,  df_1wk, macro_df, ticker=ticker, drop_recent=True) # Ensure drop_recent=True for backtesting

    if feat.empty or 'future_class' not in feat.columns or feat['future_class'].isnull().all():
        print(Fore.YELLOW + f"No features or labels for {ticker} in backtest range. Skipping." + Style.RESET_ALL)
        return

    model, best_thr = train_stacked_ensemble(feat) # Use best_thr from training
    if model is None:
        print(Fore.YELLOW + f"Model training failed for {ticker}. Skipping backtest." + Style.RESET_ALL)
        return

    # Re-prepare features for the full period if needed for prediction, or use existing `feat`
    # For simplicity, this example assumes `feat` is sufficient for prediction after training.
    # Ensure 'ATR_daily' and 'Close' are present for trade logic
    if 'ATR_daily' not in feat.columns or 'Close' not in feat.columns:
        print(r(f"Missing ATR_daily or Close in features for {ticker}. Cannot execute trades."))
        return

    feat = feat.dropna(subset=["ATR_daily", "Close"]) # Drop rows where these are NaN
    if feat.empty:
        print(r(f"No valid data rows after dropping NaN ATR/Close for {ticker}."))
        return

    X_all = feat.drop(columns=["future_class"], errors='ignore').ffill().bfill() # errors='ignore'
    # Align columns with model's expected features
    if hasattr(model, 'feature_names_in_'):
         X_all = X_all[model.feature_names_in_]

    preds = model.predict(X_all)
    prob  = model.predict_proba(X_all)

    df_pred = feat.copy() # Use a copy
    df_pred['prediction'] = preds
    df_pred['probability'] = prob.max(axis=1) # Store max probability for the predicted class


    trades, in_trade = [], False
    for i, (ts, row) in enumerate(df_pred.iterrows()): # Use df_pred here
        price, atr = row["Close"], row["ATR_daily"]

        # Ensure atr is a valid number
        if pd.isna(atr) or atr == 0:
            # print(f"Skipping trade at {ts} for {ticker} due to invalid ATR: {atr}")
            continue

        current_pred = row.prediction # Prediction for current bar
        current_prob = row.probability # Probability of that prediction

        if not in_trade and current_prob >= best_thr and current_pred != 1: # Use best_thr
            dir_ = "LONG" if current_pred == 2 else "SHORT"
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

    if log_file and trades: # Check if log_file is provided and there are trades
        pd.DataFrame(trades).to_csv(log_file, mode='a', header=not log_file.tell())


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

    daily = to_daily(df_30m, "intraday") # Label was "intraday", kept as is.
    daily_inds = compute_indicators(daily.copy(), timeframe='daily') # Use copy for daily indicators
    if 'AnchoredVWAP' not in daily_inds.columns and not daily_inds.empty : # Check before assigning
        daily_inds['AnchoredVWAP'] = compute_anchored_vwap(daily_inds)


    if daily_inds.index.tz is not None: # Use daily_inds
        daily_inds.index = daily_inds.index.tz_localize(None)
    if df_30m.index.tz is not None:
        df_30m.index = df_30m.index.tz_localize(None)

    # Ensure daily_inds has 'AnchoredVWAP' before joining
    # And use rsuffix for all daily columns to avoid clashes
    daily_filled = daily_inds.reindex(df_30m.index, method='ffill')
    df_30m = df_30m.join(daily_filled, rsuffix='_daily_ctx') # Changed rsuffix

    horizon_bars = 16 # 8 hours for 30-min bars
    multiplier = 2.0
    atr_col = 'ATR_intraday' # From compute_indicators with timeframe='intraday'
    if atr_col not in df_30m.columns:
        # Fallback if ATR_intraday is missing (should be created by compute_indicators)
        if 'High' in df_30m and 'Low' in df_30m and 'Close' in df_30m:
            df_30m[atr_col] = AverageTrueRange(df_30m['High'], df_30m['Low'], df_30m['Close'], window=14).average_true_range().fillna(0.001)
        else: # Ultimate fallback if OHLC not available
            df_30m[atr_col] = df_30m['Close'].rolling(14).std().fillna(0.001)


    up_barrier = df_30m['Close'] + multiplier * df_30m[atr_col]
    down_barrier = df_30m['Close'] - multiplier * df_30m[atr_col]
    future_class = np.ones(len(df_30m), dtype=int) # Default to class 1 (hold)
    closes = df_30m['Close'].values

    for i in range(len(df_30m) - horizon_bars):
        # Ensure ATR is not NaN or zero for barrier calculation
        if pd.isna(df_30m[atr_col].iloc[i]) or df_30m[atr_col].iloc[i] == 0:
            continue # Skip if ATR is invalid

        upper_lvl = up_barrier.iloc[i]
        lower_lvl = down_barrier.iloc[i]
        window_prices = closes[i + 1 : i + 1 + horizon_bars]

        # Check for valid levels
        if pd.isna(upper_lvl) or pd.isna(lower_lvl): continue

        above_idx = np.where(window_prices >= upper_lvl)[0]
        below_idx = np.where(window_prices <= lower_lvl)[0]

        if len(above_idx) == 0 and len(below_idx) == 0: # No barrier hit
            continue # Stays as class 1
        elif len(above_idx) > 0 and len(below_idx) > 0: # Both hit
            if above_idx[0] < below_idx[0]: # Upper hit first
                future_class[i] = 2
            else: # Lower hit first
                future_class[i] = 0
        elif len(above_idx) > 0: # Only upper hit
            future_class[i] = 2
        elif len(below_idx) > 0: # Only lower hit
            future_class[i] = 0

    df_30m['future_class'] = future_class
    df_30m = df_30m.iloc[:-horizon_bars] # Remove rows where future is unknown

    if macro_df is not None and not macro_df.empty:
        if macro_df.index.tz is not None:
            current_macro_df = macro_df.copy() # Use a copy for modification
            current_macro_df.index = current_macro_df.index.tz_localize(None)
        else:
            current_macro_df = macro_df

        macro_resampled = current_macro_df.reindex(df_30m.index, method='ffill')
        df_30m = df_30m.join(macro_resampled, how='left', rsuffix='_macro') # Added rsuffix

    return df_30m

def backtest_strategy_intraday(ticker, start_date, end_date, macro_df,
                               log_file=None):
    """
    30-minute intraday back-test with Sharpe, max-DD, time-in-market and ATR guard.
    """
    _, df_30m, _, _, _, _ = fetch_data(ticker, start=start_date, end=end_date)

    if df_30m.empty:
        print(Fore.YELLOW + f"No 30-minute data for {ticker} in range. Skipping intraday backtest." + Style.RESET_ALL)
        return

    feat = prepare_features_intraday(df_30m, macro_df)
    if feat.empty or 'future_class' not in feat.columns or feat['future_class'].isnull().all():
        print(Fore.YELLOW + f"No features or labels for {ticker} (intraday). Skipping." + Style.RESET_ALL)
        return

    model, best_thr = train_stacked_ensemble(feat) # Use best_thr from training
    if model is None:
        print(Fore.YELLOW + f"Intraday model training failed for {ticker}. Skipping." + Style.RESET_ALL)
        return

    # Ensure 'ATR_intraday' and 'Close' are present for trade logic
    if 'ATR_intraday' not in feat.columns or 'Close' not in feat.columns:
        print(r(f"Missing ATR_intraday or Close in intraday features for {ticker}. Cannot execute trades."))
        return

    feat = feat.dropna(subset=['ATR_intraday', 'Close'])
    if feat.empty:
        print(r(f"No valid data rows after dropping NaN ATR/Close for {ticker} (intraday)."))
        return


    X_all   = feat.drop(columns=['future_class'], errors='ignore').ffill().bfill()
    if hasattr(model, 'feature_names_in_'):
         X_all = X_all[model.feature_names_in_]

    preds   = model.predict(X_all)
    probas  = model.predict_proba(X_all)

    df_pred = feat.copy() # Use copy
    df_pred['prediction'] = preds
    df_pred['probability'] = probas.max(axis=1)


    trades = []
    in_trade = False
    trade_dir = None # Explicitly initialize
    entry_price = stop_price = target_price = None # Explicitly initialize
    entry_time  = None # Explicitly initialize


    for i in range(len(df_pred)): # Iterate using index for safety with iloc
        ts        = df_pred.index[i]
        row       = df_pred.iloc[i]
        price     = row['Close']
        atr       = row.get('ATR_intraday', np.nan) # Use .get for safety

        if pd.isna(atr) or atr == 0:
            # print(f"Skipping trade at {ts} for {ticker} (intraday) due to invalid ATR: {atr}")
            continue # Skip if ATR is invalid

        if in_trade:
            # Ensure entry_price, stop_price, target_price are not None
            if entry_price is None or stop_price is None or target_price is None:
                in_trade = False # Should not happen if logic is correct, but as a safeguard
                continue

            if trade_dir == "LONG":
                if price >= target_price or price <= stop_price:
                    pnl = (price - entry_price) / entry_price if entry_price != 0 else 0
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
                    pnl = (entry_price - price) / entry_price if entry_price != 0 else 0
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
            if in_trade: # If still in trade, continue to next bar
                continue

        # Entry logic
        current_pred  = int(row['prediction']) # Ensure int
        prob_score = row['probability'] # From df_pred

        if prob_score >= best_thr and current_pred !=1: # Use best_thr
            if current_pred == 2:      # Buy signal
                in_trade   = True
                trade_dir  = "LONG"
                entry_time = ts
                entry_price= price
                stop_price = price - atr
                target_price = price + 2 * atr
            elif current_pred == 0:    # Sell signal
                in_trade   = True
                trade_dir  = "SHORT"
                entry_time = ts
                entry_price= price
                stop_price = price + atr
                target_price = price - 2 * atr

    if log_file and trades:
        pd.DataFrame(trades).to_csv(log_file, mode='a', header=not log_file.tell())


    if not trades:
        print(Fore.YELLOW + f"No trades for {ticker} intraday." + Style.RESET_ALL)
        return

    trades_df = pd.DataFrame(trades)
    summary   = _summarise_performance(trades_df, len(feat))

    pct = lambda x: f"{x*100:.2f}%"
    print(
        Fore.BLUE
        + f"\nIntraday Backtest Results for {ticker}"
        + Style.RESET_ALL
        + f" ({start_date} → {end_date}):"
    )
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

def run_signals_on_watchlist(use_intraday: bool = True): # Defaulting to True as per --live-real
    fred_api_key = load_config() # Ensure FRED key is loaded if needed by macro_df
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    # Consider preloading cache if alpaca_download is still used by get_macro_data or directly
    # preload_interval_cache(tickers) # This uses alpaca_download

    today = datetime.date.today()
    # Ensure macro_df is fetched correctly
    macro_df = get_macro_data(
        (today - datetime.timedelta(days=380)).strftime("%Y-%m-%d"), # Ensure 380 days for daily indicators
        today.strftime("%Y-%m-%d"),
        fred_api_key=fred_api_key
    )

    for tkr in tickers:
        print(f"\n=== {tkr} (live) ===")
        try:
            # Fetch data using the new ETrade client via alpaca_download wrapper
            df15, df30, df1h, df4h, df1d, df1w = fetch_data(tkr)
        except Exception as e:
            print(r(f"Fetch error for {tkr}: {e}")); continue # Use r() for red

        if use_intraday:
            feats = prepare_features_intraday(df30, macro_df)
        else:
            feats = prepare_features(df15, df30, df1h, df4h, df1d, df1w, macro_df, ticker=tkr, drop_recent=False) # drop_recent=False for live

        if feats.empty or 'future_class' not in feats.columns:
            print(Fore.YELLOW + f"Not enough features to generate signal for {tkr}." + Style.RESET_ALL); continue

        # For live signals, we usually don't have 'future_class' to train on the very latest data point.
        # So, train on data up to n-1, predict on n.
        # Or, use a pre-trained model if available and suitable.
        # Current logic re-trains model for each ticker using available history.

        # If drop_recent=False, last row's future_class will be NaN. Handle for training.
        train_feats = feats.dropna(subset=['future_class'])
        if train_feats.empty:
            print(Fore.YELLOW + f"Not enough training data (after NaN drop) for {tkr}." + Style.RESET_ALL); continue

        model, best_thr = train_stacked_ensemble(train_feats) # best_thr is important here
        if model is None:
            print(r(f"Model training failed for {tkr}.")); continue # Use r()

        # Prepare the latest row for prediction (should not have future_class)
        latest_row_features = feats.drop(columns='future_class', errors='ignore').iloc[-1]

        # Ensure macro_latest is correctly obtained for the latest date
        macro_latest = {}
        if not macro_df.empty and feats.index[-1] in macro_df.index:
             macro_latest = macro_df.loc[feats.index[-1]].to_dict()
        elif not macro_df.empty: # Fallback to most recent macro data if exact date match fails
            macro_latest = macro_df.iloc[-1].to_dict()


        sig = generate_signal_output(
            tkr,
            latest_row_features, # Pass the series/row
            model,
            best_thr, # Use the threshold from training
            macro_latest
        )
        print(sig or "No actionable signal.")

def backtest_watchlist():
    """Runs back-tests on each watch-list ticker with a pre-warmed cache."""
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    start_arg = input("Enter backtest start date (YYYY-MM-DD): ").strip()
    end_arg   = input("Enter backtest end date   (YYYY-MM-DD): ").strip()
    try:
        # Validate dates
        pd.Timestamp(start_arg)
        pd.Timestamp(end_arg)
    except ValueError:
        print(r("Invalid date format. Please use YYYY-MM-DD.")); return

    # preload_interval_cache(tickers) # Uses alpaca_download

    fred_api_key = load_config() # Load FRED key if needed for macro data
    macro_df = get_macro_data(start_arg, end_arg, fred_api_key=fred_api_key) # Pass key
    for ticker in tickers:
        print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
        # Assuming backtest_strategy is adapted for ETradeClient or still uses alpaca_download correctly
        backtest_strategy(ticker, start_arg, end_arg, macro_df)

def show_signals_for_current_week():
    """Print first actionable signal (if any) for each watch-list name."""
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty.")
        return

    today   = datetime.date.today()
    monday  = today - datetime.timedelta(days=today.weekday())
    start_s = (monday - datetime.timedelta(days=380)).strftime("%Y-%m-%d") # Enough history for indicators
    end_s   = today.strftime("%Y-%m-%d")

    macro_df = get_macro_data(start_s, end_s, fred_api_key=fred_api_key)

    for tkr in tickers:
        print(f"\n=== {tkr}: {monday.strftime('%Y-%m-%d')} → {end_s} ===")
        try:
            df15, df30, df1h, df4h, df1d, df1w = fetch_data(
                tkr, start=start_s, end=end_s # Fetch full history for this week's signals
            )
        except Exception as e:
            print(r(f"Fetch error for {tkr}: {e}")); continue

        # Prepare features for the whole period, don't drop recent for signal generation part
        all_feat = prepare_features(
            df15, df30, df1h, df4h, df1d, df1w,
            macro_df, drop_recent=False, ticker=tkr
        )
        if all_feat.empty or 'future_class' not in all_feat.columns:
            print(Fore.YELLOW + f"Insufficient data to process signals for {tkr}. Skipping." + Style.RESET_ALL)
            continue

        # Train model on data excluding the very last part where future_class might be NaN
        train_feat = all_feat.dropna(subset=['future_class'])
        if train_feat.empty:
             print(Fore.YELLOW + f"Not enough training data for {tkr} this week. Skipping." + Style.RESET_ALL)
             continue
        model, best_thr = train_stacked_ensemble(train_feat)
        if model is None:
            print(r(f"Model training failed for {tkr} this week.")); continue

        # Iterate from Monday of current week to today for signals
        # Ensure feature rows passed to generate_signal_output are Series
        target_period_feats = all_feat.loc[monday.strftime('%Y-%m-%d') : end_s]
        if target_period_feats.empty:
            print(f"No data for {tkr} in the current week range {monday.strftime('%Y-%m-%d')} to {end_s}"); continue

        for dt_idx, row_series in target_period_feats.iterrows():
            # Ensure macro_latest is correctly obtained for the specific date
            macro_latest_for_dt = {}
            if not macro_df.empty and dt_idx in macro_df.index:
                 macro_latest_for_dt = macro_df.loc[dt_idx].to_dict()
            elif not macro_df.empty: # Fallback
                macro_latest_for_dt = macro_df.iloc[-1].to_dict()

            sig = generate_signal_output(tkr, row_series.drop('future_class', errors='ignore'), model, best_thr, macro_latest_for_dt)
            if sig:
                print(f"{dt_idx.date()}: {sig}")
                break # Show first signal of the week
        else: # If no signal found in the loop
            print("No actionable signal found for this week.")


def signals_performance_cli():
    """
    Dashboard of OPEN trades – price updates fetched using ETradeClient.
    """
    open_recs = [p for p in load_predictions() if p["status"] == "Open"]
    if not open_recs:
        print("No open positions. Run option 5 (Show This Weeks Signals) first to generate some."); return

    def _get_last_prices_etrade() -> dict[str, float]: # Renamed for clarity
        syms = sorted(list(set(p["symbol"] for p in open_recs)))
        # Ensure syms is list
        if not syms: return {}

        if _ET_CLIENT:
            try:
                _inc_api_calls() # Increment API call counter
                prices = _ET_CLIENT.get_quotes(syms)
                # Handle NaNs: if E*TRADE returns NaN, keep it to show data issue, or fallback
                valid_prices = {s: p for s, p in prices.items() if not pd.isna(p)}
                if len(valid_prices) < len(syms):
                    print(r(f"Warning: Could not get E*TRADE prices for all symbols. Missing: {set(syms) - set(valid_prices.keys())}"))
                # Fallback for missing symbols (though get_quotes should ideally handle this)
                for s_missed in set(syms) - set(valid_prices.keys()):
                    valid_prices[s_missed] = next((rec["entry_price"] for rec in open_recs if rec["symbol"] == s_missed), np.nan)
                return valid_prices
            except Exception as e:
                print(r(f"ETradeClient error in _get_last_prices_etrade: {e}"))
        else:
            print(r("ETradeClient not initialized. Cannot fetch live prices."))
        # Fallback if ETrade client fails or not initialized
        return {p["symbol"]: p["entry_price"] for p in open_recs}


    # ... (rest of the urwid table building logic, adapted to print if urwid is not used) ...
    # This part heavily relies on urwid. For a non-urwid version, print to console.
    _table_cache_print = [] # For non-urwid version

    def _build_table_print(px: dict[str, float]):
        nonlocal _table_cache_print
        _table_cache_print = [] # Reset cache
        header = f"{'Symbol':<8}{'Dir':<6}{'Entry':>10}{'Now':>10}{'P/L%':>8}{'Stop':>10}{'Target':>10}{'Status':>12}{'Date':>12}"
        print(b(header)) # Blue for header
        print(b("-" * len(header)))

        today_str = datetime.date.today().strftime("%Y-%m-%d")

        for rec in open_recs:
            sym, ep = rec["symbol"], rec["entry_price"]
            now = float(px.get(sym, ep)) # Default to entry price if 'now' is not available

            pnl_pct = ((now - ep) / ep * 100) if rec["direction"] == "LONG" and ep != 0 else \
                      ((ep - now) / ep * 100) if rec["direction"] == "SHORT" and ep != 0 else 0

            hit_stop   = (now <= rec["stop_loss"])   if rec["direction"] == "LONG" else (now >= rec["stop_loss"])
            hit_target = (now >= rec["profit_target"]) if rec["direction"] == "LONG" else (now <= rec["profit_target"])
            status = "Stop" if hit_stop else "Target" if hit_target else "Open"

            pnl_color = g if pnl_pct >= 0 else r
            row_str = (f"{sym:<8}{rec['direction']:<6}{ep:>10.2f}{now:>10.2f}"
                       f"{pnl_color(f'{pnl_pct:>7.2f}%')}{rec['stop_loss']:>10.2f}"
                       f"{rec['profit_target']:>10.2f}{status:>12}{rec['entry_date']:>12}")
            print(row_str)
            _table_cache_print.append({'rec': rec, 'hit': (status != "Open"), 'status': status, 'now': now, 'pnl_pct': pnl_pct, 'today_str': today_str})
        print(b("-" * len(header)))


    def _refresh_print(*_):
        prices = _get_last_prices_etrade()
        _build_table_print(prices)

    _refresh_print() # Initial display

    while True:
        action = input("(R)efresh, (D)eject hit trades, (Q)uit: ").lower().strip()
        if action == 'q': break
        elif action == 'r': _refresh_print()
        elif action == 'd':
            changed_in_loop = False
            # Need to load all predictions to modify them
            all_predictions_loaded = load_predictions()
            # Create a dictionary for faster lookups of open records that need updating
            # Key: (symbol, entry_date) assuming this is unique enough for open trades
            # Value: The modified record data from _table_cache_print

            modifications_map = {}
            for item in _table_cache_print:
                if item['hit']: # If the trade hit stop/target
                    original_rec = item['rec']
                    mod_key = (original_rec['symbol'], original_rec['entry_date'])
                    modifications_map[mod_key] = {
                        "status": item['status'],
                        "exit_price": round(item['now'], 2),
                        "exit_date": item['today_str'],
                        "pnl_pct": round(item['pnl_pct'], 2) # Ensure pnl_pct is also updated
                    }
                    changed_in_loop = True

            if changed_in_loop:
                updated_all_preds_list = []
                for pred_item in all_predictions_loaded:
                    current_key = (pred_item['symbol'], pred_item['entry_date'])
                    if pred_item['status'] == 'Open' and current_key in modifications_map:
                        pred_item.update(modifications_map[current_key])
                    updated_all_preds_list.append(pred_item)

                save_predictions(updated_all_preds_list)
                # Refresh open_recs for the current CLI view
                open_recs[:] = [p for p in updated_all_preds_list if p["status"] == "Open"]
                print(g("Hit trades ejected and predictions saved."))
            else:
                print("No trades were hit to eject.")
            _refresh_print() # Refresh display
        else: print(r("Invalid option."))


def closed_stats_cli():
    """
    Color-enhanced statistics for CLOSED positions in weekly_signals.json.
    Green  = favourable numbers, Red = unfavourable.
    """
    all_preds = load_predictions() # Load all predictions
    recs = [p for p in all_preds if p.get('status', 'Open') != 'Open'] # Use .get for safety

    if not recs:
        print("No closed trades recorded.")
        input("\nPress Enter to return …")
        return

    for r_item in recs:
        # Ensure pnl_pct is calculated if missing (e.g. for older records or manual closures)
        if 'pnl_pct' not in r_item or r_item['pnl_pct'] is None:
            ep = r_item.get('entry_price')
            xp = r_item.get('exit_price', ep) # Default exit to entry if not set (0 P/L)
            direction = r_item.get('direction')

            if ep is not None and xp is not None and direction is not None and ep != 0:
                if direction == 'LONG':
                    r_item['pnl_pct'] = ((xp - ep) / ep) * 100
                else: # SHORT
                    r_item['pnl_pct'] = ((ep - xp) / ep) * 100
                r_item['pnl_pct'] = round(r_item['pnl_pct'], 2)
            else:
                r_item['pnl_pct'] = 0.0 # Default to 0 P/L if data missing

    wins   = [r_item for r_item in recs if r_item.get('status') == 'Target' and r_item['pnl_pct'] > 0] # Target implies win
    losses = [r_item for r_item in recs if r_item.get('status') == 'Stop' or r_item['pnl_pct'] < 0] # Stop implies loss
    # Note: A trade might be closed manually (status 'Closed') and could be win or loss based on P/L

    total  = len(recs)
    win_rt = (len(wins) / total * 100) if total > 0 else 0.0

    # Calculate P/L only from records that have valid pnl_pct
    valid_pnls_wins = [w['pnl_pct'] for w in wins if 'pnl_pct' in w and isinstance(w['pnl_pct'], (int, float))]
    valid_pnls_losses = [l['pnl_pct'] for l in losses if 'pnl_pct' in l and isinstance(l['pnl_pct'], (int, float))]

    avg_win = np.mean(valid_pnls_wins) if valid_pnls_wins else 0.0
    avg_los = np.mean(valid_pnls_losses) if valid_pnls_losses else 0.0 # Losses are negative, so avg_los will be negative

    # Compounded return calculation
    equity = 1.0
    for r_item in recs:
        if 'pnl_pct' in r_item and isinstance(r_item['pnl_pct'], (int, float)):
            equity *= (1 + r_item['pnl_pct'] / 100)
    tot_ret = (equity - 1) * 100


    print(b("\nClosed-Position Statistics"))
    print(b("--------------------------------"))

    print(f"Total closed trades   : {b(str(total))}")
    print(f"Wins (hit target/pos) : {g(str(len(wins)))}" # Display count of wins
          f"  |  Avg gain : {g(f'{avg_win:+.2f}%') if wins else '--'}")
    print(f"Losses (hit stop/neg) : {r(str(len(losses)))}" # Display count of losses
          f"  |  Avg loss : {r(f'{avg_los:+.2f}%') if losses else '--'}") # avg_los is already negative

    win_color = g if win_rt >= 50 else r # Color based on win rate
    print(f"Win rate              : {win_color(f'{win_rt:.2f}%')}")

    tot_color = g if tot_ret >= 0 else r # Color based on total return
    print(f"Compounded return     : {tot_color(f'{tot_ret:+.2f}%')}")

    print(b("\nMost recent 10 closed trades:"))
    for rcd in recs[-10:]: # Iterate through the last 10 records
        pnl_val = rcd.get('pnl_pct', 0.0)
        pnl_str = f"{pnl_val:+.2f}%" # Format P/L string
        pnl_display_color = g if pnl_val >=0 else r
        print(f"{rcd.get('exit_date','N/A')} {rcd.get('symbol','N/A')} {rcd.get('direction','N/A')} {rcd.get('status','N/A')} PnL {pnl_display_color(pnl_str)}")


    if input(Fore.YELLOW + "\n(C)lear stats or Enter to return: " + Style.RESET_ALL).lower() == 'c':
        # Keep only 'Open' positions
        open_only_preds = [p for p in all_preds if p.get('status') == 'Open']
        save_predictions(open_only_preds)
        print(Fore.YELLOW + "Closed trades history cleared." + Style.RESET_ALL)
    input("\nPress Enter to return …") # Pause for user


def schedule_signals_instructions():
    script_path = Path(__file__).resolve()
    print("\nTo schedule signals (e.g., at 9:35 AM and 3:55 PM EST on weekdays):")
    print("1. Open your crontab for editing: `crontab -e`")
    print("2. Add lines like the following (adjust times and Python path as needed):")
    print(g(f"   35 9 * * 1-5 /usr/bin/python3 {script_path} $(cat {script_path.parent}/watchlist.json | jq -r '.[]' | tr '\\n' ' ') --live-real"))
    print(g(f"   55 15 * * 1-5 /usr/bin/python3 {script_path} $(cat {script_path.parent}/watchlist.json | jq -r '.[]' | tr '\\n' ' ') --live-real"))
    print("\nMake sure `jq` is installed (`sudo apt-get install jq`).")
    print("The watchlist tickers are dynamically read and passed to the script.")
    print("Logs will typically go to your system mail or as specified by cron.")


def interactive_menu():
    while True:
        # Removed API call count as it's Alpaca specific
        # print(f"\nAPI Calls Made: {get_api_call_count()}")
        print("\nMain Menu:")
        print("1. Manage Watchlist")
        print("2. Run Signals on Watchlist (Live Mode, Default Intraday)")
        print("3. Backtest All Watchlist Tickers (Daily)")
        print("4. Schedule Signals (Cron Instructions)")
        print("5. Show This Weeks Signals (Daily)")
        print("6. Show Latest Signals Performance (Open Trades)")
        print("7. Closed-Trades Statistics")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == '0': print("Exiting."); break
        elif choice == '1': manage_watchlist()
        elif choice == '2': run_signals_on_watchlist(use_intraday=True) # Explicitly True
        elif choice == '3': backtest_watchlist() # Daily backtest
        elif choice == '4': schedule_signals_instructions()
        elif choice == '5': show_signals_for_current_week() # Daily signals for week
        elif choice == '6': signals_performance_cli()
        elif choice == '7': closed_stats_cli()
        else: print(r("Invalid option. Please try again."))


def main():
    parser = argparse.ArgumentParser(description="Swing trading signal generator/backtester")
    parser.add_argument('tickers', nargs='*', help="List of stock ticker symbols to analyze")
    # --log is kept for potential future use, though not explicitly used by E*TRADE trade functions yet
    parser.add_argument('--log', action='store_true', help="Optionally log selected trades to a file (currently basic)")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode instead of live signal mode")
    parser.add_argument('--start', default=None, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument('--real', action='store_true', help="Use 30-min intraday backtest for max realism")
    parser.add_argument('--live-real', action='store_true',
                        help="Use 30‑min intraday pipeline in live mode (default for menu option 2)")
    args = parser.parse_args()

    # If no specific CLI args for tickers/mode, show interactive menu
    if not args.tickers and not args.backtest and not args.live_real :
        interactive_menu()
        return

    fred_api_key = load_config() # Load FRED key for macro data
    tickers_to_process = args.tickers
    if not tickers_to_process and not args.backtest : # If called by cron, tickers might be from watchlist directly
        # This part might need adjustment based on how cron passes tickers
        # For now, assume tickers are passed as arguments by cron as well
        loaded_wl = load_watchlist()
        if not loaded_wl:
            print(r("No tickers provided and watchlist is empty. Exiting."))
            return
        tickers_to_process = loaded_wl


    log_file_handle = None # Define log_file_handle to manage file opening/closing
    if args.log:
        log_file_handle = open("trades_log.csv", "a") # Open in append mode

    if args.backtest:
        if not args.start or not args.end:
            print(r("Backtest mode requires --start and --end dates."))
            if log_file_handle: log_file_handle.close()
            return

        macro_df = get_macro_data(args.start, args.end, fred_api_key=fred_api_key)
        backtest_func = backtest_strategy_intraday if args.real else backtest_strategy

        for ticker in tickers_to_process:
            print(f"\n=== {'Intraday' if args.real else 'Daily'} Backtesting {ticker} from {args.start} to {args.end} ===")
            # Pass the opened file handle to the backtest function
            backtest_func(ticker, args.start, args.end, macro_df, log_file=log_file_handle)
    else: # Live signal mode
        today_dt = datetime.date.today()
        macro_start_dt = today_dt - datetime.timedelta(days=380) # Ensure enough data for daily indicators
        macro_df = get_macro_data(
            macro_start_dt.strftime('%Y-%m-%d'),
            today_dt.strftime('%Y-%m-%d'),
            fred_api_key=fred_api_key
        )

        use_intraday_live = args.live_real # Determine if intraday should be used for live

        for ticker in tickers_to_process:
            print(f"\n=== Processing {ticker} (live signal mode{' - Intraday' if use_intraday_live else ''}) ===")
            try:
                df15, df30, df1h, df4h, df1d, df1w = fetch_data(ticker) # Uses ETrade via alpaca_download
            except Exception as e:
                print(r(f"Error fetching data for {ticker}: {e}"))
                continue

            if (use_intraday_live and df30.empty) or (not use_intraday_live and df1d.empty):
                print(r(f"No usable {'30m' if use_intraday_live else 'daily'} data for {ticker}, skipping."))
                continue

            if use_intraday_live:
                features_df = prepare_features_intraday(df30, macro_df)
            else: # Daily live signals
                features_df = prepare_features(df15, df30, df1h, df4h, df1d, df1w, macro_df, ticker=ticker, drop_recent=False)

            if features_df.empty or 'future_class' not in features_df.columns:
                print(Fore.YELLOW + f"Not enough features for {ticker}, skipping." + Style.RESET_ALL)
                continue

            train_feats_live = features_df.dropna(subset=['future_class'])
            if train_feats_live.empty:
                print(Fore.YELLOW + f"Not enough training data for live signal on {ticker}." + Style.RESET_ALL); continue

            model, best_thr = train_stacked_ensemble(train_feats_live)
            if model is None:
                print(r(f"Model training failed for {ticker} in live mode.")); continue

            latest_features_row = features_df.drop(columns='future_class', errors='ignore').iloc[-1]

            macro_latest_live = {}
            if not macro_df.empty and features_df.index[-1] in macro_df.index:
                 macro_latest_live = macro_df.loc[features_df.index[-1]].to_dict()
            elif not macro_df.empty:
                macro_latest_live = macro_df.iloc[-1].to_dict()


            signal_output = generate_signal_output(ticker, latest_features_row, model, best_thr, macro_latest_live)

            if signal_output:
                print(signal_output)
                if log_file_handle: # Check if file is open
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file_handle.write(f"{now_str},{ticker},{signal_output}\n")
                    log_file_handle.flush() # Ensure it's written immediately
            else:
                print("No actionable signal.")

    if log_file_handle: # Close the file if it was opened
        log_file_handle.close()


if __name__ == "__main__":
    main()
