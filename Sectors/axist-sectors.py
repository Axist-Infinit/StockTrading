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
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from yfinance.exceptions import YFRateLimitError
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



from colorama import init, Fore, Style
init(autoreset=True)
from fredapi import Fred 
import configparser
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

_MAX_QPS      = 2          # Yahoo lets ~2 download calls / second
_TOKENS       = _MAX_QPS
_LAST_REFILL  = time.time()
_LOCK         = threading.Lock()
YF_CALLS = 0  

PREDICTIONS_FILE = "weekly_signals.json"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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
                   interval="1d", max_tries=4):
    global YF_CALLS
    tries = 0
    while tries < max_tries:
        _wait_for_token()
        try:
            YF_CALLS += 1
            if period:
                df = yf.download(ticker, period=period, interval=interval,
                                 auto_adjust=True, progress=False)
                time.sleep(0.15)         # ← NEW gentle spacing
            else:
                df = yf.download(ticker, start=start, end=end,
                                 interval=interval, auto_adjust=True,
                                 progress=False)
            break
        except YFRateLimitError:
            wait = 2 ** tries
            print(f"[WARN] Yahoo 429 for {ticker} – retry in {wait}s…")
            time.sleep(wait)
            tries += 1
            continue
        except Exception as e:
            print(f"[WARN] Yahoo error {ticker}/{interval}: {e}")
            df = pd.DataFrame(); break

    if df.empty:
        return _polygon_download(ticker, start or "", end or "", interval)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

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

# ────────────────────────────────────────────────────────────────────
def preload_interval_cache(symbols: list[str],
                           *,
                           batch_size: int = 4,
                           pause_sec: float = 1.0) -> None:
    """
    Very-polite cache warmer:

    1. Splits the symbol list into batches of ≤ *batch_size* (4 by default)
    2. Sleeps *pause_sec* (1 s) after **every** yf.download call
    3. If Yahoo rate-limits the batch, falls back to per-symbol
       `_safe_download`, which itself:
           • respects the global 2 QPS token bucket
           • retries with exponential back-off
           • has a Polygon fallback

    This combination has proven safe with >100 tickers.
    """
    from yfinance.exceptions import YFRateLimitError
    import time, random, pandas as pd

    combos = [
        ('60d',  '15m'),
        ('60d',  '30m'),
        ('120d', '1h'),
        ('60d',  '90m'),
        ('380d', '1d'),
    ]
    symbols = sorted(set(symbols))
    if not symbols:
        return

    # split symbols into small, manageable chunks
    batches = [symbols[i:i + batch_size]
               for i in range(0, len(symbols), batch_size)]

    for period, ivl in combos:
        for batch in batches:
            try:
                bulk = yf.download(
                    batch, period=period, interval=ivl,
                    auto_adjust=True, progress=False, threads=False
                )
                time.sleep(pause_sec)                  # polite pause
            except YFRateLimitError:
                bulk = pd.DataFrame()                  # force fallback
            except Exception as e:
                print(f"[WARN] Bulk preload err {batch}/{ivl}: {e}")
                bulk = pd.DataFrame()

            # ---------- if bulk failed → per-symbol fallback ----------
            if bulk.empty:
                for sym in batch:
                    try:
                        df = _safe_download(sym, period=period,
                                            interval=ivl, max_tries=6)
                    except Exception:
                        continue
                    _save_cache(df, sym, ivl)
                continue

            # ---------- bulk succeeded → split & save -----------------
            if isinstance(bulk.columns, pd.MultiIndex):
                bulk = bulk.swaplevel(axis=1).sort_index(axis=1)

            for sym in batch:
                df = bulk[sym].dropna(how="all")
                if df.empty:
                    continue
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                _save_cache(df, sym, ivl)

def load_config():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.ini")
    config.read(config_path)

    return config['FRED'].get('api_key', None)

def _load_polygon_key() -> str | None:
    """Return Polygon.io API key from api.json (same folder as this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(script_dir, "api.json"), "r") as fh:
            cfg = json.load(fh)
        return (cfg.get("POLYGON_KEY")           # preferred
                or cfg.get("polygon")            # alias
                or cfg.get("api_key"))           # generic
    except Exception:
        return None

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
    Uses ONE bulk Yahoo request for all open trades instead of per-symbol
    .history() calls.  Everything else is unchanged.
    """
    preds = load_predictions()
    open_rd = [p for p in preds if p.get('status') == 'Open']
    if not open_rd:
        return

    # -------- ONE HTTPS request --------------------------------------
    try:
        px = (yf.download({p['symbol'] for p in open_rd},
                          period='1d', auto_adjust=True,
                          progress=False, threads=False)['Close']
              .iloc[-1]
              .to_dict())
    except Exception:
        # fallback – assume unchanged prices
        px = {p['symbol']: p['entry_price'] for p in open_rd}

    changed = False
    today   = datetime.date.today().isoformat()

    for p in open_rd:
        now_price = float(px.get(p['symbol'], p['entry_price']))
        if p['direction'] == 'LONG':
            hit_stop   = now_price <= p['stop_loss']
            hit_target = now_price >= p['profit_target']
        else:  # SHORT
            hit_stop   = now_price >= p['stop_loss']
            hit_target = now_price <= p['profit_target']

        if hit_stop or hit_target:
            p.update({
                'status'    : 'Stop' if hit_stop else 'Target',
                'exit_date' : today,
                'exit_price': round(now_price, 2)
            })
            changed = True

    if changed:
        save_predictions(preds)

def cached_download(ticker: str,
                    start: str,
                    end:   str,
                    interval: str) -> pd.DataFrame:
    """
    Return a DataFrame covering [start, end] (inclusive) using a local
    parquet cache.  Handles tz-aware / tz-naive mismatches internally.
    """
    cached = _load_cache(ticker, interval)

    if not cached.empty and cached.index.tz is not None:
        cached.index = cached.index.tz_localize(None)

    have_start = cached.index.min() if not cached.empty else None
    have_end   = cached.index.max() if not cached.empty else None

    need_download = (
        cached.empty
        or pd.to_datetime(start) < have_start
        or pd.to_datetime(end)   > have_end
    )

    if need_download:
        dl_start = start if cached.empty or pd.to_datetime(start) < have_start \
                   else (have_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        dl_end   = end

        new = _safe_download(ticker, start=dl_start, end=dl_end,
                             interval=interval)

        if not new.empty and new.index.tz is not None:
            new.index = new.index.tz_localize(None)

        df = (pd.concat([cached, new])
                .sort_index()
                .drop_duplicates())
        _save_cache(df, ticker, interval)
    else:
        df = cached

    return df.loc[start:end]

def fetch_data(ticker: str,
               start: str | None = None,
               end:   str | None = None,
               *,
               warmup_days: int = 300) -> tuple[pd.DataFrame, ...]:
    """
    Returns (df_15m, df_30m, df_1h, df_90m, df_1d, df_1wk)
    with Yahoo-retry + Polygon fall-back.
    """
    intraday_cap = {"15m": 60, "30m": 60, "1h": 730, "90m": 730}

    def cap(d: str, ivl: str) -> str:
        if not d: return d
        capd = intraday_cap.get(ivl)
        if not capd: return d
        lim = (pd.Timestamp(end or datetime.date.today()) -
               pd.Timedelta(days=capd)).strftime("%Y-%m-%d")
        return max(d, lim)

    if start and end:
        df_15  = _safe_download(ticker, start=cap(start, "15m"), end=end, interval="15m")
        df_30  = _safe_download(ticker, start=cap(start, "30m"), end=end, interval="30m")
        df_1h  = _safe_download(ticker, start=cap(start, "1h"),  end=end, interval="1h")
        df_90  = _safe_download(ticker, start=cap(start, "90m"), end=end, interval="90m")
        df_1d  = _safe_download(ticker,
                                start=(pd.to_datetime(start) -
                                       pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d"),
                                end=end, interval="1d")
        df_1wk = _safe_download(ticker, start=start, end=end, interval="1wk")
    else:
        df_15  = _safe_download(ticker, period="60d",  interval="15m")
        df_30  = _safe_download(ticker, period="60d",  interval="30m")
        df_1h  = _safe_download(ticker, period="120d", interval="1h")
        df_90  = _safe_download(ticker, period="60d",  interval="90m")
        df_1d  = _safe_download(ticker, period="380d", interval="1d")
        df_1wk = _safe_download(ticker, period="max",  interval="1wk")

    # normalise
    for df in (df_15, df_30, df_1h, df_90, df_1d, df_1wk):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    return df_15, df_30, df_1h, df_90, df_1d, df_1wk


def compute_indicators(df: pd.DataFrame,
                       timeframe: str = "daily") -> pd.DataFrame:
    df = df.copy()
    req = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not req.issubset(df.columns):
        return df                                    # nothing to do

    # ---- adaptive look-back ------------------------------------------
    pct_vol = (
        df["Close"].pct_change().rolling(20).std().iloc[-1]
        if len(df) >= 21 else
        df["Close"].pct_change().std()
    ) or 0.0

    window = 21 if pct_vol >= 0.06 else 10 if pct_vol <= 0.01 else 14

    if len(df) <= window:                           # e.g. 6 rows vs 21
        return pd.DataFrame(index=df.index)         # safe stub frame

    rsi  = RSIIndicator(df["Close"], window).rsi()
    adx  = ADXIndicator(df["High"], df["Low"], df["Close"], window)
    sto  = StochasticOscillator(df["High"], df["Low"], df["Close"],
                                window, smooth_window=3)
    cci  = CCIIndicator(df["High"], df["Low"], df["Close"], window, 0.015)
    macd = MACD(df["Close"], 26, 12, 9)
    bb   = BollingerBands(df["Close"], 20, 2)
    atr  = AverageTrueRange(df["High"], df["Low"], df["Close"], window)

    roc  = ROCIndicator(df["Close"], window)
    wlr  = WilliamsRIndicator(df["High"], df["Low"], df["Close"], window)
    mfi  = MFIIndicator(df["High"], df["Low"], df["Close"],
                        df["Volume"], window)
    obv  = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    cmf  = ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"],
                                     df["Volume"], window)
    kc   = KeltnerChannel(df["High"], df["Low"], df["Close"],
                          window, original_version=False)
    dc   = DonchianChannel(df["High"], df["Low"], df["Close"], window)

    df[f"RSI_{timeframe}"]         = rsi
    df[f"ADX_{timeframe}"]         = adx.adx()
    df[f"ADX_pos_{timeframe}"]     = adx.adx_pos()
    df[f"ADX_neg_{timeframe}"]     = adx.adx_neg()
    df[f"STOCHk_{timeframe}"]      = sto.stoch()
    df[f"STOCHd_{timeframe}"]      = sto.stoch_signal()
    df[f"CCI_{timeframe}"]         = cci.cci()
    df[f"MACD_{timeframe}"]        = macd.macd()
    df[f"MACD_signal_{timeframe}"] = macd.macd_signal()
    df[f"MACD_hist_{timeframe}"]   = macd.macd_diff()
    df[f"BB_upper_{timeframe}"]    = bb.bollinger_hband()
    df[f"BB_lower_{timeframe}"]    = bb.bollinger_lband()
    df[f"BB_middle_{timeframe}"]   = bb.bollinger_mavg()
    df[f"ATR_{timeframe}"]         = atr.average_true_range()
    df[f"ATR_pct_{timeframe}"]     = df[f"ATR_{timeframe}"] / df["Close"]

    df[f"ROC_{timeframe}"]         = roc.roc()
    df[f"WR_{timeframe}"]          = wlr.williams_r()
    df[f"MFI_{timeframe}"]         = mfi.money_flow_index()
    df[f"OBV_{timeframe}"]         = obv.on_balance_volume()
    df[f"CMF_{timeframe}"]         = cmf.chaikin_money_flow()

    df[f"KC_upper_{timeframe}"]  = kc.keltner_channel_hband()
    df[f"KC_lower_{timeframe}"]  = kc.keltner_channel_lband()
    df[f"KC_middle_{timeframe}"] = (
        kc.keltner_channel_mavg() if hasattr(kc, "keltner_channel_mavg")
        else kc.keltner_channel_mband()
    )

    df[f"DC_upper_{timeframe}"]  = dc.donchian_channel_hband()
    df[f"DC_lower_{timeframe}"]  = dc.donchian_channel_lband()
    df[f"DC_middle_{timeframe}"] = (
        dc.donchian_channel_mavg() if hasattr(dc, "donchian_channel_mavg")
        else dc.donchian_channel_mband()
    )

    if timeframe in {"daily", "hourly", "weekly", "1wk"}:
        df[f"SMA20_{timeframe}"] = SMAIndicator(df["Close"], 20).sma_indicator()
        df[f"SMA50_{timeframe}"] = SMAIndicator(df["Close"], 50).sma_indicator()
        df[f"EMA50_{timeframe}"] = EMAIndicator(df["Close"], 50).ema_indicator()
        df[f"EMA200_{timeframe}"]= EMAIndicator(df["Close"], 200).ema_indicator()

    return df

def enrich_higher_timeframes(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add higher-time-frame (weekly, monthly, quarterly) technical indicators
    and basic cross-time-frame ratios.

    Returns a DataFrame *aligned to daily index* with suffixes:
        _wk   – weekly
        _mth  – month-end
        _qtr  – quarter-end
    """
    tf_map = {'W-FRI': 'wk', 'ME': 'mth', 'QE': 'qtr'}
    out = []

    for freq, tag in tf_map.items():
        tf_df = compute_indicators(
            df_daily.resample(freq).last(), timeframe=tag
        ).add_suffix(f"_{tag}")
        tf_df = tf_df.reindex(df_daily.index, method='ffill')
        out.append(tf_df)

    htf = pd.concat(out, axis=1)

    # simple cross-time-frame momentum ratios
    if {'RSI_daily', 'RSI_wk'}.issubset(htf.columns.union(df_daily.columns)):
        htf['RSI_ratio_dw'] = (
            df_daily['RSI_daily'] / htf['RSI_wk']
        )

    if {'EMA50_daily', 'EMA50_wk'}.issubset(htf.columns.union(df_daily.columns)):
        htf['EMA50_slope_diff'] = (
            df_daily['EMA50_daily'] - htf['EMA50_wk']
        )

    return htf

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
    label  = np.ones(len(close), dtype=int)          # default neutral/no-move

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
# ──────────────────────────────────────────────────────────────────────
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

def prepare_features(
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_90m: pd.DataFrame,
    df_1d:  pd.DataFrame,
    df_1wk: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    horizon: int = 10,
    drop_recent: bool = True,
    ticker: str | None = None
) -> pd.DataFrame:

    # ----------- core indicators on each frame -------------------------
    ind_15m = compute_indicators(df_15m.copy(), timeframe='15m')
    ind_30m = compute_indicators(df_30m.copy(), timeframe='30m')
    ind_1h  = compute_indicators(df_1h.copy(),  timeframe='hourly')
    ind_90m = compute_indicators(df_90m.copy(), timeframe='90m')
    ind_1d  = compute_indicators(df_1d.copy(),  timeframe='daily')
    ind_1d['AnchoredVWAP'] = compute_anchored_vwap(ind_1d)
    ind_1wk = compute_indicators(df_1wk.copy(), timeframe='1wk')

    # ----------- higher-TF enrichment (weekly / monthly / qtr) ---------
    ind_htf = enrich_higher_timeframes(ind_1d)

    # ----------- last bar of intraday frames ---------------------------
    daily_15m = to_daily(ind_15m, "15m").add_suffix('_15m')
    daily_30m = to_daily(ind_30m, "30m").add_suffix('_30m')
    daily_1h  = to_daily(ind_1h,  "1h").add_suffix('_1h')
    daily_90m = to_daily(ind_90m, "90m").add_suffix('_90m')

    # ----------- align weekly set to daily dates -----------------------
    ind_1wk = ind_1wk.reindex(ind_1d.index, method='ffill')

    # ----------- assemble full feature matrix --------------------------
    ind_1d.index.name = 'Date'
    features_df = (
        ind_1d
          .join(ind_htf)          # weekly/month/quarter resamples
          .join(ind_1wk)          # true weekly-bar indicators
          .join(daily_15m)
          .join(daily_30m)
          .join(daily_1h)
          .join(daily_90m)
    )

    # ----------- macro join --------------------------------------------
    if macro_df is not None and not macro_df.empty:
        features_df = features_df.join(macro_df, on='Date')

    # ----------- simple RS column --------------------------------------
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

    # ----------- triple-barrier labels ---------------------------------
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
        tree_method='hist',      # use GPU hist algo …
        device='cuda'            # … on the GPU
    )

    split = int(0.8 * len(X))
    if len(np.unique(y[:split])) < 2:       # not enough classes
        return features_df

    model.fit(X.iloc[:split], y.iloc[:split])
    imp = pd.Series(model.feature_importances_, index=X.columns)

    X = X.drop(columns=imp[imp < importance_cutoff].index, errors='ignore')

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper.columns if any(upper[c] > corr_threshold)],
               errors='ignore')

    return X.join(y)


# ──────────────────────────────────────────────────────────────────────
def train_stacked_ensemble(features_df: pd.DataFrame,
                           *, random_state: int = 42):
    """
    Replaces tune_threshold_and_train():
      • builds a tuned XGB, RF & Logistic base stack
      • finds the best probability threshold for LONG/SHORT
      • returns (fitted_stack_model, best_prob_threshold)
    """
    if features_df.empty or "future_class" not in features_df.columns:
        return None, None

    # 1) prep X / y ------------------------------------------------------
    y = features_df["future_class"].values
    X = features_df.drop(columns=["future_class"]).ffill().bfill()

    # 2) auto-tune XGB hyper-params on full data (quick random search) --
    best_xgb_params = tune_xgb_hyperparams(X, y, n_iter=20,
                                           random_state=random_state)
    xgb_clf = XGBClassifier(**best_xgb_params,
                            objective="multi:softprob",
                            num_class=3,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            n_jobs=-1,
                            random_state=random_state)

    # 3) build base learners + stacking meta-model -----------------------
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

    # 4) walk-forward split → train / validate for prob-threshold --------
    tscv = TimeSeriesSplit(n_splits=4)
    best_thr, best_f1 = 0.60, -np.inf       # default threshold

    for train_idx, val_idx in tscv.split(X):
        stack.fit(X.iloc[train_idx], y[train_idx])
        proba = stack.predict_proba(X.iloc[val_idx])

        # test several thresholds on LONG/SHORT classes (2 & 0)
        for thr in np.arange(0.55, 0.91, 0.05):
            pred = np.where(
                proba[:, 2] >= thr, 2,    # LONG
                np.where(proba[:, 0] >= thr, 0, 1)  # SHORT else Neutral
            )
            # treat neutral as class 1 → macro F1 on 0 & 2 only
            mask = pred != 1
            if mask.sum() == 0:
                continue
            f1 = f1_score(y[val_idx][mask], pred[mask],
                          labels=[0, 2], average="macro")
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

    # final fit on *all* data -------------------------------------------
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

    # regime check
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
    Build a daily macro / cross-asset DataFrame.

    • Fetches ALL Yahoo symbols in a **single** request to avoid rate-limit.
    • Falls back to an empty (date-only) frame if every download fails.
    • Caches the result → next run in the same day is instant.
    """
    from pathlib import Path
    import joblib, datetime as _dt

    # -------- disk-level daily cache -----------------------------------
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"macro_{end}.pkl"
    if cache_file.exists():
        try:
            return joblib.load(cache_file)
        except Exception:
            cache_file.unlink(missing_ok=True)   # corrupted → rebuild

    # -------- bulk Yahoo download --------------------------------------
    ymap = {
        '^VIX': 'VIX',  '^IRX': 'IRX',
        'TIP' : 'TIP',  'SPY': 'SPY',
        'DX-Y.NYB': 'USD', 'CL=F': 'CRUDE'
    }
    try:
        bulk = yf.download(
            list(ymap.keys()),
            start=start, end=end,
            auto_adjust=True, progress=False,
            threads=False  # ONE HTTPS request
        )['Close']
        if isinstance(bulk.columns, pd.MultiIndex):      # drop L0 (“Close”)
            bulk.columns = bulk.columns.droplevel(0)
        bulk.rename(columns=ymap, inplace=True)
        price_df = bulk.ffill()
    except Exception as e:
        print(f"[WARN] Bulk macro download failed: {e}")
        price_df = pd.DataFrame()

    # --------- FRED (optional) -----------------------------------------
    fred_df = pd.DataFrame(index=price_df.index)
    if fred_api_key:
        try:
            fred = Fred(api_key=fred_api_key)
            def _fred(code):
                return (fred.get_series(code,
                                         observation_start='1970-01-01',
                                         observation_end=end)
                            .resample('D').ffill())
            fred_df['YIELD10']      = _fred('DGS10')
            fred_df['YIELD_SPREAD'] = _fred('T10Y2Y')
        except Exception as e:
            print(f"[WARN] FRED fetch failed: {e}")

    # --------- merge & post-process ------------------------------------
    macro = pd.concat([price_df, fred_df], axis=1)

    if macro.empty:
        # still return a date index so code downstream keeps working
        macro = pd.DataFrame(index=pd.date_range(start, end, freq='D'))

    macro = macro.asfreq('D').ffill()

    # simple regime flag (same logic as before)
    reg = pd.Series(0, index=macro.index, dtype=int)
    if 'VIX' in macro:          reg += (macro['VIX'] > 25).astype(int) * -1
    if 'YIELD_SPREAD' in macro: reg += (macro['YIELD_SPREAD'] > 0).astype(int)
    if 'USD' in macro:
        reg += (macro['USD'].pct_change(20) < 0).fillna(False).astype(int)
    macro['REGIME'] = reg.clip(-1, 1)

    # --------- save daily cache & return -------------------------------
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
    df_15m, df_30m, df_1h, df_90m, df_1d, df_1wk = fetch_data(
        ticker, start=start_date, end=end_date
    )

    feat = prepare_features(df_15m, df_30m, df_1h, df_90m,
                            df_1d,  df_1wk, macro_df)
    feat = refine_features(feat)

    model, _ = train_stacked_ensemble(feat)          # UPDATED
    if model is None:
        print(Fore.YELLOW + f"Model training failed for {ticker}." + Style.RESET_ALL)
        return

    feat = feat.dropna(subset=["ATR_daily"])
    X_all = feat.drop(columns=["future_class"]).ffill().bfill()
    preds = model.predict(X_all)
    prob  = model.predict_proba(X_all)

    df_pred = feat.assign(prediction=preds)

    # ---- simple long/short sim (unchanged) ----------------------------
    trades, in_trade = [], False
    for ts, row in df_pred.iterrows():
        price, atr = row["Close"], row["ATR_daily"]
        if np.isnan(atr) or atr == 0:
            continue
        if not in_trade and prob[row.name][row.prediction] >= 0.6:
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
    df_30m = df_30m.copy()  # <-- ADDED to avoid SettingWithCopyWarning

    # 1) Compute intraday indicators
    df_30m = compute_indicators(df_30m, timeframe='intraday')
    if df_30m.empty or 'Close' not in df_30m.columns:
        return pd.DataFrame()

    # 2) Convert to daily & compute daily indicators
    daily = to_daily(df_30m, "intraday")
    daily = compute_indicators(daily, timeframe='daily')
    daily['AnchoredVWAP'] = compute_anchored_vwap(daily)

    # Remove timezones before reindexing:
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)  # <-- ADDED
    if df_30m.index.tz is not None:
        df_30m.index = df_30m.index.tz_localize(None)  # <-- ADDED

    # Forward-fill the daily data onto 30-min timestamps
    daily_filled = daily.reindex(df_30m.index, method='ffill')
    df_30m = df_30m.join(daily_filled, rsuffix='_daily')

    # 3) Label future_class on intraday
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

    # 4) Merge macro data
    if macro_df is not None and not macro_df.empty:
        if macro_df.index.tz is not None:
            macro_df = macro_df.copy()
            macro_df.index = macro_df.index.tz_localize(None)  # <-- ADDED

        macro_resampled = macro_df.reindex(df_30m.index, method='ffill')
        df_30m = df_30m.join(macro_resampled, how='left')

    return df_30m

def backtest_strategy_intraday(ticker, start_date, end_date, macro_df,
                               log_file=None):
    """
    30-minute intraday back-test with Sharpe, max-DD, time-in-market and ATR guard.
    """
    _, df_30m, _, _, _ = fetch_data(ticker, start=start_date, end=end_date)

    feat = prepare_features_intraday(df_30m, macro_df)
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

def run_signals_on_watchlist(use_intraday: bool = True):
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    # ONE-time bulk warm-up of the cache ↓↓↓
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
            df15, df30, df1h, df90, df1d, df1w = fetch_data(tkr)
        except Exception as e:
            print(f"Fetch error: {e}"); continue

        feats = (prepare_features_intraday(df30, macro_df)
                 if use_intraday else
                 prepare_features(df15, df30, df1h, df90, df1d, df1w, macro_df))
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

    preload_interval_cache(tickers)      # bulk warm-up here

    macro_df = get_macro_data(start_arg, end_arg)
    for ticker in tickers:
        print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
        backtest_strategy(ticker, start_arg, end_arg, macro_df)

def show_signals_for_current_week():
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Your watchlist is empty."); return

    preload_interval_cache(tickers)      # bulk cache warm-up

    today   = datetime.date.today()
    monday  = today - datetime.timedelta(days=today.weekday())
    start_s = (monday - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    end_s   = today.strftime("%Y-%m-%d")
    macro_df = get_macro_data(start_s, end_s, fred_api_key=fred_api_key)

    for tkr in tickers:
        print(f"\n=== {tkr}: {monday} → {today} ===")
        try:
            # use cached frames, then slice locally
            df15, df30, df1h, df90, df1d, df1w = fetch_data(tkr)
            df15, df30, df1h, df90, df1d, df1w = (
                df15.loc[start_s:end_s], df30.loc[start_s:end_s],
                df1h.loc[start_s:end_s], df90.loc[start_s:end_s],
                df1d.loc[start_s:end_s], df1w.loc[start_s:end_s]
            )
        except Exception as e:
            print(f"Fetch error: {e}"); continue

        all_feat = prepare_features(
            df15, df30, df1h, df90, df1d, df1w,
            macro_df, drop_recent=False
        )
        lbl_df = refine_features(all_feat.dropna(subset=['future_class']))
        model, thr = train_stacked_ensemble(lbl_df)
        if model is None:
            continue

        for dt, row in all_feat.loc[monday:end_s].iterrows():
            sig = generate_signal_output(tkr, row, model, thr, {})
            if sig:
                print(f"{dt.date()}: {sig}")
                break

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

    fred_api_key = load_config()
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
        macro_df = get_macro_data(start_arg, end_arg, fred_api_key=fred_api_key)

        # (a) Intraday approach if -real is set
        if use_intraday:
            for ticker in tickers:
                log_filename = f"{ticker}_30m_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Intraday Backtesting {ticker} (30m) from {start_arg} to {end_arg} ===")
                    backtest_strategy_intraday(ticker, start_arg, end_arg, macro_df, log_file=lf)

        # (b) Otherwise daily-lumped approach
        else:
            for ticker in tickers:
                log_filename = f"{ticker}_{start_arg}_{end_arg}.csv"
                with open(log_filename, "a") as lf:
                    print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
                    backtest_strategy(ticker, start_arg, end_arg, macro_df, log_file=lf)

        # Once backtest is done, we can safely close the main log file (if any) and return
        if log_file:
            log_file.close()
        return

    # ---------------------------------------------------------------------
    # 2) Otherwise do the "LIVE" signal generation mode
    # ---------------------------------------------------------------------
    # (No start/end provided or no --backtest)
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
            df_5m, df_30m, df_1h, df_90m, df_1d = fetch_data(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        if df_1d.empty or 'Close' not in df_1d.columns:
            print(f"No usable daily data for {ticker}, skipping.")
            continue

        features_df = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, macro_df)
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

        # Use last date from features or fallback to macro
        latest_date = features_df.index[-1]
        if latest_date in macro_df.index:
            macro_latest = macro_df.loc[latest_date].to_dict()
        else:
            macro_latest = macro_df.iloc[-1].to_dict()

        signal_output = generate_signal_output(ticker, latest_series, model, best_thr, macro_latest)
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