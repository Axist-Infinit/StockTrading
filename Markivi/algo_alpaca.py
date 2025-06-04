#!/usr/bin/env python3
"""
minervini_scanner_simplified.py (Alpaca API Version) – Advanced breakout screener & simulator
=============================================================================================

This version is adapted to use the Alpaca API for market data instead of yfinance.
It removes all CLI arguments. **Configuration, including Alpaca API credentials
and endpoint, lives in the CONFIGURATION block below.** Simply edit the values
and run the script.

- Python ≥ 3.10
- ``pip install alpaca-trade-api pandas numpy``

The engine scans for Mark Minervini–style VCP breakouts across multi‑
time‑frames (e.g., 5Min, 1Hour, 1Day), simulates entry/stop/target, and logs
alerts to both console and CSV/JSON.
"""
# This script is an adaptation of minervini_scanner_simplified.py to use the
# Alpaca API for market data instead of yfinance.

from __future__ import annotations

import random
import datetime as dt
import json
import sys
import time
# Ensure you have alpaca-trade-api installed: pip install alpaca-trade-api
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit #, APIError (already imported via tradeapi.rest)
# from datetime import datetime, timedelta # Already imported as dt, ensure alias consistency or use full names
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
# import yfinance as yf # yfinance is no longer used

# CACHE_DIR: Directory for storing cached market data from Alpaca.
CACHE_DIR = Path(".alpaca_cache")
CACHE_DIR.mkdir(exist_ok=True)

# RATE_DELAY: Originally for yfinance. Alpaca rate limiting is handled by
# modest delays and retry logic within DataFetcher._hit, plus sequential calls in bulk.
# Alpaca's main limit is 200 API requests per minute per key.
# RATE_DELAY = (1.5, 2.5)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# ► Edit anything in this block and run.  No other changes needed.

# --- Alpaca API Credentials & Settings ---
ALPACA_API_KEY    = "PKA7WN4XFHQ1OU719HLP"  # Alpaca API Key ID
ALPACA_API_SECRET = "Qdge3c1ztv54mhVeuvqMbmH5b2EBKbUyTFA3fYYa" # Alpaca API Secret Key
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets/v2"     # Alpaca base URL for paper trading
# ALPACA_BASE_URL   = "https://api.alpaca.markets/v2"         # Uncomment for live trading
ALPACA_DATA_FEED  = "iex"      # Data feed source: "iex" (free), "sip" (paid, requires subscription)
                               # Ensure your API key has access to the chosen feed.

# --- Scanner Settings ---
WATCHLIST        = [
    "NRG", "DUOL", "GEV", "CRDO", "BROS", "CRWV"
]                               # tickers to scan

MODE: str          = "eod"       # "eod"  = run once after close
                                # "live" = loop forever every INTERVAL seconds

INTERVAL_SECONDS  = 300          # only used when MODE == "live"

OUTFILE_PATH      = "alerts.csv" # CSV + JSON written side‑by‑side

# TIMEFRAMES: Defines the chart intervals and their corresponding historical lookback periods.
# Keys are Alpaca API compatible timeframe strings (e.g., "5Min", "1Hour", "1Day").
# Values are lookback period strings (e.g., "7d" for 7 days, "2Y" for 2 years).
# These lookback strings are converted into specific start datetimes in DataFetcher._calculate_start_date.
TIMEFRAMES: Dict[str, str] = {
    "5Min": "7d",   # 5-minute bars, look back 7 days
    "1Hour": "60d", # 1-hour bars, look back 60 days
    "1Day": "2Y",   # 1-day bars, look back 2 years
}

BENCHMARK         = "SPY"        # Benchmark symbol for Relative Strength calculation.
                               # Ensure this symbol is available via your Alpaca data feed.
MAX_WORKERS       = 8            # Max concurrent workers for data fetching.
                               # Note: DataFetcher.bulk is currently sequential for Alpaca to manage rate limits.
                               # If parallel fetching is re-introduced, this needs careful handling of Alpaca's 200 req/min limit.
MAX_PORTFOLIO_RISK= 0.02         # Maximum portfolio risk per position (currently for future use).
VCP_SCORE_THRESH  = 60.0         # 0‑100, higher ⇒ stricter
RS_RANK_THRESH    = 70.0         # 0‑100, higher ⇒ stricter
MAX_RISK_PCT      = 0.08         # stop ≤ 8 % below entry
# ════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
#                       CORE ENGINE (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker:   str
    tf:       str
    entry:    float
    stop:     float
    tgt2R:    float
    tgt3R:    float
    score:    float
    rs_rank:  float
    timestamp:str

class DataFetcher:
    MAX_RETRIES = 5
    INITIAL_BACKOFF_DELAY = 5  # seconds

    def __init__(self):
        self._mem: Dict[tuple, pd.DataFrame] = {}
        # Initialize Alpaca REST API client with credentials from CONFIGURATION block.
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            base_url=ALPACA_BASE_URL,
            api_version='v2'
        )

    def _calculate_start_date(self, period_str: str, end_date: dt.datetime) -> str:
        """
        Calculates the start datetime string in ISO 8601 format for Alpaca API calls.
        Converts a lookback period string (e.g., "7d", "60d", "2Y") relative to end_date.
        """
        num = int(period_str[:-1])
        unit = period_str[-1].lower()
        start_date: dt.datetime
        if unit == 'd':
            start_date = end_date - dt.timedelta(days=num)
        elif unit == 'm': # Assuming 'm' for months based on common conventions
            start_date = end_date - dt.timedelta(days=num * 30) # Approximation for months
        elif unit == 'y':
            start_date = end_date - dt.timedelta(days=num * 365) # Approximation for years
        else:
            raise ValueError(f"Unsupported period unit: {unit} in '{period_str}'")
        return start_date.isoformat()

    def _hit(self, symbol: str, interval: str, period: str) -> pd.DataFrame: # 'period' is lookback_str e.g. "7d"
        """
        Fetches historical bar data for a single symbol and interval from Alpaca API.
        Implements caching and retry logic.
        """
        key = (symbol, interval)
        if key in self._mem: # Check memory cache first
            return self._mem[key]

        # Cache file naming includes "_alpaca" to distinguish from potential yfinance caches.
        cache_file = CACHE_DIR / f"{symbol}_{interval}_alpaca.parq"

        current_time = dt.datetime.now(dt.timezone.utc) # Use timezone-aware datetime

        if cache_file.exists():
            try:
                file_mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime, dt.timezone.utc)
                ttl_minutes = 15
                if interval == "1Day":
                    ttl_minutes = 60 * 12 # 12 hours for daily data

                if (current_time - file_mod_time) < dt.timedelta(minutes=ttl_minutes):
                    df = pd.read_parquet(cache_file)
                    self._mem[key] = df
                    # print(f"[INFO] Cache hit for {symbol} {interval}.", file=sys.stderr)
                    return df
                else:
                    print(f"[INFO] Cache expired for {symbol} {interval}. Fetching new data.", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to read Parquet cache file {cache_file}: {e}. Will attempt download.", file=sys.stderr)

        tf_map = {
            "1Min": TimeFrame.Minute, "5Min": TimeFrame(5, TimeFrameUnit.Minute), "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour, "4Hour": TimeFrame(4, TimeFrameUnit.Hour), # Kept for mapping just in case, though not in config
            "1Day": TimeFrame.Day
        }
        if interval not in tf_map:
            print(f"[ERROR] Unsupported interval: {interval} for Alpaca. Supported: {list(tf_map.keys())}", file=sys.stderr)
            return pd.DataFrame()
        alpaca_timeframe = tf_map[interval]

        # Calculate start and end date/time for the Alpaca API call.
        end_dt = current_time
        start_dt_iso = self._calculate_start_date(period, end_dt) # 'period' is lookback_str e.g. "7d"
        end_dt_iso = end_dt.isoformat()

        for attempt in range(self.MAX_RETRIES):
            try:
                # Note: Alpaca's primary rate limit is 200 requests per minute per API key.
                # Current implementation is sequential in `bulk`, so aggressive sleeps here might not be
                # as critical as when using ThreadPoolExecutor.
                # time.sleep(random.uniform(0.2, 0.5))

                # Fetch bars using Alpaca SDK's get_bars method.
                # .df property directly converts the Alpaca BarSet to a Pandas DataFrame.
                bars_df = self.api.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_dt_iso,
                    end=end_dt_iso,
                    adjustment='raw',  # No price adjustments for splits/dividends.
                    feed=ALPACA_DATA_FEED # Use configured data feed (IEX or SIP).
                ).df

                if bars_df.empty:
                    # Alpaca returns an empty DataFrame if no data exists for the period (for a valid symbol).
                    # Symbol non-existence is caught as APIError (404).
                    print(f"[INFO] Empty DataFrame (no data in period) for {symbol} ({interval}) from Alpaca.", file=sys.stderr)
                    bars_df.to_parquet(cache_file, compression="zstd")
                    self._mem[key] = bars_df
                    return bars_df

                # --- Data Transformation ---
                # Rename columns to match the convention used by the original yfinance version
                # (Open, High, Low, Close, Volume). Alpaca .df uses lowercase.
                bars_df.rename(columns={
                    "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
                }, inplace=True)

                # Ensure DataFrame index is timezone-aware (Alpaca .df provides tz-aware UTC timestamps).
                if bars_df.index.tz is None:
                    bars_df.index = bars_df.index.tz_localize('UTC')

                # Select only the necessary columns for the rest of the script.
                df = bars_df[["Open", "High", "Low", "Close", "Volume"]]

                # Cache the processed DataFrame to Parquet file.
                df.to_parquet(cache_file, compression="zstd")
                self._mem[key] = df # Add to memory cache
                return df

            except tradeapi.rest.APIError as e: # Alpaca-specific API errors
                error_message = str(e)
                is_rate_limit = e.status_code == 429
                is_not_found = e.status_code == 404 # Symbol not found or other resource not found

                if is_not_found:
                    print(f"[ERROR] Alpaca API Error (404 Not Found) for {symbol} ({interval}): {e}. Likely symbol does not exist or no data feed access. Giving up.", file=sys.stderr)
                    df_empty = pd.DataFrame()
                    self._mem[key] = df_empty
                    return df_empty

                log_message_prefix = f"[WARN] Attempt {attempt + 1}/{self.MAX_RETRIES} for {symbol} ({interval}) failed."
                if is_rate_limit:
                    # Alpaca's main limit is 200/min. Backoff could be 60s / (200/MAX_WORKERS) on average.
                    # Using a simpler fixed + exponential backoff for now.
                    delay = self.INITIAL_BACKOFF_DELAY * (2 ** attempt)
                    delay = max(delay, 30) # Ensure a significant pause for rate limits, min 30s.
                    log_message = f"{log_message_prefix} Alpaca API rate limit error ({e.status_code}): {e}. Retrying in {delay}s..."
                else: # Other API errors (e.g., 500, 403 Forbidden if key is bad for the resource)
                    delay = self.INITIAL_BACKOFF_DELAY * (2 ** attempt)
                    log_message = f"{log_message_prefix} Alpaca API Error ({e.status_code}): {e}. Retrying in {delay}s..."

                print(log_message, file=sys.stderr)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    final_err_msg = f"[ERROR] All {self.MAX_RETRIES} retries failed for {symbol} ({interval}) with Alpaca. Last Error ({e.status_code}): {e}. Giving up."
                    print(final_err_msg, file=sys.stderr)
                    df_empty = pd.DataFrame()
                    self._mem[key] = df_empty
                    return df_empty
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.INITIAL_BACKOFF_DELAY * (2 ** attempt)
                    print(f"[WARN] Attempt {attempt + 1}/{self.MAX_RETRIES} for {symbol} ({interval}) failed with general error: {type(e).__name__}: {e}. Retrying in {delay}s...", file=sys.stderr)
                    time.sleep(delay)
                else:
                    print(f"[ERROR] All {self.MAX_RETRIES} retries failed for {symbol} ({interval}). General error: {type(e).__name__}: {e}. Giving up.", file=sys.stderr)
                    df_empty = pd.DataFrame()
                    self._mem[key] = df_empty
                    return df_empty

        df_fallback_empty = pd.DataFrame() # Should ideally not be reached
        self._mem[key] = df_fallback_empty
        return df_fallback_empty

    def bulk(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetches data for multiple symbols across all configured timeframes using Alpaca.
        Currently processes symbols and their timeframes sequentially to manage API rate limits.
        """
        out: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}
        # Iterate symbols first, then timeframes for each symbol.
        # This is sequential to be respectful of Alpaca's API rate limits (200 req/min).
        # If MAX_WORKERS were to be used with threading, rate limit handling would need
        # to be more sophisticated (e.g., a token bucket or similar).
        for sym in symbols:
            out[sym] = {}
            for interval, period_str in TIMEFRAMES.items():
                # print(f"[INFO] Processing {sym} for interval {interval} (lookback {period_str}) via Alpaca.", file=sys.stderr)
                df_sym = self._hit(sym, interval, period_str)
                out[sym][interval] = df_sym.dropna(how="all")
        return out

class PatternDetector:
    """Scores VCP tightness + checks Minervini trend template & breakout."""
    def __init__(self):
        self.bench: pd.DataFrame | None = None

    # ---------- metrics
    @staticmethod
    def rsi(series: pd.Series, win: int = 14) -> float:
        delta = series.diff()
        gain = np.maximum(delta, 0).rolling(win).mean().iloc[-1]
        loss = np.abs(np.minimum(delta, 0)).rolling(win).mean().iloc[-1]
        if loss == 0:
            return 100.0
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def trend_template(df: pd.DataFrame) -> bool:
        c = df["Close"]
        sma50  = c.rolling(50).mean(); sma150 = c.rolling(150).mean(); sma200 = c.rolling(200).mean()
        i = -1
        try:
            return all([
                c[i] > sma150[i] > sma200[i],
                sma50[i] > sma150[i],
                c[i] > sma50[i],
                c[i] >= c.rolling(52).max()[i] * 0.75,
                c[i] >= c.rolling(260).min()[i] * 1.3,
            ])
        except Exception:
            return False

    def vcp_score(self, df: pd.DataFrame, lookback: int = 35) -> float:
        sub = df.tail(lookback)
        if len(sub) < lookback * 0.8:
            return 0.0
        hi_series, lo_series, vol_series = sub["High"], sub["Low"], sub["Volume"]

        # Ensure scalar values for calculations
        hi_max = hi_series.max().item()
        lo_min = lo_series.min().item()

        if hi_max == 0: # Avoid division by zero
            rng_pct = float('inf') if lo_min < 0 else 0 # Or handle as appropriate
        else:
            rng_pct = (hi_max - lo_min) / hi_max

        range_score = max(0.0, 1.0 - rng_pct / 0.10) # Ensure float literals

        idx = len(sub)

        # Calculate r1
        hi_r1_series = hi_series[:idx//3]
        lo_r1_series = lo_series[:idx//3]
        if hi_r1_series.empty or lo_r1_series.empty: # check if series are empty
             r1 = float('inf') # or some other default value
        else:
             hi_r1_max = hi_r1_series.max().item()
             lo_r1_min = lo_r1_series.min().item()
             if hi_r1_max == 0:
                 r1 = float('inf') if lo_r1_min < 0 else 0
             else:
                 r1 = (hi_r1_max - lo_r1_min) / hi_r1_max

        # Calculate r2
        hi_r2_series = hi_series[-idx//3:]
        lo_r2_series = lo_series[-idx//3:]
        if hi_r2_series.empty or lo_r2_series.empty: # check if series are empty
             r2 = float('inf') # or some other default value
        else:
             hi_r2_max = hi_r2_series.max().item()
             lo_r2_min = lo_r2_series.min().item()
             if hi_r2_max == 0:
                 r2 = float('inf') if lo_r2_min < 0 else 0
             else:
                 r2 = (hi_r2_max - lo_r2_min) / hi_r2_max

        contr_score = np.clip((r1 - r2) / max(r1, 1e-9), 0, 1) if r1 != float('inf') and r2 != float('inf') else 0.0

        vol_mean = vol_series.mean().item()
        vol_tail_mean = vol_series.tail(lookback//3).mean().item()

        if vol_mean == 0:
            vol_score = 0.0 # Or handle as appropriate if mean volume is 0
        else:
            vol_score = max(0.0, 1.0 - (vol_tail_mean / vol_mean))

        # Add small epsilon to denominators to prevent division by zero if scores are zero
        range_score_safe = range_score if range_score > 1e-9 else 1e-9
        contr_score_safe = contr_score if contr_score > 1e-9 else 1e-9
        vol_score_safe = vol_score if vol_score > 1e-9 else 1e-9

        comp = 3.0 / (1.0/range_score_safe + 1.0/contr_score_safe + 1.0/vol_score_safe)
        return round(comp * 100, 1)

    def rs_rank(self, df: pd.DataFrame) -> float:
        if self.bench is None or self.bench.empty:
            return self.rsi(df["Close"])
        rel = df["Close"] / self.bench["Close"].reindex_like(df).ffill()
        rel_pct = rel.pct_change(252).iloc[-1]
        return round(100 / (1 + np.exp(-12 * rel_pct)), 1)

    def detect(self, sym: str, tf: str, df_tf: pd.DataFrame, df_d: pd.DataFrame) -> Signal | None:
        # Ensure df_tf has enough data for calculations
        if len(df_tf) < 20: # We need at least 20 data points for .tail(20) and other calculations
            print(f"[WARN] Not enough data for {sym} ({tf}) to detect patterns.", file=sys.stderr)
            return None

        # Ensure last, hi, lo are scalar float values
        try:
            last = df_tf["Close"].iloc[-1].item()
            hi = df_tf["High"].tail(20).max().item()
            lo = df_tf["Low"].tail(20).min().item()
        except IndexError:
            # This can happen if df_tf is too short after .dropna(how="all") in _hit
            # or if .item() is called on an empty Series (e.g. if tail(20).max() on an empty series returns empty series)
            # Note: .max()/.min() on an empty series should raise ValueError or return NaN, .item() would then fail if result is not single value.
            # If df_tf is too short, iloc[-1] will raise IndexError.
            # If df_tf has some rows but less than 20, tail(20) gets what it can. Then .max()/.min() should return a scalar or NaN.
            # If .max()/.min() returns NaN (a float), .item() is not needed and would fail.
            # The original float() cast handled NaN correctly. Using .item() assumes the operation returns a single-element Series.
            # If the series is empty or has multiple elements, .item() will fail.
            # Let's assume for now the warning implies these ops *can* return single-element series.
            # If .max()/.min() can return actual empty series instead of NaN, then .item() would fail.
            # This part might need more robust handling if .item() fails.
            print(f"[WARN] Not enough data for {sym} ({tf}) after basic processing to extract last/hi/lo. Skipping.", file=sys.stderr)
            return None


        # Add checks for potential NaN values after max/min operations on potentially incomplete data
        if pd.isna(last) or pd.isna(hi) or pd.isna(lo):
             print(f"[WARN] NaN values found for {sym} ({tf}) from last/hi/lo. Skipping pattern detection.", file=sys.stderr)
             return None

        if last < 0.95 * hi:
            return None
        score = self.vcp_score(df_tf)
        if score < VCP_SCORE_THRESH:
            return None
        if last <= hi * 1.001:
            return None
        # Ensure df_tf["Volume"] has enough data for .tail(20)
        if len(df_tf["Volume"]) < 20 or df_tf["Volume"].tail(20).mean() == 0: # Added check for zero mean to prevent division by zero
             print(f"[WARN] Not enough volume data or zero volume for {sym} ({tf}). Skipping.", file=sys.stderr)
             return None

        if df_tf["Volume"].iloc[-1] < 1.5 * df_tf["Volume"].tail(20).mean():
            return None
        if not self.trend_template(df_d):
            return None
        # Ensure df_d has enough data for rs_rank and trend_template (at least 260 for trend_template's 260-day min)
        if len(df_d) < 260:
             print(f"[WARN] Not enough daily data for {sym} to check trend template or RS rank. Skipping.", file=sys.stderr)
             return None

        rs = self.rs_rank(df_d)
        if rs < RS_RANK_THRESH:
            return None
        stop = lo * 0.99
        risk = last - stop
        if risk <= 0 or risk / last > MAX_RISK_PCT: # Added risk <= 0 check
            return None
        tgt2, tgt3 = last + 2*risk, last + 3*risk
        return Signal(sym, tf, round(last,2), round(stop,2), round(tgt2,2), round(tgt3,2), score, rs, dt.datetime.now().isoformat(timespec='seconds'))
class AlertSink:
    @staticmethod
    def console(sig: Signal):
        print(f"\n📢 {sig.ticker:<5} {sig.tf:<3} | entry {sig.entry:.2f} | stop {sig.stop:.2f} | tgt2R {sig.tgt2R:.2f} | score {sig.score} | RS {sig.rs_rank}")

    @staticmethod
    def save(alerts: List[Signal], path: str | Path):
        if not alerts:
            print("\nNo qualifying setups this run.")
            return
        df = pd.DataFrame(asdict(a) for a in alerts)
        df.to_csv(path, index=False)
        Path(path).with_suffix('.json').write_text(df.to_json(orient="records", indent=2))
        print(f"\n[SAVED] {path} & *.json")

# ─────────────────────────────────────────────────────────────────────────
#                               MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────

def run_scan():
    fetcher = DataFetcher()
    detector = PatternDetector()
    data = fetcher.bulk(WATCHLIST + [BENCHMARK])
    detector.bench = data.get(BENCHMARK, {}).get("1d", pd.DataFrame())

    alerts: List[Signal] = []
    for sym in WATCHLIST:
        df_daily = data[sym]["1d"]
        for tf in TIMEFRAMES.keys():
            sig = detector.detect(sym, tf, data[sym][tf], df_daily)
            if sig:
                alerts.append(sig)
                AlertSink.console(sig)
    AlertSink.save(alerts, OUTFILE_PATH)


if __name__ == "__main__":
    if MODE == "eod":
        run_scan()
    elif MODE == "live":
        print(f"🔄 Live mode: scanning every {INTERVAL_SECONDS}s – Ctrl‑C to stop.")
        try:
            while True:
                run_scan()
                time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        print("[ERROR] MODE must be either 'eod' or 'live'.")
