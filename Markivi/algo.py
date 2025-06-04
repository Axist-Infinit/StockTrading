#!/usr/bin/env python3
"""
minervini_scanner_simplified.py – Advanced breakout screener & simulator
========================================================================

This version removes all CLI arguments.  **Everything you might want to
change lives in the CONFIGURATION block below.**  Simply edit the values
and run the script.

- Python ≥ 3.10
- ``pip install yfinance pandas numpy``

The engine scans for Mark Minervini–style VCP breakouts across multi‑
time‑frames (5 m, 1 h, 4 h, 1 d), simulates entry/stop/target, and logs
alerts to both console and CSV/JSON.
"""
from __future__ import annotations

import random
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
CACHE_DIR = Path(".yf_cache")
CACHE_DIR.mkdir(exist_ok=True)

RATE_DELAY = (0.7, 1.1)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# ► Edit anything in this block and run.  No other changes needed.

WATCHLIST        = [
    "NRG", "DUOL", "GEV", "CRDO", "BROS", "CRWV"
]                               # tickers to scan

MODE: str          = "eod"       # "eod"  = run once after close
                                # "live" = loop forever every INTERVAL seconds

INTERVAL_SECONDS  = 300          # only used when MODE == "live"

OUTFILE_PATH      = "alerts.csv" # CSV + JSON written side‑by‑side

TIMEFRAMES: Dict[str, str] = {    # yfinance_interval : max lookback period
    "5m": "7d",
    "1h": "60d",
    "4h": "60d",
    "1d": "2y",
}

BENCHMARK         = "SPY"        # for RS‑line strength
MAX_WORKERS       = 8            # parallel yfinance threads
MAX_PORTFOLIO_RISK= 0.02         # 2 % of equity per alert (future use)
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
    MAX_RETRIES = 3
    INITIAL_BACKOFF_DELAY = 1  # seconds

    def __init__(self):
        self._mem: Dict[tuple, pd.DataFrame] = {}

    def _hit(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Download a single symbol & obey local + disk cache with retries."""
        key = (symbol, interval)
        if key in self._mem:
            # If already in memory cache (e.g. from a previous failed attempt), return it.
            return self._mem[key]

        cache_file = CACHE_DIR / f"{symbol}_{interval}.parq"
        if cache_file.exists() and cache_file.stat().st_mtime > time.time() - 900:  # 15 min TTL
            try:
                df = pd.read_parquet(cache_file)
                self._mem[key] = df # Also load to memory cache if read from disk
                return df
            except Exception as e:
                print(f"[WARN] Failed to read Parquet cache file {cache_file}: {e}. Will attempt download.", file=sys.stderr)


        for attempt in range(self.MAX_RETRIES):
            try:
                # For a single symbol, yfinance returns a DataFrame directly.
                df = yf.download(symbol, period=period, interval=interval,
                                 progress=False, auto_adjust=False)
                if df.empty and symbol not in WATCHLIST + [BENCHMARK]: # Allow empty for known symbols that might genuinely have no data for a period.
                    # yf.download can return an empty df for legitimate reasons (e.g. delisted ticker partway through period)
                    # However, if it's an unexpected symbol or consistently empty, it might indicate an issue.
                    # For now, we treat truly empty DFs as potential issues to retry unless it's a known ticker.
                    # This logic might need refinement based on yfinance behavior for various errors.
                    # If a ticker genuinely has no data for the period, yfinance returns an empty DF with columns.
                    # If a ticker does not exist, yfinance prints "No data found, symbol may be delisted" and returns an empty DF.
                    print(f"[WARN] Empty DataFrame returned for {symbol} on attempt {attempt + 1}/{self.MAX_RETRIES}. Retrying...", file=sys.stderr)
                    # Raising an exception to trigger retry logic for empty DFs on unexpected symbols or if we want to be more aggressive.
                    # For now, let's only retry if yf.download itself fails, not if it returns an empty df.
                    # If df.empty, we'll proceed to cache it. If it was due to an error yf didn't raise, this might hide issues.
                    # Consider raising an error here if df is empty to force retry. For now, we let it pass.
                    pass


                df.to_parquet(cache_file, compression="zstd")
                self._mem[key] = df
                # --- pacing to avoid 429 ---
                time.sleep(random.uniform(*RATE_DELAY))
                return df
            except Exception as e:
                print(f"[WARN] Download failed for {symbol} ({interval}, {period}) on attempt {attempt + 1}/{self.MAX_RETRIES}. Error: {type(e).__name__}: {e}", file=sys.stderr)
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.INITIAL_BACKOFF_DELAY * (2 ** attempt)
                    print(f"Retrying in {delay}s...", file=sys.stderr)
                    time.sleep(delay)
                else:
                    print(f"[ERROR] All retries failed for {symbol} ({interval}, {period}) after {self.MAX_RETRIES} attempts.", file=sys.stderr)
                    df_empty = pd.DataFrame()
                    self._mem[key] = df_empty # Cache empty DF to prevent immediate re-attempts
                    return df_empty

        # Should not be reached if logic is correct, but as a fallback:
        df_fallback_empty = pd.DataFrame()
        self._mem[key] = df_fallback_empty
        return df_fallback_empty


    def bulk(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        out: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}
        # sequential – much kinder to Yahoo
        for interval, period in TIMEFRAMES.items():
            for sym in symbols:
                df_sym = self._hit(sym, interval, period)
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
        hi_max = float(hi_series.max())
        lo_min = float(lo_series.min())

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
             hi_r1_max = float(hi_r1_series.max())
             lo_r1_min = float(lo_r1_series.min())
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
             hi_r2_max = float(hi_r2_series.max())
             lo_r2_min = float(lo_r2_series.min())
             if hi_r2_max == 0:
                 r2 = float('inf') if lo_r2_min < 0 else 0
             else:
                 r2 = (hi_r2_max - lo_r2_min) / hi_r2_max

        contr_score = np.clip((r1 - r2) / max(r1, 1e-9), 0, 1) if r1 != float('inf') and r2 != float('inf') else 0.0

        vol_mean = float(vol_series.mean())
        vol_tail_mean = float(vol_series.tail(lookback//3).mean())

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
            last = float(df_tf["Close"].iloc[-1])
            hi = float(df_tf["High"].tail(20).max())
            lo = float(df_tf["Low"].tail(20).min())
        except IndexError:
            # This can happen if df_tf is too short after .dropna(how="all") in _hit
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

