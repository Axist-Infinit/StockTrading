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
    def __init__(self):
        self._mem: Dict[tuple, pd.DataFrame] = {}

    def _hit(self, symbols: list[str], interval: str, period: str) -> pd.DataFrame:
        """Download multiple symbols at once & obey local + disk cache."""
        key = (tuple(symbols), interval)
        if key in self._mem:
            return self._mem[key]

        cache_file = CACHE_DIR / f"{'_'.join(symbols)}_{interval}.parq"
        if cache_file.exists() and cache_file.stat().st_mtime > time.time() - 900:  # 15 min TTL
            df = pd.read_parquet(cache_file)
        else:
            df = yf.download(symbols, period=period, interval=interval,
                             progress=False, group_by="ticker", auto_adjust=False)
            df.to_parquet(cache_file, compression="zstd")
            # --- pacing to avoid 429 ---
            time.sleep(random.uniform(*RATE_DELAY))

        self._mem[key] = df
        return df

    def bulk(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        out: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}
        # sequential – much kinder to Yahoo
        for interval, period in TIMEFRAMES.items():
            df_all = self._hit(symbols, interval, period)
            # split the combined dataframe back into single-symbol chunks
            for sym in symbols:
                part = df_all[sym] if isinstance(df_all.columns, pd.MultiIndex) else df_all
                out[sym][interval] = part.dropna(how="all")
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
        hi, lo, vol = sub["High"], sub["Low"], sub["Volume"]
        rng_pct = (hi.max() - lo.min()) / hi.max()
        range_score = max(0, 1 - rng_pct / 0.10)
        idx = len(sub)
        r1 = (hi[:idx//3].max() - lo[:idx//3].min()) / hi[:idx//3].max()
        r2 = (hi[-idx//3:].max() - lo[-idx//3:].min()) / hi[-idx//3:].max()
        contr_score = np.clip((r1 - r2) / max(r1, 1e-9), 0, 1)
        vol_score = max(0, 1 - (vol.tail(lookback//3).mean() / vol.mean()))
        comp = 3 / (1/range_score + 1/contr_score + 1/vol_score + 1e-9)
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

        last = df_tf["Close"].iloc[-1]
        # Explicitly cast hi to float to avoid Series ambiguity
        hi = float(df_tf["High"].tail(20).max())
        lo = df_tf["Low"].tail(20).min()

        # Add checks for potential NaN values after max/min operations on potentially incomplete data
        if pd.isna(last) or pd.isna(hi) or pd.isna(lo):
             print(f"[WARN] NaN values found for {sym} ({tf}). Skipping pattern detection.", file=sys.stderr)
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

