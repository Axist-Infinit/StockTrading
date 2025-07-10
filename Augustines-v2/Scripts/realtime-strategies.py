#!/usr/bin/env python3
"""
strict_swing_system_alpaca.py
Author  : <you>       • Updated : 2025‑07‑10
─────────────────────────────────────────────────────────────────────────────
Hard‑rule swing scanner (1 day–2 week holds) using Alpaca market‑data v2.
Runs every 15 min, evaluates 40‑50 symbols, and logs any NEW all‑rules‑PASS
signals with entry, micro‑stop, swing stop and 2R/3R targets.
─────────────────────────────────────────────────────────────────────────────
DEPENDENCIES
  pip install alpaca-trade-api pandas numpy pandas_ta schedule pytz
"""
from __future__ import annotations
import os, time, json, logging, datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
import schedule
import pytz

from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

# ───────────────────────── USER SETTINGS ──────────────────────────
API_KEY    = os.getenv("APCA_API_KEY_ID",    "YOUR_KEY")
API_SECRET = os.getenv("APCA_API_SECRET_KEY","YOUR_SECRET")
BASE_URL   = os.getenv("APCA_API_BASE_URL",  "https://paper-api.alpaca.markets")

TICKERS: List[str] = [                     # ≤ 50 symbols
    "AAPL","MSFT","NVDA","AMD","AMZN","META","CRM","GOOGL","LULU","MELI"
]

ACCOUNT_EQUITY   = 100_000      # $
RISK_PCT         = 0.01         # 1 % per trade
BETA_CAP         = 2.5
MIN_DAILY_VOL    = 500_000
FEED             = "sip"        # set "iex" for free accounts without SIP

LOGFILE = "swing_signals_alpaca.log"
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(LOGFILE),
                              logging.StreamHandler()],
                    format="%(asctime)s  %(levelname)s  %(message)s")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

ALERTED: set[str] = set()          # avoid duplicate prints this session
EASTERN = pytz.timezone("America/New_York")

# ════════════════ small helpers ══════════════════════════════════
def _get_bars(sym: str, tf, days: int) -> pd.DataFrame:
    """Pull `days` calendar days of bars for `sym` and given Alpaca timeframe."""
    end   = dt.datetime.now(tz=dt.timezone.utc)
    start = end - dt.timedelta(days=days*1.3)      # cushion for weekends
    bars  = api.get_bars(sym, tf, start.isoformat(),
                         end.isoformat(),
                         adjustment="raw", feed=FEED, limit=10_000).df
    return bars.tz_convert(EASTERN) if not bars.empty else bars

def _sma(s, n):  return s.rolling(n).mean()
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def _weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    wk = daily_df.resample("W-FRI").agg({"open":"first","high":"max",
                                         "low":"min","close":"last",
                                         "volume":"sum"}).dropna()
    wk.columns = wk.columns.str.capitalize()
    return wk

def _relative_strength(prc: pd.Series, spy: pd.Series, lb: int = 40) -> float:
    ex_ret = prc.pct_change(lb).iloc[-1] - spy.pct_change(lb).iloc[-1]
    buckets = (-.30,-.20,-.10,0,.10,.20,.30)
    return 100*sum(ex_ret>b for b in buckets)/len(buckets)

def _beta(sym_ret: pd.Series, spy_ret: pd.Series) -> float:
    sym, spy = sym_ret.align(spy_ret, join="inner")
    return np.cov(sym, spy)[0,1] / spy.var() if spy.var()!=0 else 0

# ═══════════════ main evaluation ═════════════════════════════════
def evaluate(sym: str, spy_daily: pd.DataFrame) -> Dict|None:
    # --------------- fetch required bars -------------------------
    daily   = _get_bars(sym, TimeFrame.Day,        400)
    if daily.empty or daily.volume.tail(20).mean() < MIN_DAILY_VOL:
        return None

    hourly  = _get_bars(sym, TimeFrame.Hour,       60)
    tf15m   = _get_bars(sym, TimeFrame(15,TimeFrameUnit.Minute), 3)
    tf5m    = _get_bars(sym, TimeFrame(5, TimeFrameUnit.Minute), 1)

    # --------------- weekly trend filter -------------------------
    weekly  = _weekly(daily)
    sma30_w = _sma(weekly.Close, 30)
    macd_w  = ta.macd(weekly.Close)["MACDh_12_26_9"]
    if not (weekly.Close.iloc[-1] > sma30_w.iloc[-1] and
            sma30_w.diff().iloc[-5:].gt(0).all() and
            macd_w.iloc[-1] > macd_w.iloc[-2] > macd_w.iloc[-3]):
        return None

    # --------------- daily trend template ------------------------
    sma50, sma150, sma200 = (_sma(daily.Close, n) for n in (50,150,200))
    cond_daily = all([
        daily.Close.iloc[-1] > sma50.iloc[-1] > 0,
        daily.Close.iloc[-1] > sma150.iloc[-1] > 0,
        daily.Close.iloc[-1] > sma200.iloc[-1] > 0,
        sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]
    ])
    if not cond_daily:
        return None

    rs = _relative_strength(daily.Close, spy_daily.Close)
    if rs < 70:         # relative‑strength fail
        return None
    hi52 = daily.Close[-252:].max()
    if daily.Close.iloc[-1] < 0.75*hi52:
        return None

    beta = _beta(daily.Close.pct_change(), spy_daily.Close.pct_change())
    if beta > BETA_CAP:
        return None

    # --------------- setup qualifier -----------------------------
    last_15 = daily[-15:]
    pivot   = last_15.High.max()
    base_lo = last_15.Low.min()

    rngs = last_15.High - last_15.Low
    vol_contract = all(rngs.iloc[i] < rngs.iloc[i-1] for i in range(1,4))
    vol_fade     = last_15.Volume.mean() < daily.Volume.tail(50).mean()
    breakout_set = vol_contract and vol_fade

    adx_val = ta.adx(daily.High, daily.Low, daily.Close)["ADX_14"].iloc[-1]
    ema20   = _ema(daily.Close, 20)
    stoch   = ta.stoch(daily.High, daily.Low, daily.Close)
    st_cross = (stoch["STOCHk_14_3_3"].iloc[-2] < 30 <
                stoch["STOCHk_14_3_3"].iloc[-1] >
                stoch["STOCHd_14_3_3"].iloc[-1])
    pullback_set = (adx_val >= 25 and
                    daily.Close.iloc[-2] < ema20.iloc[-2] and
                    daily.Close.iloc[-1] > ema20.iloc[-1] and
                    st_cross)

    if not (breakout_set or pullback_set):
        return None

    # --------------- 4‑hour confirmation -------------------------
    h4 = hourly.Close.resample("4H").last().dropna()
    if breakout_set and h4.iloc[-1] <= pivot:
        return None

    # --------------- hourly TTM squeeze release ------------------
    bbands = ta.bbands(hourly.Close, length=20, std=2)
    kc     = ta.kc(hourly.High, hourly.Low, hourly.Close,
                   length=20, scalar=1.5)
    sq_on  = (bbands["BBL_20_2.0"] > kc["KCLe_20_1.5"]) & \
             (bbands["BBU_20_2.0"] < kc["KCUe_20_1.5"])
    sq_off = sq_on.shift(1) & (~sq_on)
    macdh  = ta.macd(hourly.Close)["MACDh_12_26_9"]
    sq_long = (sq_off & (macdh > 0)).iloc[-3:].any()
    if not sq_long:
        return None

    # --------------- 5‑min volume gate ---------------------------
    if tf5m.shape[0] < 20:
        return None
    avg5 = tf5m.volume.tail(20).mean()
    cum15 = tf5m.volume.tail(3).sum()
    if cum15 < 5*avg5:
        return None

    # --------------- entry checks -------------------------------
    today_vol = daily.Volume.iloc[-1]
    if today_vol < 1.5*daily.Volume.tail(50).mean():
        return None
    price_now = daily.Close.iloc[-1]
    if breakout_set and price_now > 1.02*pivot:
        return None

    # --------------- stops / targets ----------------------------
    atr_day  = ta.atr(daily.High, daily.Low, daily.Close).iloc[-1]
    atr15m   = ta.atr(tf15m.high, tf15m.low, tf15m.close).iloc[-1]
    stop_intra = price_now - atr15m               # micro stop (≤1×ATR‑15m)
    stop_swing = price_now - 2*atr_day            # daily swing stop
    initial_stop = max(base_lo, stop_intra)       # reconcile w/ base low
    risk_pct = (price_now - initial_stop)/price_now
    if risk_pct > 0.08:
        return None

    shares = int((ACCOUNT_EQUITY*RISK_PCT) / (price_now - initial_stop))
    if   beta > 2.0: shares = int(shares*0.5)
    elif beta > 1.5: shares = int(shares*0.75)

    tgt1 = price_now + 2*(price_now - initial_stop)
    tgt2 = price_now + 3*(price_now - initial_stop)

    return dict(ticker=sym, pattern="breakout" if breakout_set else "pullback",
                entry=round(price_now,2), stop=round(initial_stop,2),
                target1=round(tgt1,2), target2=round(tgt2,2),
                shares=shares, beta=round(beta,2), rs=int(rs))

# ═══════════════ scheduler loop ═════════════════════════════════
def scan():
    spy = _get_bars("SPY", TimeFrame.Day, 400)
    if spy.empty:
        logging.warning("SPY data fetch failed.")
        return
    new = []
    for s in TICKERS:
        sig = evaluate(s, spy)
        if sig and (sig_id:=f"{sig['ticker']}@{sig['entry']}") not in ALERTED:
            ALERTED.add(sig_id)
            new.append(sig)
    if new:
        for sig in new:
            logging.info("SIGNAL  " + json.dumps(sig))
    else:
        logging.info("No new signals this run.")

def main():
    schedule.every(15).minutes.do(scan)
    logging.info("Swing engine started – running every 15 min.")
    scan()  # immediate run
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")

if __name__ == "__main__":
    main()
