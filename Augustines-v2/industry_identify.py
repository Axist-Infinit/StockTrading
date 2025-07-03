#!/usr/bin/env python3
"""
industry_momentum.py – nightly screener + back‑tester
author  : <you>
updated : 2025‑06‑27
"""
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import json, datetime as dt, itertools, sys
import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

# ── API keys ────────────────────────────────────────────────────────────────
ALPACA_KEY     = "PKA7WN4XFHQ1OU719HLP"
ALPACA_SECRET  = "Qdge3c1ztv54mhVeuvqMbmH5b2EBKbUyTFA3fYYa"
BASE_URL       = "https://paper-api.alpaca.markets"
api = REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL, api_version="v2")

# ── directories & file patterns ─────────────────────────────────────────────
THIS_DIR  = Path(__file__).resolve().parent
DATA_DIR  = THIS_DIR / "Data"                # holds daily FinViz JSONs
IND_FILE  = THIS_DIR / "industries.json"     # {industry: [tickers]}
SECTOR_MAP_FILE  = THIS_DIR / "industry_to_sector.json"  # ← mapping we built
RELATIVE         = True   # switch OFF to return to raw momentum
SIGNALS_TV_DIR      = THIS_DIR / "Signals-TradingView"
SIGNALS_HISTORY_DIR = THIS_DIR / "Signals-History"
SIGNALS_TV_DIR.mkdir(exist_ok=True)
SIGNALS_HISTORY_DIR.mkdir(exist_ok=True)
_DAY_CACHE: dict[str, pd.DataFrame] = {}   # key = iso‑date string ("YYYY‑MM‑DD")


if not SECTOR_MAP_FILE.exists():
    print("[WARN] sector‑map file is missing – relative momentum disabled.")
    RELATIVE = False

def fv_file_for(date_: dt.date) -> Path:
    """Assumes files are named M.D.json e.g. 6.26.json in DATA_DIR."""
    return DATA_DIR / f"{date_.month}.{date_.day}.json"
# path helper for fundamentals (expects Data/6.28-value.json etc.)

def value_file_for(date_: dt.date) -> Path:
    return DATA_DIR / f"{date_.month}.{date_.day}-value.json"

# ── strategy tunables ───────────────────────────────────────────────────────
# ── data‑feed preference & dynamic volume floor ─────────────────────────────
PREFERRED_FEED = "sip"        # set "iex" to suppress any SIP attempt
FALLBACK_FEED  = "iex"        # never change this
MIN_VOL_SIP    = 200_000      # realistic with consolidated tape
MIN_VOL_IEX    = 20_000       # ~1/10th because IEX ≈2 % market share :contentReference[oaicite:1]{index=1}

TOP_N_INDS        = 10
LOOKBACKS         = {"Perf_Week": .10, "Perf_Month": .25, "Perf_Quart": .30,
                     "Perf_Half": .20, "Perf_Year": .15}
BREADTH_MIN_PCT   = 0.30
RS_MIN            = 70
VOL_THRESHOLD     = 1.00
MAX_RECOM         = 2.20
MARKET_ETFS       = {"SPY", "QQQ", "MDY", "IWM"}
ATR_PARAMS = {
    "bull":     {"stop_mult": 2.0, "target_mult": 3.0},   # trending (long bias)
    "bear":     {"stop_mult": 2.0, "target_mult": 3.0},   # trending (short bias)
    "sideways": {"stop_mult": .8, "target_mult": 1.5},   # choppy / range‑bound
}

TREND_LOOKBACK_DAYS = 7          # minimum history before trend table appears
W_TREND   = 0.05                 # 5 % of final score comes from 7‑day trend

W_MOM  = 0.38                    # 0.40 → 0.38 after allocating 0.05 to trend
W_TECH = 0.29                    # 0.30 → 0.29
W_VOL  = 0.19                    # 0.20 → 0.19
W_FUND = 0.09                    # 0.10 → 0.09



# ── regime & extra strategy parameters ──────────────────────────────────────
MEAN_REV_EXTREME   = 0.20      # 20 % beyond / below 20‑sma counts as extreme
MEAN_REV_STOP_PCT  = 0.03      # 3 % hard stop on mean‑reversion trades
MEAN_REV_TARGET_PCT= 0.05      # 5 % initial profit target
SHORT_STOP_PCT     = 0.08      # 8 % stop on short trades
SHORT_TARGET_RMULT = 1.8       # cover 1st half at 1.8× initial risk
RS_WEAK_MAX        = 30        # RS ≤ 30 to qualify for short list
BOTTOM_N_INDS      = 10        # how many laggards to keep for short scans

MIN_VOL = None
# ══════════════════════════ H E L P E R S ════════════════════════════════════
def _load_json(path):              return json.loads(Path(path).read_text())
def _pct_to_float(v):              return float(v) if isinstance(v,(int,float)) else float(v.strip('%'))/100

# ── performance‑column normaliser ───────────────────────────────────────────
_PERF_ALIASES = {
    "Perf_Week" : ["Perf_Week", "Perf Week", "PerfWeek", "Perf 1W", "Perf1W"],
    "Perf_Month": ["Perf_Month", "Perf Month", "PerfMonth", "Perf 1M", "Perf1M"],
    "Perf_Quart": ["Perf_Quart", "Perf Quart", "PerfQuarter", "Perf 3M"],
    "Perf_Half" : ["Perf_Half", "Perf Half", "PerfHalf", "Perf 6M"],
    "Perf_Year" : ["Perf_Year", "Perf Year", "PerfYear", "Perf 1Y", "Perf1Y"],
}

def _normalize_perf_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure every canonical perf column in LOOKBACKS exists.
    If FinViz used a space (e.g. 'Perf Week') or no underscore,
    copy that column to the canonical underscore name.
    Missing columns are filled with 0 and a warning is printed.
    """
    for canon, aliases in _PERF_ALIASES.items():
        if canon in df.columns:
            continue
        for alt in aliases:
            if alt in df.columns:
                df[canon] = df[alt]
                break
        else:
            # none found – fill neutral value so the math still works
            df[canon] = 0.0
            print(f"[WARN] FinViz export lacks column '{canon}' "
                  f"(tried {aliases}); filled with 0.")
    return df

# ── breadth column extractor ────────────────────────────────────────────────
def _extract_breadth(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series (index = df.index) with breadth in **0‑1** range.
    If none of the known FinViz breadth columns are present, returns 0.0.
    """
    breadth_cols = ["High_50d_pct", "Pct_From_50d_High", "Near_50d_High"]
    for col in breadth_cols:
        if col in df.columns:
            return df[col].apply(_safe_float).div(100).clip(0, 1)
    print("[WARN] breadth column missing – neutral breadth used.")
    return pd.Series(0.0, index=df.index)


# ── helpers to browse previously‑saved signals ────────────────────────────
def _list_signal_files() -> list[Path]:
    """Return all JSON signal files, newest → oldest."""
    return sorted(SIGNALS_HISTORY_DIR.glob("*-Signals.json"), reverse=True)

def _load_signals(path: Path) -> pd.DataFrame:
    """Read a JSON signal file into a DataFrame (raises on failure)."""
    return pd.read_json(path)

def _to_dt(d):
    return d if isinstance(d, dt.datetime) else dt.datetime.combine(d, dt.time.min)

def _get_bars(
    sym: str,
    start: dt.date | dt.datetime,
    end:   dt.date | dt.datetime,
    limit: int | None = None
) -> pd.DataFrame | None:
    """
    Fetch daily bars, automatically falling back from SIP → IEX when
    the account has no SIP entitlement (or any other feed‑specific error).

    • Keeps the 16‑min delay to avoid partial bars.
    • Returns None when *both* feeds provide no data.
    """
    def _naive(d):
        return d.replace(tzinfo=None) if isinstance(d, dt.datetime) \
               else dt.datetime.combine(d, dt.time.min)

    start, end = map(_naive, (start, end))
    end = min(end, api.get_clock().timestamp.replace(tzinfo=None)
                    - dt.timedelta(minutes=16))

    for feed in (PREFERRED_FEED, FALLBACK_FEED) if PREFERRED_FEED != FALLBACK_FEED else (FALLBACK_FEED,):
        try:
            df = api.get_bars(
                sym, timeframe="1Day",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                limit=limit,
                feed=feed
            ).df
        except APIError as e:
            # typical message: “subscription does not permit querying recent SIP data”
            if "subscription does not permit" in str(e).lower() and feed == PREFERRED_FEED:
                print(f"[INFO] feed {feed.upper()} not permitted – switching to {FALLBACK_FEED.upper()}")
                continue   # try fallback
            print(f"[WARN] {sym}: {e}")
            return None

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=0)

        df.columns = [c.lower() for c in df.columns]
        if "vw" in df.columns:
            df.rename(columns={"vw": "vwap"}, inplace=True)

        # adjust global MIN_VOL once per session when we confirm the feed
        global MIN_VOL
        if MIN_VOL is None:             # first successful call
            MIN_VOL = MIN_VOL_SIP if feed == "sip" else MIN_VOL_IEX
        return df

    return None


# ───────── convenience accessors (add near other helpers) ─────────
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Return df[name] if present, else an empty Series so downstream
    logic never raises KeyError.  Keeps column names case‑safe.
    """
    return df[name] if name in df.columns else pd.Series(dtype=float)

def _last(series: pd.Series):
    """Shorthand for the last element using .iloc to silence warnings."""
    return series.iloc[-1] if not series.empty else None


def _get_bars_lookback(sym: str, days: int):
    """
    Convenience wrapper: fetch `days` calendar days back from *today*.
    Uses the same IEX feed and 16‑minute delay as _get_bars().
    """
    end   = dt.date.today()
    start = end - dt.timedelta(days=days*1.7)   # same buffer logic
    return _get_bars(sym, start, end, limit=days)

def _ema(series, span):            return series.ewm(span=span, adjust=False).mean()
def _sma(series, n):           return series.rolling(n).mean()

_SPY_BARS: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}

def _rs(series: pd.Series) -> float:
    """
    40‑day Relative Strength vs SPY, scaled 0‑100.
    Uses a tiny cache so we don’t fetch SPY bars on every call.
    """
    if series.empty:
        return 0

    first, last = series.index[0].date(), series.index[-1].date()
    key = (first, last)

    if key not in _SPY_BARS:
        spy = _get_bars("SPY", first, last, limit=60)
        if spy is None or spy.empty:
            return 100          # assume strong if benchmark is missing
        _SPY_BARS[key] = spy
    else:
        spy = _SPY_BARS[key]

    rel = series.pct_change(40).iloc[-1] - spy.close.pct_change(40).iloc[-1]
    buckets = (-.30, -.20, -.10, 0, .10, .20, .30)
    return 100 * sum(rel > b for b in buckets) / len(buckets)

def finviz_file(date_: dt.date) -> Path:
    """
    Resolve the FinViz industry‑level stats snapshot for `date_`.
    Tries:  Data/YYYY‑MM‑DD.json   → preferred
            Data/M.D.json          → fallback (legacy)
    """
    iso_name  = DATA_DIR / f"{date_}.json"
    legacy    = DATA_DIR / f"{date_.month}.{date_.day}.json"
    if iso_name.exists():
        return iso_name
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"No FinViz industry file found for {date_} "
        f"(looked for {iso_name.name} and {legacy.name})")

def sector_file(date_: dt.date) -> Path:
    """
    FinViz sector‑level snapshot for `date_`.
    Looks for  Data/YYYY‑MM‑DD‑sectors.json   then legacy M.D‑sectors.json.
    """
    iso = DATA_DIR / f"{date_}-sectors.json"
    leg = DATA_DIR / f"{date_.month}.{date_.day}-sectors.json"
    if iso.exists(): return iso
    if leg.exists(): return leg
    raise FileNotFoundError(f"No sector snapshot for {date_}")

def _save_snapshot(df: pd.DataFrame, day: dt.date) -> None:
    """
    Persist today's ranked table as JSON:  Data/YYYY‑MM‑DD-top.json
    """
    snap_path = DATA_DIR / f"{day.isoformat()}-top.json"
    df.to_json(snap_path, orient="records", indent=2)

def _get_leaders_for(day: dt.date) -> pd.DataFrame:
    """
    Return the Top‑N industry table for `day`, using the in‑memory cache
    if present; otherwise run rank_industries_enhanced() exactly once.
    """
    key = day.isoformat()
    if key not in _DAY_CACHE:
        fv_path = fv_file_for(day)
        if not fv_path.exists():
            raise FileNotFoundError(f"No FinViz file for {day}")
        _DAY_CACHE[key] = rank_industries_enhanced(fv_path, today=day)
    return _DAY_CACHE[key]


def _compute_trend_score(recent: list[pd.DataFrame]) -> pd.Series:
    """
    Simple method: for each industry count how many times it APPEARED and its
    AVERAGE rank improvement over the look‑back window.
        trend_score = appearance_pct * rank_improvement_pct
    Returns a Series aligned by industry name (0‑100 scale).
    """
    if not recent:
        return pd.Series(dtype=float)

    first, last = recent[0], recent[-1]
    all_names   = set(first.industry) | set(last.industry)
    scores = {}
    for ind in all_names:
        # initial / final rank (None means not in Top‑10 that day)
        r0 = first.loc[first.industry == ind, 'rank'].squeeze() if ind in first.industry.values else 11
        r1 = last .loc[last .industry == ind, 'rank'].squeeze() if ind in last .industry.values else 11
        # positive number means rank got better (moved closer to #1)
        rank_delta = (r0 - r1)
        # presence ratio
        presence = sum(ind in df.industry.values for df in recent) / len(recent)
        scores[ind] = presence * rank_delta   # can be negative
    # scale to 0‑100 percentile
    return _scale_rank(pd.Series(scores), ascending=True)

def _load_recent_snapshots(days: int) -> list[pd.DataFrame]:
    """
    Return a list of DataFrames (oldest → newest) for the last `days` calendar
    days where snapshot files exist.
    """
    dfs = []
    for i in range(days, 0, -1):
        day = dt.date.today() - dt.timedelta(days=i)
        p   = DATA_DIR / f"{day.isoformat()}-top.json"
        if p.exists():
            dfs.append(pd.read_json(p))
    return dfs

def _follow_through_day(symbol: str, day: dt.date, lookback: int = 20) -> bool:
    """
    CAN‑SLIM style follow‑through check:

        • Find the lowest CLOSE in the last `lookback` sessions
        • Day 3‑10 *after* that low must close > 1 % above the prior day
          on higher volume.
    """
    bars = _get_bars(symbol,
                     day - dt.timedelta(days=lookback + 10),
                     day,
                     limit=lookback + 10)
    if bars is None or bars.empty:
        return False

    closes, vols = bars.close, bars.volume

    # position (integer) of the look‑back low, not the label
    low_pos = closes[-lookback:].argmin() + (len(closes) - lookback)

    # scan the next 3‑10 bars
    for pos in range(low_pos + 3, min(low_pos + 11, len(bars))):
        if (closes.iloc[pos] > 1.01 * closes.iloc[pos - 1] and
                vols.iloc[pos]   >       vols.iloc[pos - 1]):
            return True
    return False


def _safe_float(x, default: float = 0.0):
    try: return float(str(x).replace('%','').replace('M','').replace('B',''))
    except (TypeError, ValueError): return default

def _scale_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending).mul(100)

def _show_pct_bar(pct: float, width: int = 40) -> None:
    """
    Print an in‑place ASCII progress bar, e.g.  [########------] 45 %
    Call repeatedly with pct in 0‑100.  Finishes with a newline at 100 %.
    """
    pct = max(0, min(100, pct))
    filled = int(width * pct // 100)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if pct >= 100 else "\r"
    print(f"[{bar}] {pct:3.0f} %", end=end, flush=True)

# ──────────────  risk sizing helpers (price/volume only)  ──────────────
def _atr(bars, period: int = 14) -> float:
    """
    Average‑True‑Range based on daily OHLCV bars.
    Uses a simple rolling mean of the true‑range series.
    """
    high, low, close = bars.high, bars.low, bars.close
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def _atr_levels(entry: float, atr: float, regime: str, *, side: str = "long"):
    """
    Convert ATR to stop / target using regime‑specific multipliers.
    Returns: stop_price, target_price
    """
    mults = ATR_PARAMS.get(regime, ATR_PARAMS["sideways"])
    s_mult, t_mult = mults["stop_mult"], mults["target_mult"]

    if side == "long":
        stop   = round(entry - atr * s_mult, 2)
        target = round(entry + atr * t_mult, 2)
    else:  # short
        stop   = round(entry + atr * s_mult, 2)
        target = round(entry - atr * t_mult, 2)

    return stop, target

def _trade_plan(trigger: float, atr: float, risk_mult: float = 1.2):
    """
    ATR‑based uniform‑risk levels:
      • buy zone  = trigger ± 0.5 ATR
      • stop loss = trigger − 1.2 ATR
      • targets   = trigger + {2, 3} ATR
    Returns: buyLow, buyHigh, stopLoss, targetOne, targetTwo
    """
    buy_low   = round(trigger - 0.5 * atr, 2)
    buy_high  = round(trigger + 0.5 * atr, 2)
    stop_loss = round(trigger - risk_mult * atr, 2)
    tgt_one   = round(trigger + 2 * atr, 2)
    tgt_two   = round(trigger + 3 * atr, 2)
    return buy_low, buy_high, stop_loss, tgt_one, tgt_two

def _long_signal_dict(ticker: str, entry: float, atr_val: float) -> dict:
    """
    Return a dict with buyLow/buyHigh/stopLoss/targetOne/targetTwo plus price.
    """
    buy_low, buy_high, stop, tgt1, tgt2 = _trade_plan(entry, atr_val)
    return dict(
        ticker      = ticker,
        price_now   = entry,
        buyLow      = buy_low,
        buyHigh     = buy_high,
        stopLoss    = stop,
        targetOne   = tgt1,
        targetTwo   = tgt2
    )

# ---------------------------------------------------------------------------
def _compute_momentum(df: pd.DataFrame) -> pd.Series:
    # Uses FinViz perf columns already converted to float (−1..+1)
    weights = {"Perf_Week": 0.25, "Perf_Month": 0.30,
               "Perf_Quart": 0.25, "Perf_Half": 0.15,
               "Perf_Year": 0.05}
    mom_raw = sum(df[col] * w for col, w in weights.items())
    return _scale_rank(mom_raw, ascending=True)          # higher better

# ---------------------------------------------------------------------------
def _compute_breadth(df: pd.DataFrame) -> pd.Series:
    # breadth column should already be 0..1 (% of stocks near highs)
    breadth_pct = df["breadth"].clip(0, 1) * 100
    return breadth_pct                                  # already 0‑100

# ---------------------------------------------------------------------------
def _compute_volume_accum(industry_map: dict, lookback_days: int = 20) -> pd.Series:
    """
    Returns a pd.Series indexed by industry with volume‑accum score (0‑100).
    Simple version: Up‑volume / Down‑volume ratio percentile.
    """
    scores = {}
    for ind, tickers in industry_map.items():
        up_vol = down_vol = 0
        for t in tickers[:10]:                      # limit to 10 symbols for speed
            bars = _get_bars_lookback(t, lookback_days)
            if bars is None or bars.empty: continue
            # up vs down volume
            up_vol   += bars[bars.close > bars.close.shift()].volume.sum()
            down_vol += bars[bars.close < bars.close.shift()].volume.sum()
        if down_vol == 0:
            score = 100 if up_vol > 0 else 0
        else:
            ratio = up_vol / down_vol
            score = ratio          # will be ranked later
        scores[ind] = score
    return _scale_rank(pd.Series(scores), ascending=True)

# ---------------------------------------------------------------------------
def _compute_fundamental(df_val: pd.DataFrame) -> pd.Series:
    """
    Simple composite:  (low Fwd_PE  + low PEG)  and  high EPS_next_5Y growth.
    Returns 0‑100 score per industry.
    """
    df = df_val.copy()
    df["Fwd_PE"] = df["Fwd_PE"].apply(_safe_float)
    df["PEG"]    = df["PEG"].apply(_safe_float)
    df["EPS_next_5Y"] = df["EPS_next_5Y"].apply(_safe_float)

    # lower PE & PEG better ⇒ ascending=False
    pe_score  = _scale_rank(-df["Fwd_PE"],  ascending=True)
    peg_score = _scale_rank(-df["PEG"],     ascending=True)
    eps_score = _scale_rank(df["EPS_next_5Y"], ascending=True)

    comp = (pe_score + peg_score + eps_score) / 3
    return comp


# ═══════════════════ 1) MACRO TAPE CHECK ════════════════════════════════════
def macro_ok(day: dt.date) -> bool:
    """
    Confirms a *bull* tape before allowing long setups:
      1) SPY, QQQ, MDY, IWM **and** equal‑weight RSP all > 200‑dma
      2) At least one of those ETFs printed a valid follow‑through day
    """
    etfs = ["SPY", "QQQ", "MDY", "IWM", "RSP"]
    bars = {e: _get_bars(e, day - dt.timedelta(days=250), day, limit=250)
            for e in etfs}
    if any(b is None or b.empty for b in bars.values()):
        return False

    # 200‑day moving‑average test
    for b in bars.values():
        if b.close[-1] <= b.close.rolling(200).mean().iloc[-1]:
            return False

    # follow‑through‑day test – pass if ANY index triggers
    return any(_follow_through_day(e, day) for e in etfs)


def market_regime(day: dt.date) -> str:
    """
    Returns 'bull', 'bear', or 'sideways' based on SPY/QQQ/IWM location
    vs. 50‑ & 200‑day SMAs.
    """
    spy = _get_bars("SPY", day - dt.timedelta(days=250), day, limit=250)
    qqq = _get_bars("QQQ", day - dt.timedelta(days=250), day, limit=250)
    iwm = _get_bars("IWM", day - dt.timedelta(days=250), day, limit=250)
    if any(x is None or x.empty for x in (spy, qqq, iwm)):
        return "unknown"

    def _trend(bar_df):
        c   = bar_df.close.iloc[-1]
        ma50  = _sma(bar_df.close, 50).iloc[-1]
        ma200 = _sma(bar_df.close, 200).iloc[-1]
        if c > ma50 > ma200:   return "up"
        if c < ma50 < ma200:   return "down"
        return "side"

    trends = [_trend(df) for df in (spy, qqq, iwm)]
    if trends.count("up")  >= 2: return "bull"
    if trends.count("down")>= 2: return "bear"
    return "sideways"


# ═══════════ 2) RANK INDUSTRIES FOR A GIVEN DAY ═════════════════════════════
_BAR_CACHE: dict[str, pd.DataFrame] = {}          # simple in‑memory cache

def rank_industries_enhanced(fv_path: Path, *, today: dt.date) -> pd.DataFrame:
    """
    Smart ranking (momentum + breadth + volume + fundamentals + trend‑boost).
    Robust to FinViz header variations and missing columns.
    """
    # 1) load & basic filters
    fin = pd.DataFrame(_load_json(fv_path))
    fin = fin[(fin["Rel_Volume"].astype(float) >= VOL_THRESHOLD) &
              (fin["Recom"].astype(float)      <= MAX_RECOM)].copy()
    fin.set_index("Name", inplace=True)

    # 2) normalise performance columns (creates zeros if absent)
    fin = _normalize_perf_cols(fin)

    # 3) sector‑relative adjustment (unchanged logic, still works)
    sector_map = _load_json(SECTOR_MAP_FILE) if SECTOR_MAP_FILE.exists() else {}
    fin["sector"] = fin.index.map(sector_map) if sector_map else None
    sec_df = None
    if RELATIVE and fin["sector"].notna().any():
        try:
            sec_df = pd.DataFrame(_load_json(sector_file(today))).set_index("Name")
            sec_df = _normalize_perf_cols(sec_df)
        except FileNotFoundError:
            print(f"[WARN] sector file missing for {today}; using raw momentum.")

    # 4) convert % strings → float & optionally make relative
    for col in LOOKBACKS:
        fin[col] = fin[col].apply(_pct_to_float)
        if RELATIVE and sec_df is not None:
            fin[col] -= fin["sector"].map(sec_df[col]).fillna(0)

    # 5) breadth (NEW helper call)
    fin["breadth"] = _extract_breadth(fin)

    # 6) fundamentals (neutral 50 when value file missing)
    val_path = value_file_for(today)
    if val_path.exists():
        df_val = pd.DataFrame(_load_json(val_path)).set_index("Name")
        fund_s = _compute_fundamental(df_val).reindex(fin.index).fillna(50)
    else:
        fund_s = pd.Series(50, index=fin.index)

    # 7) component scores
    mom_s  = _compute_momentum(fin)
    tech_s = _compute_breadth(fin)
    vol_s  = _compute_volume_accum(_load_json(IND_FILE), 20).reindex(fin.index).fillna(50)

    base   = W_MOM*mom_s + W_TECH*tech_s + W_VOL*vol_s + W_FUND*fund_s

    # 8) 7‑day trend boost (unchanged)
    recent = _load_recent_snapshots(TREND_LOOKBACK_DAYS)
    if len(recent) == TREND_LOOKBACK_DAYS:
        trend_s = _compute_trend_score(recent).reindex(fin.index).fillna(0)
        fin["pred_score"] = base + W_TREND*trend_s
    else:
        fin["pred_score"] = base

    # 9) final table
    fin.drop(columns=["sector"], inplace=True, errors="ignore")
    leaders = (fin.sort_values("pred_score", ascending=False)
                   .head(TOP_N_INDS)
                   .reset_index()
                   .rename(columns={"Name": "industry"}))
    leaders.insert(0, "rank", range(1, len(leaders)+1))
    _save_snapshot(leaders, today)
    return leaders[["rank","industry","pred_score","breadth"]]


def rank_industries(momentum_path: Path) -> pd.DataFrame:
    """
    Legacy simple ranking.  Now tolerates header variations and no‑breadth files.
    """
    raw = pd.DataFrame(_load_json(momentum_path))
    raw = raw[(raw["Rel_Volume"].astype(float) >= VOL_THRESHOLD) &
              (raw["Recom"].astype(float)      <= MAX_RECOM)].copy()

    raw = _normalize_perf_cols(raw)
    raw["breadth"] = _extract_breadth(raw)

    for col, w in LOOKBACKS.items():
        raw[col] = raw[col].apply(_pct_to_float)
    raw["mom_score"] = sum(raw[c]*w for c,w in LOOKBACKS.items()) + raw["breadth"]

    leaders = (raw.sort_values("mom_score", ascending=False)
                   .head(TOP_N_INDS)
                   .rename(columns={"Name": "industry"}))
    return leaders[["industry","mom_score","breadth"]]


# ═══════════ 3) SCREEN STOCKS ON A DAY ══════════════════════════════════════
def screen_on(day: dt.date, industry: str, tickers: list[str], regime: str) -> list[dict]:
    """
    Scan tickers for momentum burst or pull‑back patterns and build ATR‑sized
    stops/targets. Prints a short rejection log when nothing passes.
    """
    setups, rejected = [], {"nodata": 0, "thinvol": 0}
    for t in tickers:
        bars = _get_bars(t, day - dt.timedelta(days=120), day, limit=60)
        if bars is None:
            rejected["nodata"] += 1
            continue
        if _col(bars, "volume").tail(21).mean() < MIN_VOL:
            rejected["thinvol"] += 1
            continue

        price  = _col(bars, "close")
        volume = _col(bars, "volume")
        hi20   = price.tail(20).max()

        if _rs(price) < RS_MIN:
            continue
        ema20 = _ema(price, 20)

        burst = (_last(price) > 1.02 * hi20 and
                 volume.iloc[-1] > 2 * volume.tail(50).mean())

        pullbk = (_last(price) > ema20.iloc[-1] > ema20.iloc[-3] and
                  price.iloc[-2] < ema20.iloc[-2] and
                  volume.iloc[-1] > 1.5 * volume.tail(50).mean())

        if not (burst or pullbk):
            continue
        if _last(price) < 0.90 * price.tail(50).max():
            continue

        atr_val = _atr(bars)
        stop, target = _atr_levels(_last(price), atr_val, regime, side="long")
        setups.append(dict(
            date=str(day), ticker=t, industry=industry,
            pattern="burst" if burst else "pullback",
            entry=float(_last(price)), target=target, stop=stop
        ))

    if not setups:
        print(f"[{industry}] rejected – nodata:{rejected['nodata']} "
              f"thin‑vol:{rejected['thinvol']}")
    return setups

def rank_weak_industries(momentum_path: Path) -> pd.DataFrame:
    """Mirror of rank_industries but returns the BOTTOM‑N laggards."""
    raw = pd.DataFrame(_load_json(momentum_path))
    raw = raw[(raw["Rel_Volume"].astype(float) >= VOL_THRESHOLD) &
              (raw["Recom"].astype(float)      >= MAX_RECOM)].copy()

    for col, w in LOOKBACKS.items():
        raw[col] = raw[col].apply(_pct_to_float)
    raw["mom_score"] = sum(raw[c]*w for c,w in LOOKBACKS.items())
    laggards = (raw.sort_values("mom_score", ascending=True)     # ascending!
                   .head(BOTTOM_N_INDS)
                   .rename(columns={"Name":"industry"}))
    return laggards[["industry","mom_score"]]

def screen_short(day: dt.date, industry: str, tickers: list[str], regime: str) -> list[dict]:
    shorts = []
    for t in tickers:
        bars = _get_bars(t, day - dt.timedelta(days=120), day, limit=60)
        if bars is None or _col(bars, "volume").tail(21).mean() < MIN_VOL:
            continue

        price  = _col(bars, "close")
        volume = _col(bars, "volume")
        ema20, ema50 = _ema(price, 20), _ema(price, 50)

        if _rs(price) > RS_WEAK_MAX:
            continue

        bkdwn  = (_last(price) < 0.96 * price.tail(20).min() and
                  volume.iloc[-1] > 1.5 * volume.tail(50).mean())

        pullup = (_last(price) < ema20.iloc[-1] < ema50.iloc[-1] and
                  price.iloc[-2] > ema20.iloc[-2] and
                  (_last(price) / price.iloc[-2] - 1) <= -0.03)

        if not (bkdwn or pullup):
            continue

        atr_val = _atr(bars)
        stop, target = _atr_levels(_last(price), atr_val, regime, side="short")
        shorts.append(dict(
            date=str(day), ticker=t, industry=industry,
            pattern="breakdown" if bkdwn else "pullup‑short",
            side="short", entry=float(_last(price)), target=target, stop=stop
        ))
    return shorts

def screen_mean_reversion(day: dt.date, universe, regime: str) -> list[dict]:
    setups = []
    for t in universe:
        bars = _get_bars(t, day - dt.timedelta(days=30), day, limit=30)
        if bars is None or _col(bars, "volume").tail(21).mean() < MIN_VOL:
            continue

        price  = _col(bars, "close")
        volume = _col(bars, "volume")
        sma20  = price.rolling(20).mean().iloc[-1]
        deviation = (_last(price) - sma20) / sma20
        atr_val = _atr(bars)

        # extreme reversion
        if deviation > MEAN_REV_EXTREME and _last(price) < price.iloc[-2]:
            stop, target = _atr_levels(_last(price), atr_val, regime, side="short")
            setups.append(dict(date=str(day), ticker=t, industry="N/A",
                               pattern="mean‑rev‑short", side="short",
                               entry=float(_last(price)), target=target, stop=stop))
            continue
        if deviation < -MEAN_REV_EXTREME and _last(price) > price.iloc[-2]:
            stop, target = _atr_levels(_last(price), atr_val, regime, side="long")
            setups.append(dict(date=str(day), ticker=t, industry="N/A",
                               pattern="mean‑rev‑long", side="long",
                               entry=float(_last(price)), target=target, stop=stop))
            continue

        # daily‑VWAP bounce (requires vwap column added earlier)
        if "vwap" in bars.columns:
            if (price.iloc[-2] < bars.vwap.iloc[-2] and
                _last(price)   > bars.vwap.iloc[-1] and
                volume.iloc[-1] > 1.2 * volume.tail(20).mean()):
                stop, target = _atr_levels(_last(price), atr_val, regime, side="long")
                setups.append(dict(date=str(day), ticker=t, industry="N/A",
                                   pattern="VWAP‑bounce", side="long",
                                   entry=float(_last(price)), target=target, stop=stop))
    return setups

def generate_signals_for_industry(day: dt.date,
                                  industry: str,
                                  tickers: list[str],
                                  regime: str) -> list[dict]:
    raw_setups = screen_on(day, industry, tickers, regime)
    signals = []
    for s in raw_setups:
        bars = _get_bars(s["ticker"], day - dt.timedelta(days=60), day, limit=30)
        if bars is None or bars.empty:
            continue
        atr_val = _atr(bars)
        sig = _long_signal_dict(s["ticker"], s["entry"], atr_val)
        signals.append(sig)

    if signals:
        # ---------- save TradingView text -------------------------------
        tv_lines = []
        for sig in signals:
            tv_lines.append(f'"{sig["ticker"]}" =>')
            tv_lines.append(f'    buyLow := {sig["buyLow"]}')
            tv_lines.append(f'    buyHigh := {sig["buyHigh"]}')
            tv_lines.append(f'    stopLoss := {sig["stopLoss"]}')
            tv_lines.append(f'    targetOne := {sig["targetOne"]}')
            tv_lines.append(f'    targetTwo := {sig["targetTwo"]}\n')
        tv_content = "\n".join(tv_lines)
        fname = day.strftime("%m-%d-%Y-Signals.txt")
        (SIGNALS_TV_DIR / fname).write_text(tv_content)

        # ---------- save JSON history -----------------------------------
        hist_fname = day.strftime("%m-%d-%Y-Signals.json")
        (SIGNALS_HISTORY_DIR / hist_fname).write_text(json.dumps(signals, indent=2))

    return signals

# ───────────────────────────────────────────────────────── back‑test
def backtest_range(start: dt.date, end: dt.date):
    try:
        fv_path = finviz_file(start)        
    except FileNotFoundError as err:
        print(err); return

    ind_map = _load_json(IND_FILE)   # industry → tickers
    leaders = rank_industries_enhanced(fv_path, today=start)
    lead_inds = set(leaders.industry)
    if leaders.empty:
        print("Momentum filter produced no industries."); return

    # -------- generate all entry signals on every day in window ----------
    all_trades = []
    day = start
    while day <= end:
        regime = market_regime(day)
        if regime == "bull":
            if not macro_ok(day):
                day += dt.timedelta(days=1); continue
            for ind in lead_inds:
                all_trades.extend(screen_on(day, ind, ind_map.get(ind, []), regime))

        elif regime == "bear":
            weak = rank_weak_industries(fv_path)
            for ind in weak.industry:
                all_trades.extend(screen_short(day, ind, ind_map.get(ind, []), regime))

        else:  # sideways
            universe = itertools.chain.from_iterable(ind_map.values())
            all_trades.extend(screen_mean_reversion(day, universe, regime))
        day += dt.timedelta(days=1)

# ═══════════ 5) INTERACTIVE MENU ════════════════════════════════════════════
def menu():
    MENU = """
════════════════════════════════════════════════════════
Industry Momentum Screener
  1) Identify top industries (today)
  2) Find setups within specified industry (today)
  3) Identify top industries & scan for trade setups (today)
  4) Back‑test industry‑driven trade setups (date range)
  5) View historically generated trade signals
  0) Quit
════════════════════════════════════════════════════════
Selection: """
    while True:
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye"); sys.exit(0)

        today = dt.date.today()                     # used by several branches

        # ---------- Option 1 : top‑10 ------------------------------------------------
        if choice == "1":
            try:
                leaders = _get_leaders_for(today)
            except FileNotFoundError as e:
                print(e); continue

            print("\nTOP‑10 INDUSTRIES (Today)\n")
            print(leaders.to_string(index=False,
                  formatters={'pred_score':"{:.1f}".format,
                              'breadth':   "{:.0%}".format}))

            recent = _load_recent_snapshots(TREND_LOOKBACK_DAYS)
            if len(recent) == TREND_LOOKBACK_DAYS:
                trend = (_compute_trend_score(recent)
                         .sort_values(ascending=False)
                         .head(10)
                         .reset_index())
                trend.columns = ["industry", "trend_score"]
                print("\n7‑DAY UPWARD TREND (Top movers)\n")
                print(trend.to_string(index=False,
                                      formatters={'trend_score':"{:.1f}".format}))
            else:
                need = TREND_LOOKBACK_DAYS - len(recent)
                print(f"\n(need {need} more daily snapshots to build trend table)")

        # ---------- Option 2 : setups in one industry -------------------------------
        elif choice == "2":
            try:
                leaders = _get_leaders_for(today)
            except FileNotFoundError as e:
                print(e); continue

            ind_map = _load_json(IND_FILE)
            regime  = market_regime(today)

            if regime == "bull" and not macro_ok(today):
                print("Broad market conditions not favourable for longs today.")
                continue

            while True:
                print("\nSelect an industry to view trade setups:\n")
                for i, ind in enumerate(leaders.industry, 1):
                    print(f"  {i}) {ind}")
                sel = input("\nEnter number (or name)   (ENTER = back): ").strip()
                if not sel:
                    break

                if sel.isdigit() and 1 <= int(sel) <= len(leaders):
                    industry = leaders.industry.iloc[int(sel) - 1]
                else:
                    industry = sel
                    if industry not in leaders.industry.values:
                        print("Industry not recognised. Try again.")
                        continue

                tickers = ind_map.get(industry, [])
                if not tickers:
                    print("No tickers found for that industry."); continue

                signals = generate_signals_for_industry(today, industry, tickers, regime)
                if not signals:
                    print("No valid trade setups found.")
                    continue

                df_sig = pd.DataFrame(signals)
                print(f"\nTrade setups for {industry} ({today}):\n")
                print(df_sig[["ticker","price_now","buyLow","buyHigh",
                               "stopLoss","targetOne","targetTwo"]]
                      .to_string(index=False))

                print("\nSignals saved to:")
                print("  •", (SIGNALS_TV_DIR / today.strftime("%m-%d-%Y-Signals.txt")).name)
                print("  •", (SIGNALS_HISTORY_DIR / today.strftime("%m-%d-%Y-Signals.json")).name)

                again = input("\nView another industry? [y/N] ").strip().lower()
                if again != "y":
                    break

        # ---------- Option 3 : full scan today --------------------------------------
        elif choice == "3":
            backtest_range(today, today)            # performs scan & prints warnings

        # ---------- Option 4 : back‑test range --------------------------------------
        elif choice == "4":
            s = input("Start YYYY‑MM‑DD: ").strip()
            e = input("End   YYYY‑MM‑DD: ").strip()
            try:
                start = dt.datetime.strptime(s, "%Y-%m-%d").date()
                end   = dt.datetime.strptime(e, "%Y-%m-%d").date()
                if start > end: raise ValueError
            except ValueError:
                print("Invalid date(s)."); continue
            backtest_range(start, end)

        # ---------- Option 5 : view historical signal files -------------------------
        elif choice == "5":
            files = _list_signal_files()
            if not files:
                print("No historical signal files found."); continue

            while True:
                print("\nAvailable signal files:\n")
                for i, p in enumerate(files, 1):
                    print(f"  {i}) {p.name}")
                sel = input("\nEnter number (or file name)  (ENTER = back): ").strip()
                if not sel:
                    break

                if sel.isdigit() and 1 <= int(sel) <= len(files):
                    chosen = files[int(sel) - 1]
                else:
                    chosen = SIGNALS_HISTORY_DIR / sel
                    if not chosen.exists():
                        print("File not found. Try again."); continue

                try:
                    df = _load_signals(chosen)
                except Exception as exc:
                    print(f"Could not read file: {exc}")
                    continue

                print(f"\nSignals in {chosen.name}:\n")
                print(df.to_string(index=False))

                again = input("\nView another file? [y/N] ").strip().lower()
                if again != "y":
                    break

        # ---------- Quit -------------------------------------------------------------
        elif choice == "0":
            break
        else:
            print("Invalid selection.")



# ═══════════ 6) MAIN ════════════════════════════════════════════════════════
if __name__ == "__main__":
    menu()
