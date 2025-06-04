#!/usr/bin/env python3
"""
Swing Trading Strategy Script with Feature Refinement and Backtesting
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
import time # Added for Urwid get_prices error handling

try:
    from watchlist_utils import load_watchlist, save_watchlist, manage_watchlist
except ImportError:
    if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] == '--run-tests'):
        print("Warning: Could not import watchlist_utils. Watchlist features might be limited.")
        def load_watchlist(): return ["AAPL", "MSFT"]
        def save_watchlist(w): pass
        def manage_watchlist(): pass

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from ta.volume import MFIIndicator
from ta.volatility import DonchianChannel
from sklearn.model_selection import train_test_split


from colorama import init, Fore, Style
init(autoreset=True)

PREDICTIONS_FILE = "weekly_signals.json"
horizon = 10

DEFAULT_INTERVALS_CONFIG = {
    '5m':  {'period': '14d',  'interval': '5m'},
    '30m': {'period': '60d',  'interval': '30m'},
    '1h':  {'period': '120d', 'interval': '1h'},
    '90m': {'period': '60d',  'interval': '90m'},
    '1d':  {'period': '380d', 'interval': '1d'}
}
DATA_CACHE: dict[tuple, tuple] = {}

def get_or_fetch(ticker: str, start=None, end=None):
    key = (ticker, start, end)
    if key not in DATA_CACHE:
        DATA_CACHE[key] = fetch_data(ticker, start=start, end=end)
    return DATA_CACHE[key]

def batch_fetch_data_for_tickers(
    tickers: list[str], requested_intervals: list[str] | None = None, cache_dir: str = ".ohlcv_cache",
    start_date_override_1d: str | None = None, end_date_override_1d: str | None = None
) -> dict:
    Path(cache_dir).mkdir(exist_ok=True)
    all_data = {ticker: {} for ticker in tickers}
    intervals_to_fetch = DEFAULT_INTERVALS_CONFIG.keys()
    if requested_intervals and len(requested_intervals) > 0:
        intervals_to_fetch = [i for i in requested_intervals if i in DEFAULT_INTERVALS_CONFIG]

    if not intervals_to_fetch: print("Warning: No valid intervals to fetch specified."); return all_data

    for interval_name in intervals_to_fetch:
        params = DEFAULT_INTERVALS_CONFIG[interval_name]
        batch_cache_filename_parts = sorted(list(set(tickers)))
        use_override_for_1d = interval_name == '1d' and start_date_override_1d and end_date_override_1d
        cache_file_name = Path(cache_dir) / (f"{'_'.join(batch_cache_filename_parts)}_1d_{start_date_override_1d}_{end_date_override_1d}.pkl" if use_override_for_1d else f"{'_'.join(batch_cache_filename_parts)}_{interval_name}_{params['period']}_{params['interval']}.pkl")

        if cache_file_name.exists():
            try:
                interval_batch_data = joblib.load(cache_file_name)
                for ticker_val in tickers:
                    if ticker_val in interval_batch_data: all_data[ticker_val][interval_name] = interval_batch_data[ticker_val]
                if all(interval_name in all_data[t] for t in tickers): continue
            except Exception as e: print(f"Error loading cache file {cache_file_name}: {e}. Refetching."); cache_file_name.unlink(missing_ok=True)
        try:
            bulk_data = safe_download(tickers, start=start_date_override_1d, end=end_date_override_1d, interval=params['interval'], auto_adjust=True, progress=False, threads=True) if use_override_for_1d else safe_download(tickers, period=params['period'], interval=params['interval'], auto_adjust=True, progress=False, threads=True)
        except Exception as e: print(f"Error downloading {interval_name} for {tickers}: {e}"); bulk_data = pd.DataFrame()
        if bulk_data.empty:
            for ticker_val in tickers: all_data[ticker_val][interval_name] = pd.DataFrame(); continue
        current_interval_data_to_cache = {}
        for ticker_val in tickers:
            try:
                df = bulk_data.xs(ticker_val, level=1, axis=1).copy() if len(tickers) > 1 and isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data.copy()
                if 'Adj Close' in df.columns and 'Close' not in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col not in df.columns: df[col] = np.nan
                df.dropna(how='all', inplace=True)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                all_data[ticker_val][interval_name] = df; current_interval_data_to_cache[ticker_val] = df
            except KeyError: all_data[ticker_val][interval_name] = pd.DataFrame()
            except Exception as e: print(f"Error processing {ticker_val} for {interval_name}: {e}"); all_data[ticker_val][interval_name] = pd.DataFrame()
        if current_interval_data_to_cache:
            try: joblib.dump(current_interval_data_to_cache, cache_file_name)
            except Exception as e: print(f"Error saving cache file {cache_file_name}: {e}")
    for ticker_val in tickers:
        for interval_name_val in intervals_to_fetch:
            if interval_name_val not in all_data[ticker_val]: all_data[ticker_val][interval_name_val] = pd.DataFrame()
    return all_data

def load_predictions():
    if not os.path.isfile(PREDICTIONS_FILE): return []
    try:
        with open(PREDICTIONS_FILE, "r") as fh: data = json.load(fh)
    except Exception: return []
    for rec in data: rec.setdefault("status", "Open"); rec.setdefault("entry_date", datetime.datetime.today().strftime("%Y-%m-%d"))
    return data

def save_predictions(pred_list):
    with open(PREDICTIONS_FILE, "w") as fh: json.dump(pred_list, fh, indent=2)

def _update_positions_status() -> None:
    preds = load_predictions(); changed = False
    for p in preds:
        if p.get('status') != 'Open': continue
        try: now_price = yf.Ticker(p['symbol']).history(period='1d')['Close'].iloc[-1]
        except Exception: continue
        hit_stop = (p['direction'] == 'LONG' and now_price <= p['stop_loss']) or (p['direction'] == 'SHORT' and now_price >= p['stop_loss'])
        hit_target = (p['direction'] == 'LONG' and now_price >= p['profit_target']) or (p['direction'] == 'SHORT' and now_price <= p['profit_target'])
        if hit_stop or hit_target: p['status'] = 'Stop' if hit_stop else 'Target'; p['exit_date'] = datetime.date.today().isoformat(); p['exit_price'] = round(float(now_price), 2); changed = True
    if changed: save_predictions(preds)

def safe_download(*args, **kwargs):
    import time; from yfinance import exceptions as yf_exceptions
    for attempt in range(5):
        try: return yf.download(*args, **kwargs)
        except yf_exceptions.YFRateLimitError: wait = 2**attempt; print(f"⚠️  Yahoo rate-limit – retrying in {wait}s …"); time.sleep(wait)
    return yf.download(*args, **kwargs)

def _download_cached(ticker: str, period: str, interval: str, cache_dir: str = ".ohlcv_cache") -> pd.DataFrame:
    Path(cache_dir).mkdir(exist_ok=True); fname = Path(cache_dir) / f"{ticker}_{period}_{interval}.pkl"
    sec_per_bar = {"5m": 300, "30m": 1800, "1h": 3600, "90m": 5400, "1d": 86400}; max_age = sec_per_bar.get(interval, 3600) * 6
    if fname.exists():
        try:
            cached: pd.DataFrame = joblib.load(fname)
            if not cached.empty and (pd.Timestamp.utcnow() - cached.index[-1]).total_seconds() < max_age: return cached
        except Exception: fname.unlink(missing_ok=True)
    fresh = safe_download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if isinstance(fresh.columns, pd.MultiIndex): fresh.columns = fresh.columns.droplevel(1)
    if 'Adj Close' in fresh.columns and 'Close' not in fresh.columns: fresh.rename(columns={'Adj Close': 'Close'}, inplace=True)
    if fname.exists():
        try: old = joblib.load(fname); fresh = pd.concat([old, fresh]).sort_index().drop_duplicates()
        except Exception: pass
    if fresh.index.tz is not None: fresh.index = fresh.index.tz_localize(None)
    joblib.dump(fresh, fname); return fresh

def preload_interval_cache(symbols: list[str], period: str, interval: str, cache_dir: str = ".ohlcv_cache") -> None:
    symbols = sorted(set(symbols));
    if not symbols: return
    bulk = safe_download(symbols, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if bulk.empty: return
    if isinstance(bulk.columns, pd.MultiIndex): bulk = bulk.swaplevel(axis=1).sort_index(axis=1)
    Path(cache_dir).mkdir(exist_ok=True)
    for sym in symbols:
        df = bulk[sym].dropna(how="all")
        if not df.empty:
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            joblib.dump(df, Path(cache_dir) / f"{sym}_{period}_{interval}.pkl")

def fetch_data(ticker, start=None, end=None, intervals=None, warmup_days=300):
    if intervals is None: intervals = DEFAULT_INTERVALS_CONFIG
    dfs = {}
    for key, params in intervals.items():
        if key == '1d' and start and end:
            real_start = (pd.to_datetime(start) - datetime.timedelta(days=warmup_days)).strftime('%Y-%m-%d')
            dfs[key] = safe_download(ticker, start=real_start, end=end, interval='1d', auto_adjust=True, progress=False)
        else: dfs[key] = _download_cached(ticker, params['period'], params['interval'])
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) > 1: df.columns = df.columns.droplevel(1)
        if 'Adj Close' in df.columns and 'Close' not in df.columns: df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        df.dropna(how='all', inplace=True);
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    return tuple(_clean(dfs.get(k, pd.DataFrame())) for k in DEFAULT_INTERVALS_CONFIG.keys())

def compute_indicators(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    df = df.copy(); required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not required_cols.issubset(df.columns) or len(df) < 2:
        stub_cols = [f"RSI_{timeframe}", f"ADX_{timeframe}", f"ADX_pos_{timeframe}", f"ADX_neg_{timeframe}", f"STOCHk_{timeframe}", f"STOCHd_{timeframe}", f"CCI_{timeframe}", f"MACD_{timeframe}", f"MACD_signal_{timeframe}", f"MACD_hist_{timeframe}", f"BB_upper_{timeframe}", f"BB_lower_{timeframe}", f"BB_middle_{timeframe}", f"ATR_{timeframe}", f"EMA20_{timeframe}", f"EMA50_{timeframe}", f"EMA100_{timeframe}", f"EMA200_{timeframe}", f"SUPERT_{timeframe}", f"SUPERTd_{timeframe}", f"SUPERTl_{timeframe}", f"SUPERTs_{timeframe}", f"DCL_{timeframe}", f"DCM_{timeframe}", f"DCU_{timeframe}"]
        for col in stub_cols: df[col] = np.nan
        return df
    window = 14
    if len(df) < window:
        for col_suffix in ["", "d", "l", "s"]: df[f"SUPERT{col_suffix}_{timeframe}"] = np.nan
        for col_prefix in ["DCL", "DCM", "DCU"]: df[f"{col_prefix}_{timeframe}"] = np.nan
    df[f"RSI_{timeframe}"] = RSIIndicator(close=df["Close"], window=window).rsi()
    adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=window); df[f"ADX_{timeframe}"] = adx_ind.adx(); df[f"ADX_pos_{timeframe}"] = adx_ind.adx_pos(); df[f"ADX_neg_{timeframe}"] = adx_ind.adx_neg()
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=window, smooth_window=3); df[f"STOCHk_{timeframe}"] = stoch.stoch(); df[f"STOCHd_{timeframe}"] = stoch.stoch_signal()
    df[f"CCI_{timeframe}"] = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20, constant=0.015).cci()
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9); df[f"MACD_{timeframe}"] = macd.macd(); df[f"MACD_signal_{timeframe}"] = macd.macd_signal(); df[f"MACD_hist_{timeframe}"] = macd.macd_diff()
    df[f"EMA20_{timeframe}"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df[f"EMA50_{timeframe}"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df[f"EMA100_{timeframe}"] = EMAIndicator(close=df["Close"], window=100).ema_indicator()
    df[f"EMA200_{timeframe}"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()
    supertrend_df = df.ta.supertrend(length=10, multiplier=3)
    if supertrend_df is not None and not supertrend_df.empty and isinstance(supertrend_df, pd.DataFrame) and len(supertrend_df.columns) >= 4:
        df[f"SUPERT_{timeframe}"] = supertrend_df.iloc[:,0]; df[f"SUPERTd_{timeframe}"] = supertrend_df.iloc[:,1]; df[f"SUPERTl_{timeframe}"] = supertrend_df.iloc[:,2]; df[f"SUPERTs_{timeframe}"] = supertrend_df.iloc[:,3]
    else:
        for col_suffix in ["", "d", "l", "s"]: df[f"SUPERT{col_suffix}_{timeframe}"] = np.nan
    if timeframe in {"1h", "daily", "hourly"}:
        donchian = DonchianChannel(high=df["High"], low=df["Low"], close=df["Close"], window=20); df[f"DCL_{timeframe}"] = donchian.donchian_channel_lband(); df[f"DCM_{timeframe}"] = donchian.donchian_channel_mband(); df[f"DCU_{timeframe}"] = donchian.donchian_channel_hband()
    else:
        for col_prefix in ["DCL", "DCM", "DCU"]: df[f"{col_prefix}_{timeframe}"] = np.nan
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2); df[f"BB_upper_{timeframe}"] = bb.bollinger_hband(); df[f"BB_lower_{timeframe}"] = bb.bollinger_lband(); df[f"BB_middle_{timeframe}"] = bb.bollinger_mavg()
    df[f"ATR_{timeframe}"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=window).average_true_range()
    return df

def compute_anchored_vwap(df: pd.DataFrame, lookback_bars: int = 252) -> pd.Series:
    if df.empty or not {"Close", "Low", "Volume"}.issubset(df.columns): return pd.Series(dtype=float, index=df.index)
    recent = df.tail(lookback_bars); anchor = recent['Low'].idxmin()
    if pd.isna(anchor): return pd.Series(np.nan, index=df.index)
    after = df.loc[anchor:].copy(); cum_vol = after['Volume'].cumsum(); cum_dollars = (after['Close'] * after['Volume']).cumsum()
    vwap_series = cum_dollars / cum_vol.replace(0, np.nan); out = pd.Series(np.nan, index=df.index)
    out.loc[vwap_series.index] = vwap_series; out.name = 'AnchoredVWAP'; return out

def to_daily(intra_df, label):
    if intra_df.empty: return pd.DataFrame()
    daily_data = intra_df.groupby(intra_df.index.date).tail(1)
    daily_data.index = pd.to_datetime(daily_data.index.date); daily_data.index.name = 'Date'
    return daily_data

def prepare_features(df_5m: pd.DataFrame, df_30m: pd.DataFrame, df_1h: pd.DataFrame, df_90m: pd.DataFrame, df_1d: pd.DataFrame, *, horizon: int = 10, drop_recent: bool = True) -> pd.DataFrame:
    ind_5m = compute_indicators(df_5m.copy(), timeframe='5m'); ind_30m = compute_indicators(df_30m.copy(), timeframe='30m')
    ind_1h = compute_indicators(df_1h.copy(), timeframe='hourly'); ind_90m = compute_indicators(df_90m.copy(), timeframe='90m')
    ind_1d = compute_indicators(df_1d.copy(), timeframe='daily')
    ind_5m['AnchoredVWAP_5m'] = compute_anchored_vwap(ind_5m, lookback_bars=2000); ind_30m['AnchoredVWAP_30m'] = compute_anchored_vwap(ind_30m, lookback_bars=200)
    ind_1h['AnchoredVWAP_1h'] = compute_anchored_vwap(ind_1h, lookback_bars=120); ind_1d['AnchoredVWAP'] = compute_anchored_vwap(ind_1d, lookback_bars=252)
    daily_5m = to_daily(ind_5m, "5m"); daily_30m = to_daily(ind_30m, "30m"); daily_1h = to_daily(ind_1h, "1h"); daily_90m = to_daily(ind_90m, "90m")
    ind_1d.index.name = 'Date'
    features_df = ind_1d.join(daily_5m, rsuffix='_5m').join(daily_30m, rsuffix='_30m').join(daily_1h, rsuffix='_1h').join(daily_90m, rsuffix='_90m')
    features_df.dropna(subset=['Close'], inplace=True)
    if features_df.empty: return features_df
    atr = features_df['ATR_daily'].fillna(0); upper = features_df['Close'] + 2.0 * atr; lower = features_df['Close'] - 2.0 * atr
    closes = features_df['Close'].values; labels = np.ones(len(features_df), dtype=int)
    for i in range(len(features_df) - horizon):
        win = closes[i + 1 : i + 1 + horizon]; up = np.where(win >= upper.iloc[i])[0]; dn = np.where(win <= lower.iloc[i])[0]
        if up.size and dn.size: labels[i] = 2 if up[0] < dn[0] else 0
        elif up.size: labels[i] = 2
        elif dn.size: labels[i] = 0
    features_df['future_class'] = labels
    if drop_recent: features_df = features_df.iloc[:-horizon]
    else: features_df.loc[features_df.index[-horizon:], 'future_class'] = np.nan
    return features_df

def prepare_features_intraday(df_30m: pd.DataFrame) -> pd.DataFrame:
    df_30m = df_30m.copy(); df_30m = compute_indicators(df_30m, timeframe='intraday')
    df_30m['AnchoredVWAP_30m'] = compute_anchored_vwap(df_30m, lookback_bars=200)
    if df_30m.empty or 'Close' not in df_30m.columns: return pd.DataFrame()
    daily = to_daily(df_30m, "intraday"); daily = compute_indicators(daily, timeframe='daily'); daily['AnchoredVWAP'] = compute_anchored_vwap(daily, lookback_bars=252)
    if daily.index.tz is not None: daily.index = daily.index.tz_localize(None)
    if df_30m.index.tz is not None: df_30m.index = df_30m.index.tz_localize(None)
    df_30m = df_30m.join(daily.reindex(df_30m.index, method='ffill'), rsuffix='_daily')
    horizon_bars = 16; atr_col = 'ATR_intraday'
    if atr_col not in df_30m.columns: df_30m[atr_col] = df_30m['Close'].rolling(14).std().fillna(0)
    up = df_30m['Close'] + 2 * df_30m[atr_col]; dn = df_30m['Close'] - 2 * df_30m[atr_col]; cls = df_30m['Close'].values; lbl = np.ones(len(df_30m), int)
    for i in range(len(df_30m) - horizon_bars):
        win = cls[i+1 : i+1+horizon_bars]; a = np.where(win >= up.iloc[i])[0]; b = np.where(win <= dn.iloc[i])[0]
        if a.size and b.size: lbl[i] = 2 if a[0] < b[0] else 0
        elif a.size: lbl[i] = 2
        elif b.size: lbl[i] = 0
    df_30m['future_class'] = lbl; df_30m = df_30m.iloc[:-horizon_bars]
    return df_30m

def refine_features(features_df, importance_cutoff=0.0001, corr_threshold=0.9):
    if features_df.empty or 'future_class' not in features_df.columns: return features_df
    y = features_df['future_class']; X = features_df.drop(columns=['future_class']).copy()
    X = X.fillna(method='ffill').fillna(method='bfill'); X.replace([np.inf, -np.inf], np.nan, inplace=True); X = X.fillna(0)
    if X.empty: return features_df
    model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
    split_idx = int(0.8 * len(X));
    if split_idx < 1 : return X.join(y)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    if len(np.unique(y_train.dropna())) < 2: return X.join(y)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    drop_by_importance = importances[importances < importance_cutoff].index.tolist()
    if drop_by_importance: X.drop(columns=drop_by_importance, inplace=True, errors='ignore')
    if X.empty: return features_df
    numeric_cols = X.select_dtypes(include=[np.number]).columns; corr_matrix = X[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_by_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    if drop_by_corr: X.drop(columns=drop_by_corr, inplace=True, errors='ignore')
    return X.join(y)

def tune_threshold_and_train(features_df, ret_horizon=5):
    if features_df.empty or 'Close' not in features_df.columns:
        return None, 0.03

    feature_cols = [c for c in features_df.columns if c != 'future_class' and c != 'future_return']

    features_df['future_return'] = features_df['Close'].shift(-ret_horizon) / features_df['Close'] - 1.0
    operational_df = features_df.dropna(subset=['future_return', 'future_class'])

    if operational_df.empty:
        X_full_fallback = features_df[feature_cols].copy()
        X_full_fallback = X_full_fallback.fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], 0).fillna(0)
        y_full_fallback = features_df['future_class'].fillna(1)
        if X_full_fallback.empty or y_full_fallback.isnull().all() or len(np.unique(y_full_fallback.dropna())) < 1:
            return None, 0.03
        model = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
        model.fit(X_full_fallback, y_full_fallback)
        return model, 0.03

    X = operational_df[feature_cols].copy()
    X = X.fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], 0).fillna(0)

    if X.empty:
        X_full_fallback = features_df[feature_cols].copy().fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], 0).fillna(0)
        y_full_fallback = features_df['future_class'].fillna(1)
        if X_full_fallback.empty or y_full_fallback.isnull().all() or len(np.unique(y_full_fallback.dropna())) < 1: return None, 0.03
        model = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
        model.fit(X_full_fallback, y_full_fallback)
        return model, 0.03

    aligned_future_returns = operational_df.loc[X.index, 'future_return']

    if len(X) < 2 or len(np.unique(operational_df.loc[X.index, 'future_class'].dropna())) < 2 :
        X_train, X_test, future_ret_train, future_ret_test = X, X, aligned_future_returns, aligned_future_returns
    else:
        try:
            X_train, X_test, future_ret_train, future_ret_test = train_test_split(
                X, aligned_future_returns, test_size=0.2, random_state=42,
                stratify=operational_df.loc[X.index, 'future_class'].fillna(1)
            )
        except ValueError:
             X_train, X_test, future_ret_train, future_ret_test = train_test_split(
                X, aligned_future_returns, test_size=0.2, random_state=42)

    best_f1 = -1
    best_thr = 0.03

    thresholds_to_test = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    for thr in thresholds_to_test:
        y_train_temp = np.ones(len(future_ret_train))
        y_train_temp[future_ret_train > thr] = 2
        y_train_temp[future_ret_train < -thr] = 0

        df_train_temp = X_train.copy()
        df_train_temp['temp_class'] = y_train_temp

        majority_class_size = df_train_temp['temp_class'].value_counts().max()
        resampled_dfs = []
        for class_val in [0, 1, 2]:
            df_class = df_train_temp[df_train_temp['temp_class'] == class_val]
            if len(df_class) == 0: continue
            if len(df_class) == majority_class_size :
                resampled_dfs.append(df_class)
            else:
                resampled_dfs.append(resample(df_class, replace=True, n_samples=majority_class_size, random_state=42))

        if not resampled_dfs: continue

        df_resampled = pd.concat(resampled_dfs)
        X_train_resampled = df_resampled.drop(columns=['temp_class'])
        y_train_resampled = df_resampled['temp_class']

        if len(np.unique(y_train_resampled)) < 2: continue

        model_temp = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
        model_temp.fit(X_train_resampled, y_train_resampled)

        y_pred_probs_test = model_temp.predict_proba(X_test)
        y_pred_test = np.argmax(y_pred_probs_test, axis=1)

        y_test_actual_thr = np.ones(len(future_ret_test))
        y_test_actual_thr[future_ret_test > thr] = 2
        y_test_actual_thr[future_ret_test < -thr] = 0

        f1_long = f1_score(y_test_actual_thr, y_pred_test, labels=[2], average='macro', zero_division=0)
        f1_short = f1_score(y_test_actual_thr, y_pred_test, labels=[0], average='macro', zero_division=0)
        avg_f1 = (f1_long + f1_short) / 2.0

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thr = thr

    final_y = np.ones(len(operational_df))
    final_y[operational_df['future_return'] > best_thr] = 2
    final_y[operational_df['future_return'] < -best_thr] = 0

    df_final_train = X.copy()
    df_final_train['final_class'] = final_y

    majority_class_size_final = df_final_train['final_class'].value_counts().max()
    resampled_dfs_final = []
    for class_val in [0, 1, 2]:
        df_class_final = df_final_train[df_final_train['final_class'] == class_val]
        if len(df_class_final) == 0: continue
        if len(df_class_final) == majority_class_size_final:
             resampled_dfs_final.append(df_class_final)
        else:
            resampled_dfs_final.append(resample(df_class_final, replace=True, n_samples=majority_class_size_final, random_state=42))

    if not resampled_dfs_final:
        X_full_fallback = features_df[feature_cols].copy().fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], 0).fillna(0)
        y_full_fallback = features_df['future_class'].fillna(1)
        model = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
        model.fit(X_full_fallback, y_full_fallback)
        return model, 0.03

    df_final_resampled = pd.concat(resampled_dfs_final)
    X_final_resampled = df_final_resampled.drop(columns=['final_class'])
    y_final_resampled = df_final_resampled['final_class']

    if len(np.unique(y_final_resampled)) < 1:
        X_full_fallback = features_df[feature_cols].copy().fillna(method='ffill').fillna(method='bfill').replace([np.inf, -np.inf], 0).fillna(0)
        y_full_fallback = features_df['future_class'].fillna(1)
        model = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
        model.fit(X_full_fallback, y_full_fallback)
        return model, 0.03

    final_model = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, tree_method='hist')
    final_model.fit(X_final_resampled, y_final_resampled)

    if 'future_return' in features_df.columns:
        del features_df['future_return']

    return final_model, best_thr


def generate_signal_output(ticker, latest_row, model, thr):
    if model is None: return "Model not available"
    latest_df = latest_row.to_frame().T

    trained_feature_names = model.get_booster().feature_names
    latest_df_for_pred = latest_df[trained_feature_names].copy()

    latest_df_for_pred = latest_df_for_pred.fillna(method='ffill').fillna(method='bfill'); latest_df_for_pred.replace([np.inf, -np.inf], 0, inplace=True); latest_df_for_pred = latest_df_for_pred.fillna(0)

    probs = model.predict_proba(latest_df_for_pred)[0]; class_idx = int(np.argmax(probs)); prob_score = probs[class_idx]

    min_signal_prob = 0.60
    if prob_score < min_signal_prob or class_idx == 1: return None

    direction = "LONG" if class_idx == 2 else "SHORT"; price = float(latest_row.get("Close", np.nan))
    atr = latest_row.get("ATR_daily", np.nan); atr_note = ""
    if pd.isna(atr) or atr == 0 : atr = 0.05 * price if not pd.isna(price) else 0.01; atr_note = " (est. ATR)"
    if pd.isna(price): return "Price is NaN, cannot generate signal."
    stop = price - atr if direction == "LONG" else price + atr; target = price + 2 * atr if direction == "LONG" else price - 2 * atr
    ema20_daily = latest_row.get('EMA20_daily', np.nan); ema50_daily = latest_row.get('EMA50_daily', np.nan); ema100_daily = latest_row.get('EMA100_daily', np.nan); ema200_daily = latest_row.get('EMA200_daily', np.nan); supert_d_daily = latest_row.get('SUPERTd_daily', np.nan)
    trend_score = 0; trend_details = []
    if not pd.isna(price) and not pd.isna(ema20_daily) and not pd.isna(ema50_daily):
        if price > ema20_daily > ema50_daily: trend_score += 1; trend_details.append("P>E20>E50")
    if not pd.isna(ema50_daily) and not pd.isna(ema100_daily) and not pd.isna(ema200_daily):
        if ema50_daily > ema100_daily > ema200_daily: trend_score += 1; trend_details.append("E50>E100>E200")
    if not pd.isna(price) and not pd.isna(ema20_daily) and not pd.isna(ema50_daily):
        if price < ema20_daily < ema50_daily: trend_score -= 1; trend_details.append("P<E20<E50")
    if not pd.isna(ema50_daily) and not pd.isna(ema100_daily) and not pd.isna(ema200_daily):
        if ema50_daily < ema100_daily < ema200_daily: trend_score -= 1; trend_details.append("E50<E100<E200")
    if not pd.isna(supert_d_daily):
        if supert_d_daily == 1: trend_score += 1; trend_details.append("ST_Up")
        elif supert_d_daily == -1: trend_score -= 1; trend_details.append("ST_Down")
    trend_description = "Strong Uptrend" if trend_score >= 2 else "Weak Uptrend" if trend_score == 1 else "Strong Downtrend" if trend_score <= -2 else "Weak Downtrend" if trend_score == -1 else "Neutral/Choppy"
    trend_rationale_str = f"{trend_description} ({', '.join(trend_details)})" if trend_details else trend_description
    colour = Fore.GREEN if direction == "LONG" else Fore.RED
    return (f"{Fore.CYAN}{ticker}{Style.RESET_ALL}: {colour}{direction}{Style.RESET_ALL} " f"at ${price:.2f}, Stop ${stop:.2f}, Target ${target:.2f}, " f"Prob {prob_score:.2f} (Thr {thr:.2f}) – {trend_rationale_str}{atr_note}")

# --- Backtesting Functions ---
def _summarise_performance(trades_df: pd.DataFrame, total_days: int) -> dict:
    if trades_df.empty:
        return {"total": 0, "win_rate": 0, "avg_pnl": 0, "sharpe": 0, "max_dd": 0, "tim": 0}
    trades_df['pnl_pct'] = trades_df['pnl_pct'].astype(float)
    wins = trades_df[trades_df['pnl_pct'] > 0]
    num_trades = len(trades_df)
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_pnl = trades_df['pnl_pct'].mean()
    std_dev_pnl = trades_df['pnl_pct'].std()
    sharpe = avg_pnl / std_dev_pnl if std_dev_pnl != 0 else 0

    trades_df['timestamp'] = pd.to_datetime(trades_df['exit_timestamp'])
    trades_df.sort_values('timestamp', inplace=True)
    trades_df['equity_curve'] = (1 + trades_df['pnl_pct']).cumprod()
    rolling_max = trades_df['equity_curve'].cummax()
    drawdown = trades_df['equity_curve'] / rolling_max - 1.0
    max_dd = drawdown.min()

    trades_df['trade_duration'] = (trades_df['exit_timestamp'] - trades_df['entry_timestamp']).dt.total_seconds()
    total_trade_duration_seconds = trades_df['trade_duration'].sum()
    total_period_seconds = total_days * 24 * 60 * 60
    time_in_market = total_trade_duration_seconds / total_period_seconds if total_period_seconds > 0 else 0

    return {"total": num_trades, "win_rate": win_rate, "avg_pnl": avg_pnl, "sharpe": sharpe, "max_dd": max_dd, "tim": time_in_market}

def backtest_strategy(ticker, start_date, end_date, macro_df: pd.DataFrame | None = None, log_file=None):
    df_5m, df_30m, df_1h, df_90m, df_1d = fetch_data(ticker, start=start_date, end=end_date)
    feat = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, drop_recent=False)
    if feat.empty: print(f"No features for {ticker}, skipping backtest."); return

    feat_refined = refine_features(feat.dropna(subset=['future_class']))
    if feat_refined.empty or 'future_class' not in feat_refined.columns: print(f"Feature refinement failed or no valid data for {ticker}. Skipping backtest."); return

    model, best_thr = tune_threshold_and_train(feat_refined.copy(), ret_horizon=horizon)

    if model is None: print(Fore.YELLOW + f"Model training failed for {ticker}." + Style.RESET_ALL); return

    trained_feature_names = model.get_booster().feature_names
    X_all_candidate = feat.drop(columns=['future_class','future_return'], errors='ignore')

    for col in trained_feature_names:
        if col not in X_all_candidate.columns:
            X_all_candidate[col] = np.nan

    X_all = X_all_candidate[trained_feature_names].copy()
    X_all = X_all.fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_all.replace([np.inf, -np.inf], 0, inplace=True)

    preds_probs = model.predict_proba(X_all)
    preds_classes = np.argmax(preds_probs, axis=1)

    df_pred = X_all.copy()
    df_pred['prediction'] = preds_classes
    df_pred['probability'] = [preds_probs[i, c] for i, c in enumerate(preds_classes)]

    cols_to_add_from_feat = ['Close']
    if 'ATR_daily' in feat.columns: cols_to_add_from_feat.append('ATR_daily')

    for col_to_add in cols_to_add_from_feat:
        if col_to_add in feat.columns:
            if X_all.index.equals(feat.index):
                df_pred[col_to_add] = feat[col_to_add]
            else:
                df_pred = df_pred.join(feat[[col_to_add]], how='left')
        else:
            df_pred[col_to_add] = np.nan

    trades = []; in_trade = False; trade_dir = None; entry_price = 0.0; stop_price = 0.0; target_price = 0.0; entry_time = None
    for i in range(len(df_pred)):
        ts = df_pred.index[i]; row = df_pred.iloc[i]; price = row['Close']
        atr = row.get('ATR_daily', np.nan)
        if pd.isna(atr) or atr == 0: atr = 0.05 * price
        if pd.isna(price): continue
        if in_trade:
            pnl = 0.0; exit_trade = False
            if trade_dir == "LONG":
                if price >= target_price: exit_trade = True; pnl = (target_price - entry_price) / entry_price
                elif price <= stop_price: exit_trade = True; pnl = (stop_price - entry_price) / entry_price
            else:
                if price <= target_price: exit_trade = True; pnl = (entry_price - target_price) / entry_price
                elif price >= stop_price: exit_trade = True; pnl = (entry_price - stop_price) / entry_price
            if exit_trade:
                trades.append({'entry_timestamp': entry_time, 'exit_timestamp': ts, 'direction': trade_dir, 'entry_price': round(entry_price, 2), 'exit_price': round(price,2), 'pnl_pct': round(pnl, 4), 'stop_price': round(stop_price, 2), 'target_price': round(target_price, 2)})
                in_trade = False
            if in_trade: continue

        class_idx = int(row['prediction']); prob_score = row['probability']
        min_signal_prob_backtest = 0.5
        if prob_score < min_signal_prob_backtest or class_idx == 1 : continue

        if class_idx == 2:
            in_trade = True; trade_dir = "LONG"; entry_time = ts; entry_price = price; stop_price = price - atr; target_price = price + 2 * atr
        elif class_idx == 0:
            in_trade = True; trade_dir = "SHORT"; entry_time = ts; entry_price = price; stop_price = price + atr; target_price = price - 2 * atr

    if not trades: print(Fore.YELLOW + f"No trades for {ticker}." + Style.RESET_ALL); return
    trades_df = pd.DataFrame(trades); total_days_in_data = (df_pred.index.max() - df_pred.index.min()).days
    summary = _summarise_performance(trades_df, total_days_in_data if total_days_in_data > 0 else 1)
    pct = lambda x: f"{x*100:.2f}%"
    print(Fore.BLUE + f"\nBacktest Results for {ticker}" + Style.RESET_ALL + f" ({start_date} → {end_date}):")
    print(f"  Total trades        : {Fore.CYAN}{summary['total']}{Style.RESET_ALL}"); print(f"  Win rate            : {Fore.CYAN}{pct(summary['win_rate'])}{Style.RESET_ALL}"); print(f"  Avg P/L per trade   : {Fore.CYAN}{pct(summary['avg_pnl'])}{Style.RESET_ALL}"); print(f"  Sharpe (trade)      : {Fore.CYAN}{summary['sharpe']:.2f}{Style.RESET_ALL}"); print(f"  Max drawdown        : {Fore.CYAN}{pct(summary['max_dd'])}{Style.RESET_ALL}"); print(f"  Time-in-market      : {Fore.CYAN}{pct(summary['tim'])}{Style.RESET_ALL}")
    if log_file: trades_df.to_csv(log_file, mode='a', header=not log_file.tell() > 0, index=False)

def backtest_strategy_intraday(ticker, start_date, end_date, macro_df: pd.DataFrame | None = None, log_file=None):
    _, df_30m, _, _, _ = fetch_data(ticker, start=start_date, end=end_date)
    feat = prepare_features_intraday(df_30m)
    if feat.empty: print(f"No features for {ticker} (intraday), skipping backtest."); return

    feat_refined = refine_features(feat.dropna(subset=['future_class']))
    if feat_refined.empty or 'future_class' not in feat_refined.columns: print(f"Feature refinement failed for {ticker} (intraday). Skipping backtest."); return

    model, best_thr = tune_threshold_and_train(feat_refined.copy(), ret_horizon=16)
    if model is None: print(Fore.YELLOW + f"Model training failed for {ticker} (intraday)." + Style.RESET_ALL); return

    trained_feature_names = model.get_booster().feature_names
    X_all_candidate = feat.drop(columns=['future_class', 'future_return'], errors='ignore')
    for col in trained_feature_names:
        if col not in X_all_candidate.columns: X_all_candidate[col] = np.nan
    X_all = X_all_candidate[trained_feature_names].copy()
    X_all = X_all.fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_all.replace([np.inf, -np.inf], 0, inplace=True)

    preds_probs = model.predict_proba(X_all)
    preds_classes = np.argmax(preds_probs, axis=1)

    df_pred = X_all.copy()
    df_pred['prediction'] = preds_classes
    df_pred['probability'] = [preds_probs[i,c] for i,c in enumerate(preds_classes)]

    cols_to_add_from_feat = ['Close']
    if 'ATR_intraday' in feat.columns: cols_to_add_from_feat.append('ATR_intraday')

    for col_to_add in cols_to_add_from_feat:
        if col_to_add in feat.columns:
            if X_all.index.equals(feat.index):
                df_pred[col_to_add] = feat[col_to_add]
            else:
                df_pred = df_pred.join(feat[[col_to_add]], how='left')
        else:
            df_pred[col_to_add] = np.nan

    trades = []; in_trade = False; trade_dir = None; entry_price = 0.0; stop_price = 0.0; target_price = 0.0; entry_time = None
    for i in range(len(df_pred)):
        ts = df_pred.index[i]; row = df_pred.iloc[i]; price = row['Close']
        atr = row.get('ATR_intraday', np.nan)
        if pd.isna(atr) or atr == 0: atr = 0.02 * price
        if pd.isna(price): continue
        if in_trade:
            pnl = 0.0; exit_trade = False
            if trade_dir == "LONG":
                if price >= target_price: exit_trade = True; pnl = (target_price - entry_price) / entry_price
                elif price <= stop_price: exit_trade = True; pnl = (stop_price - entry_price) / entry_price
            else:
                if price <= target_price: exit_trade = True; pnl = (entry_price - target_price) / entry_price
                elif price >= stop_price: exit_trade = True; pnl = (entry_price - stop_price) / entry_price
            if exit_trade:
                trades.append({'entry_timestamp': entry_time, 'exit_timestamp': ts, 'direction': trade_dir, 'entry_price': round(entry_price,2), 'exit_price': round(price,2), 'pnl_pct': round(pnl,4), 'stop_price': round(stop_price,2), 'target_price': round(target_price,2)})
                in_trade = False
            if in_trade: continue
        class_idx = int(row['prediction']); prob_score = row['probability']
        min_signal_prob_backtest = 0.5
        if prob_score < min_signal_prob_backtest or class_idx == 1 : continue
        if class_idx == 2: in_trade=True; trade_dir="LONG"; entry_time=ts; entry_price=price; stop_price=price-atr; target_price=price+2*atr
        elif class_idx == 0: in_trade=True; trade_dir="SHORT"; entry_time=ts; entry_price=price; stop_price=price+atr; target_price=price-2*atr
    if not trades: print(Fore.YELLOW + f"No intraday trades for {ticker}." + Style.RESET_ALL); return
    trades_df = pd.DataFrame(trades); total_days_in_data = (df_pred.index.max() - df_pred.index.min()).days
    summary = _summarise_performance(trades_df, total_days_in_data if total_days_in_data > 0 else 1)
    pct = lambda x: f"{x*100:.2f}%"
    print(Fore.BLUE + f"\nIntraday Backtest Results for {ticker}" + Style.RESET_ALL + f" ({start_date} → {end_date}):")
    print(f"  Total trades        : {Fore.CYAN}{summary['total']}{Style.RESET_ALL}"); print(f"  Win rate            : {Fore.CYAN}{pct(summary['win_rate'])}{Style.RESET_ALL}"); print(f"  Avg P/L per trade   : {Fore.CYAN}{pct(summary['avg_pnl'])}{Style.RESET_ALL}"); print(f"  Sharpe (trade)      : {Fore.CYAN}{summary['sharpe']:.2f}{Style.RESET_ALL}"); print(f"  Max drawdown        : {Fore.CYAN}{pct(summary['max_dd'])}{Style.RESET_ALL}"); print(f"  Time-in-market      : {Fore.CYAN}{pct(summary['tim'])}{Style.RESET_ALL}")
    if log_file: trades_df.to_csv(log_file, mode='a', header=not log_file.tell() > 0, index=False)

def schedule_signals_instructions():
    print("\n--- Schedule Signals Instructions ---")
    print("To schedule signals, you would typically set up a cron job.")
    print("Example cron job to run daily at a specific time (e.g., 9:00 AM):")
    print("0 9 * * * /usr/bin/python3 /path/to/your/axist-technical.py SYMBOL1 SYMBOL2")
    print("Ensure the path to the script and python interpreter are correct.")
    print("You might want to log output to a file: >> /path/to/your/cron_output.log 2>&1")
    print("-------------------------------------\n")

def run_signals_on_watchlist(): print("STUB: run_signals_on_watchlist called.")
def backtest_watchlist(): print("STUB: backtest_watchlist called.")

def show_signals_for_current_week():
    """
    Displays trading signals for the current week (Monday to Sunday).
    """
    all_predictions = load_predictions()

    if not all_predictions:
        print("No signals found.")
        return

    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    end_of_week = start_of_week + datetime.timedelta(days=6)

    current_week_signals = []
    for signal in all_predictions:
        if 'entry_date' not in signal:
            continue  # Skip signals without an entry date

        try:
            entry_date_str = signal['entry_date']
            # Handle potential datetime objects if load_predictions starts returning them
            if isinstance(entry_date_str, datetime.datetime):
                entry_date = entry_date_str.date()
            elif isinstance(entry_date_str, datetime.date):
                entry_date = entry_date_str
            else:
                entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()

            if start_of_week <= entry_date <= end_of_week:
                current_week_signals.append(signal)
        except ValueError:
            # Skip signals with invalid date format
            print(f"Warning: Skipping signal for {signal.get('symbol', 'Unknown')} due to invalid date format: {signal['entry_date']}")
            continue

    if not current_week_signals:
        print(f"No signals found for the current week ({start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}).")
        return

    print(f"--- Signals for Current Week ({start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}) ---")
    for signal in current_week_signals:
        direction_color = Fore.GREEN if signal.get('direction') == 'LONG' else Fore.RED
        direction_str = signal.get('direction', 'N/A')
        symbol_str = signal.get('symbol', 'N/A')
        entry_price_str = f"${signal.get('entry_price', 0.0):.2f}"
        stop_loss_str = f"${signal.get('stop_loss', 0.0):.2f}"
        profit_target_str = f"${signal.get('profit_target', 0.0):.2f}"
        entry_date_display_str = signal.get('entry_date', 'N/A')
        # If entry_date was converted, ensure it's formatted back to string for display
        if isinstance(entry_date_display_str, (datetime.date, datetime.datetime)):
            entry_date_display_str = entry_date_display_str.strftime("%Y-%m-%d")


        print(f"{symbol_str}: {direction_color}{direction_str}{Style.RESET_ALL} "
              f"at {entry_price_str}, Stop {stop_loss_str}, Target {profit_target_str} "
              f"(Entry: {entry_date_display_str})")

# --- Urwid Dashboard for Signals Performance ---
# Global/shared state for Urwid app
rec_cache_urwid = []
all_recs_urwid = []
open_tr_urwid = []
current_prices_urwid = {}
main_loop_urwid = None
# table_content_urwid = urwid.Text("") # Replaced by ListBox walker
list_walker_urwid = urwid.SimpleFocusListWalker([]) # For ListBox content

def get_prices_urwid(loop=None, user_data=None):
    global current_prices_urwid, open_tr_urwid
    if not open_tr_urwid: return

    symbols = sorted(list(set(r['symbol'] for r in open_tr_urwid)))
    if not symbols: return

    try:
        data = yf.download(symbols, period="1d", progress=False, auto_adjust=True)
        if data.empty: return

        for symbol in symbols:
            try:
                if len(symbols) == 1: # yf.download behaves differently for single ticker
                    current_prices_urwid[symbol] = data['Close'].iloc[-1]
                else:
                    current_prices_urwid[symbol] = data['Close'][symbol].iloc[-1]
            except (KeyError, IndexError):
                current_prices_urwid[symbol] = np.nan # Or some other placeholder
    except Exception as e: # Broad exception to catch various yfinance/network issues
        # Fallback or error indication for individual symbols if needed
        for symbol in symbols:
             if symbol not in current_prices_urwid: current_prices_urwid[symbol] = np.nan
        # Optionally log `e` or inform user in footer
        footer_text_urwid.set_text(f"Price fetch error: {str(e)[:50]}")


def build_table_urwid(loop=None, user_data=None):
    global rec_cache_urwid, open_tr_urwid, current_prices_urwid
    lines = []
    rec_cache_urwid = []

    if not open_tr_urwid:
        lines.append(urwid.Text("No open positions to display."))
        return lines

    header = [
        ("fixed", 10, urwid.Text(('header_bold', "Symbol"))),
        ("fixed", 12, urwid.Text(('header_bold', "Entry Date"))),
        ("fixed", 8, urwid.Text(('header_bold', "Dir."))),
        ("fixed", 10, urwid.Text(('header_bold', "Entry"))),
        ("fixed", 10, urwid.Text(('header_bold', "Stop"))),
        ("fixed", 10, urwid.Text(('header_bold', "Target"))),
        ("fixed", 10, urwid.Text(('header_bold', "Current"))),
        ("fixed", 10, urwid.Text(('header_bold', "P&L %"))),
        ("fixed", 12, urwid.Text(('header_bold', "Status")))
    ]
    lines.append(urwid.Columns(header, dividechars=1))

    for rec in open_tr_urwid:
        current_price = current_prices_urwid.get(rec['symbol'], np.nan)
        if isinstance(current_price, pd.Series): # Ensure current_price is scalar
            if not current_price.empty:
                current_price = current_price.item()
            else:
                current_price = np.nan

        pnl_pct = 0.0
        status_val = "Open"
        status_attr = "open"

        if not pd.isna(current_price):
            if rec['direction'] == 'LONG':
                pnl_pct = (current_price / rec['entry_price'] - 1) * 100
                if current_price <= rec['stop_loss']: status_val = "Stop"; status_attr = "hit"
                elif current_price >= rec['profit_target']: status_val = "Target"; status_attr = "hit"
            else: # SHORT
                pnl_pct = (rec['entry_price'] / current_price - 1) * 100
                if current_price >= rec['stop_loss']: status_val = "Stop"; status_attr = "hit"
                elif current_price <= rec['profit_target']: status_val = "Target"; status_attr = "hit"

        pnl_str = f"{pnl_pct:+.2f}%"
        pnl_attr = "positive" if pnl_pct > 0 else "negative" if pnl_pct < 0 else "default"

        row_data = [
            rec['symbol'], rec.get('entry_date', 'N/A')[:10], rec['direction'],
            f"{rec['entry_price']:.2f}", f"{rec['stop_loss']:.2f}", f"{rec['profit_target']:.2f}",
            f"{current_price:.2f}" if not pd.isna(current_price) else "N/A",
            (pnl_attr, pnl_str), (status_attr, status_val)
        ]
        rec_cache_urwid.append({'data': rec, 'current_status': status_val, 'current_price': current_price}) # Store for 'd' key

        columns = [
            ("fixed", 10, urwid.Text(row_data[0])),
            ("fixed", 12, urwid.Text(row_data[1])),
            ("fixed", 8, urwid.Text(row_data[2])),
            ("fixed", 10, urwid.Text(row_data[3])),
            ("fixed", 10, urwid.Text(row_data[4])),
            ("fixed", 10, urwid.Text(row_data[5])),
            ("fixed", 10, urwid.Text(row_data[6])),
            ("fixed", 10, urwid.Text(row_data[7])),
            ("fixed", 12, urwid.Text(row_data[8]))
        ]
        lines.append(urwid.AttrMap(urwid.Columns(columns, dividechars=1), None, focus_map={None: 'reversed'}))
    return lines

def refresh_urwid(loop=None, user_data=None):
    global list_walker_urwid, main_loop_urwid
    get_prices_urwid(loop, user_data)
    new_lines_widgets = build_table_urwid(loop, user_data)

    list_walker_urwid[:] = new_lines_widgets if new_lines_widgets else [urwid.Text("No open positions or error building table.")]

    if main_loop_urwid:
         main_loop_urwid.set_alarm_in(30, refresh_urwid)

def unhandled_input_urwid(key):
    global all_recs_urwid, open_tr_urwid, main_loop_urwid, rec_cache_urwid
    if key in ('q', 'Q'):
        raise urwid.ExitMainLoop()
    if key in ('r', 'R'):
        refresh_urwid(main_loop_urwid) # Pass main_loop for alarm scheduling
    if key in ('d', 'D'): # Deject (close out) hit trades
        changed = False
        today_iso = datetime.date.today().isoformat()
        for cached_item in rec_cache_urwid:
            if cached_item['current_status'] in ("Stop", "Target"):
                # Find corresponding record in all_recs_urwid and open_tr_urwid to update
                for rec in all_recs_urwid:
                    # Assuming symbol and entry_price and entry_date make it unique enough
                    if rec['symbol'] == cached_item['data']['symbol'] and \
                       rec['entry_price'] == cached_item['data']['entry_price'] and \
                       rec.get('entry_date', 'N/A')[:10] == cached_item['data'].get('entry_date', 'N/A')[:10] and \
                       rec['status'] == 'Open':
                        rec['status'] = cached_item['current_status']
                        rec['exit_price'] = cached_item['current_price']
                        rec['exit_date'] = today_iso
                        changed = True
                        break
        if changed:
            save_predictions(all_recs_urwid)
            # Update open_tr_urwid to reflect changes before refresh
            open_tr_urwid = [p for p in all_recs_urwid if p.get('status', 'Open') == 'Open']
            refresh_urwid(main_loop_urwid)
        return True
    return False

header_text_urwid = urwid.Text(('header', "Open Positions Performance Dashboard"), align='center')
footer_text_urwid = urwid.Text(('footer', "Q: Quit | R: Refresh | D: Deject Hit Trades (Save & Close)"))

def signals_performance_cli():
    global all_recs_urwid, open_tr_urwid, main_loop_urwid, table_content_urwid

    all_recs_urwid = load_predictions()
    open_tr_urwid = [p for p in all_recs_urwid if p.get('status', 'Open') == 'Open']

    if not open_tr_urwid:
        print("No open positions to display in dashboard.")
        return

    palette = [
        ('reversed', 'standout', ''),
        ('header', 'white', 'dark blue', 'bold'),
        ('header_bold', 'white', 'dark blue', 'bold,underline'),
        ('footer', 'white', 'dark blue'),
        ('positive', 'dark green', ''),
        ('negative', 'dark red', ''),
        ('hit', 'yellow', 'dark red'),
        ('open', 'light gray', ''),
        ('default', 'light gray', ''),
    ]

    # Initial data fetch and table build before starting loop
    get_prices_urwid()
    initial_lines_widgets = build_table_urwid()
    list_walker_urwid[:] = initial_lines_widgets if initial_lines_widgets else [urwid.Text("No open positions.")]

    body_listbox = urwid.ListBox(list_walker_urwid)

    layout = urwid.Frame(
        header=urwid.AttrMap(header_text_urwid, 'header'),
        body=body_listbox, # Use ListBox here
        footer=urwid.AttrMap(footer_text_urwid, 'footer')
    )

    main_loop_urwid = urwid.MainLoop(layout, palette, unhandled_input=unhandled_input_urwid)
    main_loop_urwid.set_alarm_in(0.1, refresh_urwid) # Initial refresh slightly delayed

    try:
        main_loop_urwid.run()
    except Exception as e:
        # This will print the error to stderr after Urwid exits, useful for debugging
        # In a real non-sandboxed environment, Urwid might print its own errors to screen.
        sys.stderr.write(f"Urwid dashboard error: {e}\n")
        sys.stderr.write("If display is garbled, your terminal might not support Urwid well or TERM var is not set.\n")


def closed_stats_cli(): print("STUB: closed_stats_cli called.")

def interactive_menu():
    while True:
        print("\nMain Menu:")
        print("1. Manage Watchlist"); print("2. Run Signals on Watchlist (Live Mode)"); print("3. Backtest All Watchlist Tickers"); print("4. Schedule Signals"); print("5. Show This Weeks Signals"); print("6. Show Latest Signals Performance"); print("7. Closed-Trades Statistics"); print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == '0': print("Exiting."); break
        elif choice == '1': manage_watchlist()
        elif choice == '2': run_signals_on_watchlist()
        elif choice == '3': backtest_watchlist()
        elif choice == '4': schedule_signals_instructions()
        elif choice == '5': show_signals_for_current_week()
        elif choice == '6': signals_performance_cli()
        elif choice == '7': closed_stats_cli()
        else: print("Invalid option. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Swing trading signal generator/backtester")
    parser.add_argument('tickers', nargs='*', help="List of stock ticker symbols to analyze")
    parser.add_argument('--log', action='store_true', help="Optionally log selected trades to a file")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode")
    parser.add_argument('--start', default=None, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument('--real', action='store_true', help="Use 30-min intraday backtest for max realism")
    args = parser.parse_args()
    action_arg_present = args.backtest
    if not args.tickers and not action_arg_present and not args.log:
        if not args.tickers : parser.print_help(); return
    if args.backtest:
        if not args.tickers or not args.start or not args.end: print("For backtesting, please provide ticker(s), start date, and end date."); return
        log_file_main = None
        if args.log: log_file_main = open("trades_log.csv", "a")
        for ticker in args.tickers:
            print(f"\n=== Backtesting {ticker} from {args.start} to {args.end} ===")
            if args.real: backtest_strategy_intraday(ticker, args.start, args.end, macro_df=pd.DataFrame(), log_file=log_file_main)
            else: backtest_strategy(ticker, args.start, args.end, macro_df=pd.DataFrame(), log_file=log_file_main)
        if log_file_main: log_file_main.close()
        return

    tickers_to_process = args.tickers
    if not tickers_to_process:
        tickers_to_process = load_watchlist()
        if not tickers_to_process: print("No tickers provided and watchlist is empty."); return
        print(f"Processing watchlist: {', '.join(tickers_to_process)}")

    all_ticker_data = batch_fetch_data_for_tickers(tickers_to_process)
    for ticker in tickers_to_process:
        print(f"\n=== Processing {ticker} (live signal mode) ===")
        ticker_data = all_ticker_data.get(ticker, {})
        df_5m, df_30m, df_1h, df_90m, df_1d = (ticker_data.get('5m', pd.DataFrame()), ticker_data.get('30m', pd.DataFrame()), ticker_data.get('1h', pd.DataFrame()), ticker_data.get('90m', pd.DataFrame()), ticker_data.get('1d', pd.DataFrame()))
        if df_1d.empty : print(f"No usable daily data for {ticker}, skipping."); continue

        features_df_live = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, drop_recent=False)
        if features_df_live.empty: print(f"Insufficient data for {ticker} after feature prep, skipping."); continue

        trainable_df = features_df_live.dropna(subset=['future_class'])

        if trainable_df.empty:
            print(f"Not enough historical data with defined outcomes to train model for {ticker}. Skipping."); continue

        refined_trainable_df = refine_features(trainable_df)
        if refined_trainable_df.empty:
            print(f"Feature refinement failed for training data on {ticker}. Skipping."); continue

        model, best_thr = tune_threshold_and_train(refined_trainable_df.copy(), ret_horizon=horizon)

        if model is None: print(f"Model training failed for {ticker}."); continue

        latest_features_for_signal = features_df_live.iloc[-1:].copy()
        signal_output = generate_signal_output(ticker, latest_features_for_signal.iloc[0], model, best_thr if best_thr is not None else 0.03)
        print(signal_output)

# ---- Unit Test Definitions Start ----
import unittest
class MockYF:
    def __init__(self, ticker_symbol=None): self.ticker = ticker_symbol
    def history(self, period, interval=None, start=None, end=None, **kwargs):
        data_size = 50
        if period == '1d' and start and end: data_size = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        elif period == '14d' and interval == '5m': data_size = 14 * 78
        elif period == '60d' and interval == '30m': data_size = 60 * 13
        elif period == '120d' and interval == '1h': data_size = 120 * 7
        freq = 'B' if interval == '1d' else '5T' if interval == '5m' else '30T' if interval == '30m' else 'H' if interval == '1h' else '90T'
        base_start_date = pd.Timestamp('2023-01-01')
        if start: base_start_date = pd.to_datetime(start)
        index = pd.date_range(start=base_start_date, periods=data_size, freq=freq)
        df = pd.DataFrame(index=index)
        df['Open'] = np.random.uniform(100, 200, size=data_size); df['Low'] = df['Open'] - np.random.uniform(0, 5, size=data_size); df['High'] = df['Open'] + np.random.uniform(0, 5, size=data_size); df['Close'] = np.random.uniform(df['Low'], df['High']); df['Volume'] = np.random.randint(10000, 1000000, size=data_size)
        return df
class MockYFinanceTickerGlobal:
    def __init__(self, ticker_symbol): self.mock_yf = MockYF(ticker_symbol)
    def history(self, period, interval=None, start=None, end=None): return self.mock_yf.history(period=period, interval=interval, start=start, end=end)
_original_yf_download = None; _original_yf_Ticker = None
def _mock_yf_download(*args, **kwargs):
    tickers = args[0]
    if isinstance(tickers, str): return MockYF(tickers).history(**kwargs)
    if len(tickers) == 1 : return MockYF(tickers[0]).history(**kwargs)
    all_dfs = {}
    for ticker_sym in tickers:
        df_temp = MockYF(ticker_sym).history(**kwargs); new_cols = pd.MultiIndex.from_product([[ticker_sym], df_temp.columns]); df_temp.columns = new_cols; all_dfs[ticker_sym] = df_temp
    if not all_dfs: return pd.DataFrame()
    final_df = pd.concat(all_dfs.values(), axis=1); final_df.columns = pd.MultiIndex.from_tuples([(col[1], col[0]) for col in final_df.columns])
    return final_df
class EmbeddedMockModel:
    def __init__(self): self.feature_names_in_ = None
    def predict_proba(self, X):
        if not isinstance(X, np.ndarray): X = X.values
        if hasattr(self, 'mock_proba_dist'): return np.array([self.mock_proba_dist] * len(X))
        return np.array([[0.1, 0.1, 0.8]] * len(X))
    def get_booster(self):
        class MockBooster:
            feature_names = ['mock_feature1', 'mock_feature2']
        return MockBooster()

class TestComputeIndicators(unittest.TestCase):
    def setUp(self):
        data_size = 250; self.df = pd.DataFrame({'Open': np.random.rand(data_size) * 100 + 100, 'High': np.random.rand(data_size) * 100 + 105, 'Low': np.random.rand(data_size) * 100 + 95, 'Close': np.random.rand(data_size) * 100 + 100, 'Volume': np.random.rand(data_size) * 1000 + 100}, index=pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(data_size)]))
        self.df['High'] = self.df[['Open', 'High', 'Close']].max(axis=1); self.df['Low'] = self.df[['Open', 'Low', 'Close']].min(axis=1)
    def test_compute_daily_indicators(self):
        timeframe = 'daily'; df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns); self.assertIn(f'EMA100_{timeframe}', df_processed.columns); self.assertIn(f'SUPERT_{timeframe}', df_processed.columns); self.assertIn(f'SUPERTd_{timeframe}', df_processed.columns); self.assertIn(f'DCL_{timeframe}', df_processed.columns)
        supert_d_values = df_processed[f'SUPERTd_{timeframe}'].dropna()
        if not supert_d_values.empty: self.assertTrue(all(val in [-1, 1] for val in supert_d_values))
    def test_compute_hourly_indicators(self):
        timeframe = 'hourly'; df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns); self.assertIn(f'DCL_{timeframe}', df_processed.columns)
        if not df_processed[f'DCL_{timeframe}'].dropna().empty: self.assertFalse(df_processed[f'DCL_{timeframe}'].isna().all())
    def test_compute_5m_indicators(self):
        timeframe = '5m'; df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns); self.assertIn(f'DCL_{timeframe}', df_processed.columns); self.assertTrue(df_processed[f'DCL_{timeframe}'].isna().all())
class TestPrepareFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls): global _original_yf_download, _original_yf_Ticker; _original_yf_download = yf.download; _original_yf_Ticker = yf.Ticker; yf.download = _mock_yf_download; yf.Ticker = MockYFinanceTickerGlobal
    @classmethod
    def tearDownClass(cls): global _original_yf_download, _original_yf_Ticker; yf.download = _original_yf_download; yf.Ticker = _original_yf_Ticker
    def setUp(self):
        data_size = 250; self.df_1d = _download_cached("TEST_D1", period="380d", interval="1d"); self.df_5m = _download_cached("TEST_5M", period="14d", interval="5m"); self.df_30m = _download_cached("TEST_30M", period="60d", interval="30m"); self.df_1h = _download_cached("TEST_1H", period="120d", interval="1h"); self.df_90m = _download_cached("TEST_90M", period="60d", interval="90m")
        if any(df.empty for df in [self.df_1d, self.df_5m, self.df_30m, self.df_1h, self.df_90m]): self.skipTest("Mock yf.download returned empty DataFrame in TestPrepareFeatures.setUp.")
    def test_prepare_features_columns(self):
        features_df = prepare_features(self.df_5m.copy(), self.df_30m.copy(), self.df_1h.copy(), self.df_90m.copy(), self.df_1d.copy(), horizon=5, drop_recent=True)
        if features_df.empty: self.fail("features_df is empty after prepare_features call.")
        self.assertIn('EMA20_daily', features_df.columns); self.assertIn('SUPERT_daily', features_df.columns); self.assertIn('DCL_daily', features_df.columns)
        self.assertIn('EMA20_5m', features_df.columns); self.assertIn('SUPERT_5m', features_df.columns)
        if 'DCL_5m' in features_df.columns: self.assertTrue(features_df['DCL_5m'].isna().all())
        self.assertIn('EMA20_hourly', features_df.columns); self.assertIn('SUPERT_hourly', features_df.columns); self.assertIn('DCL_hourly', features_df.columns)
class TestGenerateSignalOutput(unittest.TestCase):
    def setUp(self):
        self.mock_model = EmbeddedMockModel()
        self.mock_model.get_booster().feature_names = ['Close', 'EMA20_daily', 'EMA50_daily', 'EMA100_daily', 'EMA200_daily', 'SUPERTd_daily', 'ATR_daily', 'DCL_daily', 'DCM_daily', 'DCU_daily', 'EMA20_hourly', 'EMA50_hourly', 'SUPERTd_hourly']
        self.ticker = "TEST"
        self.base_latest_row = pd.Series({
            'Close': 100.0, 'EMA20_daily': 95.0, 'EMA50_daily': 90.0,
            'EMA100_daily': 85.0, 'EMA200_daily': 80.0, 'SUPERTd_daily': 1.0,
            'ATR_daily': 2.0, 'DCL_daily': 90.0, 'DCM_daily': 95.0, 'DCU_daily': 100.0,
            'EMA20_hourly': 96, 'EMA50_hourly':92, 'SUPERTd_hourly': 1
        })
        for feature in self.mock_model.get_booster().feature_names:
            if feature not in self.base_latest_row:
                self.base_latest_row[feature] = 0.0


    def test_strong_uptrend_long_signal(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]; latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal); self.assertIn("Strong Uptrend (P>E20>E50, E50>E100>E200, ST_Up)", signal)
        self.assertIn("Prob 0.80 (Thr 0.03)", signal)
    def test_no_signal_neutral_class_high_prob(self):
        self.mock_model.mock_proba_dist = [0.15, 0.7, 0.15]; latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNone(signal)
    def test_signal_uses_best_thr_in_output(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]; latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.05)
        self.assertIsNotNone(signal)
        self.assertIn("Prob 0.80 (Thr 0.05)", signal)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--run-tests':
        print("=== RUNNING EMBEDDED TESTS ===")
        _original_yf_download = yf.download; _original_yf_Ticker = yf.Ticker
        yf.download = _mock_yf_download; yf.Ticker = MockYFinanceTickerGlobal

        test_argv = [sys.argv[0]] + sys.argv[2:]
        suite = unittest.TestSuite(); loader = unittest.TestLoader()
        suite.addTest(loader.loadTestsFromTestCase(TestComputeIndicators)); suite.addTest(loader.loadTestsFromTestCase(TestPrepareFeatures)); suite.addTest(loader.loadTestsFromTestCase(TestGenerateSignalOutput))
        runner = unittest.TextTestRunner(verbosity=2); result = runner.run(suite)
        yf.download = _original_yf_download; yf.Ticker = _original_yf_Ticker
        if not result.wasSuccessful(): sys.exit(1)
        sys.exit(0)
    elif len(sys.argv) == 1:
        interactive_menu()
    else:
        main()
