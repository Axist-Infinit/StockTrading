#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import ta
from colorama import Fore, Style, init

# Attempt advanced libraries
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    import optuna
    from xgboost import XGBClassifier
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

init(autoreset=True)

CONFIG_FILE = "config.json"
DATA_DIR = "data_cache"
LOGFILE = "algobot.log"

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

###############################################################################
# 1. CONFIG + BASIC UTILS
###############################################################################

def remove_json_comments(json_str):
    """Remove single-line (//) and multi-line (/* */) comments."""
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    return json_str

def load_config(path=CONFIG_FILE):
    if not os.path.exists(path):
        print(Fore.RED + f"Config '{path}' not found." + Style.RESET_ALL)
        logging.error(f"Config '{path}' not found.")
        exit(1)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content_no_comments = remove_json_comments(content)
    return json.loads(content_no_comments)

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def force_datetime_index(df, date_col="Date"):
    """Convert a DataFrame to have a proper DatetimeIndex on 'Date' column."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.set_index(date_col, inplace=True, drop=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)
    return df

###############################################################################
# 2. YFINANCE DATA CACHING
###############################################################################

def get_cache_filepath(symbol, interval):
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

def download_and_cache_yf(symbol, period="5y", interval="1d"):
    """Download from yfinance, store to CSV, merge any existing data, return DataFrame."""
    yf_map = {"VIX": "^VIX"}
    yf_symbol = yf_map.get(symbol.upper(), symbol)
    fp = get_cache_filepath(symbol, interval)
    existing = None

    if os.path.exists(fp):
        try:
            existing = pd.read_csv(fp, parse_dates=["Date"])
            existing = force_datetime_index(existing, "Date")
        except Exception as e:
            print(Fore.RED + f"Error reading {fp}: {e}" + Style.RESET_ALL)
            logging.error(f"Error reading {fp}: {e}")
            existing = None

    # Attempt incremental update
    if existing is not None and not existing.empty:
        last_date = existing.index[-1]
        start_dt = last_date + timedelta(days=1)
        if start_dt > datetime.now():
            return existing
        new_data = yf.download(
            yf_symbol,
            start=start_dt,
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [
                "_".join(col).strip() for col in new_data.columns.values
            ]
        if new_data.empty:
            logging.info(f"No new data for {symbol}")
            return existing
        new_data.reset_index(inplace=True)
        new_data.dropna(how="any", inplace=True)
        combined = pd.concat([existing.reset_index(), new_data])
        combined.drop_duplicates(subset="Date", keep="last", inplace=True)
        combined = force_datetime_index(combined, "Date")
        combined.to_csv(fp, index=True, index_label="Date")
        return combined
    else:
        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip() for col in df.columns.values]
        if df.empty:
            logging.warning(f"No data for {symbol}")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.dropna(how="any", inplace=True)
        df = force_datetime_index(df, "Date")
        df.to_csv(fp, index=True, index_label="Date")
        return df

def load_timeframe_data(symbol, yf_period, yf_interval):
    try:
        df = download_and_cache_yf(symbol, period=yf_period, interval=yf_interval)
        if not df.empty:
            df = force_datetime_index(df)
        return df
    except Exception as e:
        print(Fore.RED + f"Error loading {symbol} ({yf_interval}): {e}" + Style.RESET_ALL)
        logging.error(f"Error loading {symbol} ({yf_interval}): {e}")
        return pd.DataFrame()

###############################################################################
# 3. FRED / MACRO
###############################################################################

def fetch_fred_series(series_id, start_date, end_date, api_key):
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id)
    df = pd.DataFrame(data, columns=[series_id])
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df.loc[start_date:end_date]

def load_macro_data(series_list, api_key, start_date, end_date):
    macro_df = pd.DataFrame()
    for sid in series_list:
        try:
            tmp = fetch_fred_series(sid, start_date, end_date, api_key)
            if macro_df.empty:
                macro_df = tmp
            else:
                macro_df = macro_df.join(tmp, how="outer")
            logging.info(f"Fetched FRED series '{sid}'.")
        except Exception as e:
            print(Fore.RED + f"Error fetching '{sid}': {e}" + Style.RESET_ALL)
            logging.error(f"Error fetching '{sid}': {e}")
    macro_df = macro_df.ffill().bfill()
    return macro_df

def yoy_change(series, shift_period=365):
    series = series.dropna()
    yoy = pd.Series(index=series.index, dtype=float)
    for dt in series.index:
        dt_m = dt - timedelta(days=shift_period)
        sub = series.loc[series.index <= dt_m]
        if sub.empty:
            yoy[dt] = np.nan
            continue
        ref_val = sub.iloc[-1]
        curr_val = series.loc[dt]
        yoy[dt] = (curr_val - ref_val) / abs(ref_val) * 100 if ref_val != 0 else np.nan
    return yoy

###############################################################################
# 4. TECHNICAL INDICATORS
###############################################################################

def compute_anchored_vwap(df, anchor_date=None):
    if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        df["anchored_vwap"] = np.nan
        return df
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    if df["Volume"].sum() == 0:
        df["anchored_vwap"] = np.nan
        return df
    if anchor_date is None and "Low" in df.columns:
        anchor_date = df["Low"].idxmin()
    if anchor_date not in df.index:
        mask = df.index >= anchor_date
        if not mask.any():
            df["anchored_vwap"] = np.nan
            return df
        anchor_date = df.index[mask][0]
    df["cum_pxvol"] = (df["Close"] * df["Volume"]).cumsum()
    df["cum_vol"] = df["Volume"].cumsum()
    try:
        anchor_loc = df.index.get_loc(anchor_date)
    except KeyError:
        df.drop(columns=["cum_pxvol", "cum_vol"], inplace=True)
        df["anchored_vwap"] = np.nan
        return df
    if anchor_loc == 0:
        df["anchored_vwap"] = df["cum_pxvol"] / df["cum_vol"]
    else:
        anchor_pxvol = df["cum_pxvol"].iloc[anchor_loc - 1]
        anchor_vol = df["cum_vol"].iloc[anchor_loc - 1]
        df["anchored_vwap"] = (df["cum_pxvol"] - anchor_pxvol) / (
            df["cum_vol"] - anchor_vol
        )
    df.drop(columns=["cum_pxvol", "cum_vol"], inplace=True)
    return df

def compute_indicators(df):
    import ta
    df = df.copy()
    if df.empty:
        return df
    for c in ["Close", "High", "Low"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df.dropna(subset=["Close", "High", "Low"], inplace=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close)
    df["macd_diff"] = macd.macd_diff()
    df["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    df["rsi_2"] = ta.momentum.RSIIndicator(close=close, window=2).rsi()

    boll = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_high"] = boll.bollinger_hband()
    df["bb_low"] = boll.bollinger_lband()
    df["bb_pct_b"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    df["pct_chg_20"] = close.pct_change(20, fill_method=None).fillna(0) * 100
    df["pct_chg_60"] = close.pct_change(60, fill_method=None).fillna(0) * 100

    df.dropna(inplace=True)
    return df

###############################################################################
# 5. ML FEATURE ENGINEERING + TUNING
###############################################################################

def create_features_for_ml(df):
    feats = [
        "rsi_14",
        "macd_diff",
        "adx_14",
        "rsi_2",
        "bb_pct_b",
        "atr_14",
        "ma20",
        "ma50",
        "ma200",
    ]
    df = df.copy()
    df.dropna(subset=feats, inplace=True)
    X = df[feats].values
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).values
    df["target"] = y
    return X, y, df.index, feats

def optuna_xgb_tuner(X, y, n_trials=20):
    if not OPTUNA_AVAILABLE:
        return None, {}
    import optuna
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        size = int(len(X) * 0.8)
        X_train, X_val = X[:size], X[size:]
        y_train, y_val = y[:size], y[size:]
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    final_model = XGBClassifier(**best)
    final_model.fit(X, y)
    return final_model, best

def train_ml_for_symbol(df, do_autotune=False):
    if df.empty:
        df["ml_signal"] = 0
        return df
    X, y, idx, feats = create_features_for_ml(df)
    if len(X) < 200:
        df["ml_signal"] = 0
        return df

    if do_autotune and OPTUNA_AVAILABLE:
        model, best = optuna_xgb_tuner(X, y, n_trials=25)
        if model is None:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss"
            )
            model.fit(X, y)
    else:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss"
        )
        model.fit(X, y)

    preds = model.predict(X)
    df["ml_signal"] = preds
    return df

###############################################################################
# 6. REGIME SWITCHING
###############################################################################

def tune_hmm_states(df, min_states=2, max_states=4):
    if not HMM_AVAILABLE:
        return None, min_states
    if df.empty or len(df) < 250:
        return None, min_states
    from hmmlearn import hmm

    ret = df["Close"].pct_change().fillna(0).values.reshape(-1, 1)
    if "atr_14" in df.columns:
        atr = df["atr_14"].fillna(method="bfill").values.reshape(-1, 1)
    else:
        atr = np.zeros((len(ret), 1))
    obs = np.concatenate([ret, np.log(1 + np.abs(atr))], axis=1)
    keep = ~np.isnan(obs).any(axis=1)
    obs2 = obs[keep]
    if len(obs2) < 200:
        return None, min_states
    best_model = None
    best_states = min_states
    best_bic = np.inf

    for n in range(min_states, max_states + 1):
        m = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=100, random_state=42)
        m.fit(obs2)
        logL = m.score(obs2)
        d = obs2.shape[1]
        num_params = n * (n - 1) + n * (2 * d + d * (d + 1) / 2)
        bic = -2 * logL + num_params * np.log(len(obs2))
        if bic < best_bic:
            best_bic = bic
            best_model = m
            best_states = n
    return best_model, best_states

def detect_regime_hmm(df, do_tuning=False):
    if not HMM_AVAILABLE or df.empty:
        df["regime"] = 1
        return df
    from hmmlearn import hmm

    df = df.copy()
    if do_tuning:
        model, nstates = tune_hmm_states(df)
        if model is None:
            df["regime"] = 1
            return df
    else:
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        ret = df["Close"].pct_change().fillna(0).values.reshape(-1, 1)
        if "atr_14" in df.columns:
            atr = df["atr_14"].fillna(method="bfill").values.reshape(-1, 1)
        else:
            atr = np.zeros((len(ret), 1))
        obs = np.concatenate([ret, np.log(1 + np.abs(atr))], axis=1)
        keep = ~np.isnan(obs).any(axis=1)
        obs2 = obs[keep]
        if len(obs2) < 200:
            df["regime"] = 1
            return df
        model.fit(obs2)

    # predict states
    ret = df["Close"].pct_change().fillna(0).values.reshape(-1, 1)
    if "atr_14" in df.columns:
        atr = df["atr_14"].fillna(method="bfill").values.reshape(-1, 1)
    else:
        atr = np.zeros((len(ret), 1))
    obs = np.concatenate([ret, np.log(1 + np.abs(atr))], axis=1)
    keep_mask = ~np.isnan(obs).any(axis=1)
    obs2 = obs[keep_mask]
    states = model.predict(obs2)
    regime_col = np.full(len(df), np.nan)
    j = 0
    for i in range(len(df)):
        if keep_mask[i]:
            regime_col[i] = states[j]
            j += 1
    ret2 = ret[keep_mask]
    num_s = int(regime_col[~np.isnan(regime_col)].max()) + 1
    means = []
    for s in range(num_s):
        means.append(ret2[states == s].mean())
    best_idx = np.argmax(means)
    final_regime = np.where(regime_col == best_idx, 1, 0)
    df["regime"] = final_regime
    return df

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h_t = out[:, -1, :]
        out = self.fc(h_t)
        return out

def detect_regime_lstm(df):
    if not LSTM_AVAILABLE or df.empty:
        df["regime"] = 1
        return df
    df = df.copy()
    ret = df["Close"].pct_change().fillna(0)
    window = 10
    Xlist, Ylist = [], []
    for i in range(window, len(ret) - 1):
        past = ret.iloc[i - window : i].values.reshape(window, 1)
        future = ret.iloc[i + 1]
        Xlist.append(past)
        Ylist.append(future)
    Xarr = np.array(Xlist, dtype=np.float32)
    Yarr = np.array(Ylist, dtype=np.float32).reshape(-1, 1)
    if len(Xarr) < 50:
        df["regime"] = 1
        return df
    split = int(0.8 * len(Xarr))
    Xtrain, Xtest = Xarr[:split], Xarr[split:]
    Ytrain, Ytest = Yarr[:split], Yarr[split:]

    device = "cpu"
    model = SimpleLSTM(input_dim=1, hidden_dim=8, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    Xtrain_t = torch.from_numpy(Xtrain).to(device)
    Ytrain_t = torch.from_numpy(Ytrain).to(device)

    for epoch in range(30):
        model.train()
        pred = model(Xtrain_t)
        loss = criterion(pred, Ytrain_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    i_t = torch.from_numpy(Xarr).to(device)
    with torch.no_grad():
        p = model(i_t).cpu().numpy().squeeze()
    regime_col = np.zeros(len(ret))
    offset = window
    for i in range(len(p)):
        idx = offset + i + 1
        if idx < len(regime_col):
            regime_col[idx] = 1 if p[i] > 0 else 0
    df["regime"] = regime_col
    return df

###############################################################################
# 7. STYLE/FACTOR TILT
###############################################################################

_factor_cache = {}

def fetch_factor_data(symbol):
    """ Real call to FMP or stub. Provide your real API key. """
    api_key = "<YOUR_FMP_API_KEY>"
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
    try:
        r = requests.get(url)
        data = r.json()
        if isinstance(data, list) and data:
            d = data[0]
            momentum = float(d.get("beta", 1))
            pe = float(d.get("pe", 9999))
            roe = float(d.get("returnOnEquity", 0))
            return {
                "momentum_factor": momentum,
                "value_factor": (1 / pe if pe > 0 else 0),
                "quality_factor": roe,
            }
    except:
        pass
    return {
        "momentum_factor": 1.0,
        "value_factor": 0.0,
        "quality_factor": 0.0,
    }

def get_factor_score(symbol):
    if symbol not in _factor_cache:
        _factor_cache[symbol] = fetch_factor_data(symbol)
    return _factor_cache[symbol]

def style_factor_rank(symbol_list):
    ranks = {}
    if len(symbol_list) == 0:
        return ranks
    for sym in symbol_list:
        f = get_factor_score(sym)
        combined = (
            0.4 * f["momentum_factor"]
            + 0.3 * f["value_factor"]
            + 0.3 * f["quality_factor"]
        )
        ranks[sym] = combined
    mx = max(ranks.values()) if ranks else 1
    mn = min(ranks.values()) if ranks else 0
    for sym in ranks:
        if mx > mn:
            ranks[sym] = (ranks[sym] - mn) / (mx - mn)
        else:
            ranks[sym] = 0.5
    return ranks

###############################################################################
# 8. MULTI-TF LOADING
###############################################################################

def load_symbol_timeframes(symbol, config):
    result = {}
    tfs = config.get("TIMEFRAMES", {})
    regime_method = config.get("REGIME_SWITCH_METHOD", "HMM").upper()
    hmm_tune = (regime_method == "HMM_TUNED")
    do_autotune = config.get("ALGO_SETTINGS", {}).get("ml_autotune", False)

    for tf_key, tf_info in tfs.items():
        yf_period = tf_info["yf_period"]
        yf_interval = tf_info["yf_interval"]
        df_tf = load_timeframe_data(symbol, yf_period, yf_interval)
        if df_tf.empty:
            result[tf_key] = pd.DataFrame()
            continue

        # Debug flatten/rename columns if needed
        if isinstance(df_tf.columns, pd.MultiIndex):
            df_tf.columns = ["_".join(col).strip() for col in df_tf.columns.values]

        # rename any "Close_SPY" -> "Close" if we see that pattern
        possible_close_cols = [c for c in df_tf.columns if c.lower().startswith("close")]
        if "Close" not in df_tf.columns and len(possible_close_cols) == 1:
            df_tf["Close"] = df_tf[possible_close_cols[0]]

        df_tf = compute_anchored_vwap(df_tf)
        df_tf = compute_indicators(df_tf)

        if regime_method == "NONE":
            df_tf["regime"] = 1
        elif regime_method == "LSTM":
            df_tf = detect_regime_lstm(df_tf)
        elif regime_method in ("HMM", "HMM_TUNED"):
            df_tf = detect_regime_hmm(df_tf, do_tuning=hmm_tune)

        df_tf = train_ml_for_symbol(df_tf, do_autotune)
        result[tf_key] = df_tf
    return result

###############################################################################
# 9. CROSS-ASSET
###############################################################################

def compute_cross_asset_signal(ticker_data_map, vix_threshold=25.0):
    score = 1.0
    vix_df = ticker_data_map.get("VIX", {}).get("daily", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        lv = vix_df.iloc[-1]["Close"]
        if lv > vix_threshold:
            score -= 0.3
    return max(0, min(1, score))

###############################################################################
# 10. PORTFOLIO OPTIMIZATION
###############################################################################

def risk_parity_weights(returns_df):
    vol = returns_df.std()
    inv = 1.0 / vol
    w = inv / inv.sum()
    return w

def mean_variance_weights(returns_df):
    mu = returns_df.mean()
    Sigma = returns_df.cov()
    try:
        invS = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(Sigma)
    raw = invS.dot(mu)
    raw[raw < 0] = 0
    if raw.sum() > 0:
        w = raw / raw.sum()
    else:
        w = np.ones(len(raw)) / len(raw)
    w = pd.Series(w, index=returns_df.columns)
    return w

def style_factor_tilt(base_weights):
    syms = base_weights.index
    ranks = style_factor_rank(syms)
    adj = {}
    for sym in syms:
        rank = ranks.get(sym, 0.5)
        adj[sym] = base_weights[sym] * (0.5 + rank)
    w2 = pd.Series(adj)
    s = w2.sum()
    if s > 0:
        w2 = w2 / s
    return w2

def compute_optimized_weights(selected_syms, ticker_data_map, start_dt, end_dt,
                              use_risk_parity=False, tf_key="daily"):
    rets = {}
    for sym in selected_syms:
        df = ticker_data_map.get(sym, {}).get(tf_key, pd.DataFrame())
        sub = df.loc[start_dt:end_dt]
        if not sub.empty and "Close" in sub.columns:
            # fill_method=None => no forward fill
            r = sub["Close"].pct_change(fill_method=None).fillna(0)
            rets[sym] = r
    if not rets:
        return pd.Series(dtype=float)
    df_rets = pd.DataFrame(rets).dropna(how="all")
    if df_rets.empty:
        return pd.Series(dtype=float)
    if use_risk_parity:
        w = risk_parity_weights(df_rets)
    else:
        w = mean_variance_weights(df_rets)
    w = style_factor_tilt(w)
    w = w.clip(upper=0.3)
    s = w.sum()
    if s > 0:
        w = w / s
    return w

###############################################################################
# 11. BACKTEST
###############################################################################

def run_backtest(ticker_data_map, macro_df, config, start_date, end_date):
    algo = config["ALGO_SETTINGS"]
    init_cap = algo["initial_capital"]
    rebal_freq = algo.get("rebal_frequency", 10)
    yoy_cut = algo.get("yoy_inflation_cutoff", 8.0)
    vix_thr = algo.get("vix_threshold", 25.0)
    use_rp = algo.get("use_risk_parity", False)
    vol_stop = algo.get("vol_stop_multiplier", 2.0)

    capital = float(init_cap)
    open_pos = {}
    trades_log = []

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    yoy_infl = 0
    if not macro_df.empty and "CPIAUCSL" in macro_df.columns:
        sub = macro_df.loc[macro_df.index <= start_dt]
        if not sub.empty:
            last = sub.iloc[-1]

            def yoy_cpi():
                c = yoy_change(macro_df["CPIAUCSL"])
                if last.name in c.index:
                    return c.loc[last.name]
                return 0

            yoy_infl = yoy_cpi()

    # Ensure we have SPY daily data, else fallback to a "5y" fetch
    spy_daily = ticker_data_map.get("SPY", {}).get("daily", pd.DataFrame())
    if spy_daily.empty:
        print(
            Fore.YELLOW
            + "SPY daily data not found from config timeframe. Trying fallback 5y fetch..."
            + Style.RESET_ALL
        )
        fallback = load_timeframe_data("SPY", "5y", "1d")
        if fallback.empty:
            print(Fore.RED + "Unable to get SPY daily data. Stopping." + Style.RESET_ALL)
            return pd.DataFrame(), {}
        ticker_data_map["SPY"]["daily"] = fallback
        spy_daily = fallback

    # Flatten/rename columns if needed
    if isinstance(spy_daily.columns, pd.MultiIndex):
        spy_daily.columns = ["_".join(col).strip() for col in spy_daily.columns.values]
    # rename any "Close_SPY" -> "Close" if that pattern
    poss_close_cols = [c for c in spy_daily.columns if c.lower().startswith("close")]
    if "Close" not in spy_daily.columns and len(poss_close_cols) == 1:
        spy_daily["Close"] = spy_daily[poss_close_cols[0]]

    mask = (spy_daily.index >= start_dt) & (spy_daily.index <= end_dt)
    days = list(spy_daily.loc[mask].index)
    if len(days) < 2:
        print(
            Fore.RED
            + f"SPY data in date range {start_dt}~{end_dt} is empty => cannot proceed."
            + Style.RESET_ALL
        )
        return pd.DataFrame(), {}

    eq_curve = []
    for i, day in enumerate(days):
        eq = capital
        for sym, pos in open_pos.items():
            df_d = ticker_data_map.get(sym, {}).get("daily", pd.DataFrame())
            sb = df_d.loc[df_d.index <= day]
            if not sb.empty:
                px = sb.iloc[-1]["Close"]
                eq += pos["shares"] * px
        eq_curve.append((day, eq))

        # check stops
        for sym, pos in list(open_pos.items()):
            df_d = ticker_data_map.get(sym, {}).get("daily", pd.DataFrame())
            sb = df_d.loc[df_d.index <= day]
            if sb.empty:
                continue
            stop = pos["entry_price"] - vol_stop * pos["entry_atr"]
            lastp = sb.iloc[-1]["Close"]
            if lastp < stop:
                sh = pos["shares"]
                outp = lastp
                pnl = (outp - pos["entry_price"]) * sh
                capital += sh * outp
                trades_log.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "STOP_SELL",
                        "shares": round(sh, 4),
                        "price": round(outp, 2),
                        "pnl": round(pnl, 2),
                    }
                )
                del open_pos[sym]

        # rebal?
        do_rebal = (i % rebal_freq == 0) or (i == len(days) - 1)
        if do_rebal:
            # close
            for sym, pos in list(open_pos.items()):
                df_d = ticker_data_map.get(sym, {}).get("daily", pd.DataFrame())
                sb = df_d.loc[df_d.index <= day]
                if sb.empty:
                    continue
                px = sb.iloc[-1]["Close"]
                sh = pos["shares"]
                pnl = (px - pos["entry_price"]) * sh
                capital += sh * px
                trades_log.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "SELL",
                        "shares": round(sh, 4),
                        "price": round(px, 2),
                        "pnl": round(pnl, 2),
                    }
                )
                del open_pos[sym]

            # cross-asset
            ras = compute_cross_asset_signal(ticker_data_map, vix_threshold=vix_thr)
            if yoy_infl > yoy_cut:
                ras -= 0.3
            ras = max(0, min(ras, 1))

            # pick syms where regime=1
            cands = []
            for sym in ticker_data_map.keys():
                df_d = ticker_data_map[sym].get("daily", pd.DataFrame())
                if df_d.empty:
                    continue
                sub = df_d.loc[df_d.index <= day]
                if sub.empty:
                    continue
                lr = sub.iloc[-1]
                if "regime" in lr:
                    if lr["regime"] < 1:
                        continue
                cands.append(sym)

            # build rets window
            st_win = day - timedelta(days=60)
            rets_map = {}
            for sym in cands:
                df_d = ticker_data_map[sym].get("daily", pd.DataFrame())
                sb = df_d.loc[(df_d.index >= st_win) & (df_d.index <= day)]
                if not sb.empty and "Close" in sb.columns:
                    r_ = sb["Close"].pct_change(fill_method=None).fillna(0)
                    rets_map[sym] = r_
            if not rets_map:
                continue
            rets_df = pd.DataFrame(rets_map).dropna(how="all")
            if rets_df.empty:
                continue

            w = compute_optimized_weights(
                list(rets_df.columns), ticker_data_map, st_win, day, use_risk_parity=use_rp
            )
            if w.empty:
                continue
            w = w * ras
            s_ = w.sum()
            if s_ <= 0:
                w = pd.Series([1.0], index=["SPY"])
            else:
                w = w / s_

            # allocate
            for sym, wt in w.items():
                df_d = ticker_data_map[sym].get("daily", pd.DataFrame())
                sb = df_d.loc[df_d.index >= day]
                if sb.empty:
                    continue
                ent_px = sb.iloc[0]["Open"]
                cap_alloc = capital * wt
                sh = 0
                if ent_px > 0:
                    sh = cap_alloc / ent_px
                cost = sh * ent_px
                if cost > capital:
                    sh = capital / ent_px
                    cost = sh * ent_px
                capital -= cost
                last_sub = df_d.loc[df_d.index <= day]
                if not last_sub.empty and "atr_14" in last_sub.columns:
                    at = last_sub.iloc[-1]["atr_14"]
                else:
                    at = 0
                open_pos[sym] = {"shares": sh, "entry_price": ent_px, "entry_atr": at}
                trades_log.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "BUY",
                        "shares": round(sh, 4),
                        "price": round(ent_px, 2),
                        "pnl": 0,
                    }
                )

    eq_df = pd.DataFrame(eq_curve, columns=["Date", "Equity"]).set_index("Date")
    if eq_df.empty:
        return pd.DataFrame(), {}
    final_val = eq_df.iloc[-1]["Equity"]
    tot_pnl = final_val - init_cap
    trades_df = pd.DataFrame(trades_log)
    results = {
        "Initial Capital": init_cap,
        "Final Capital": round(final_val, 2),
        "Total P/L": round(tot_pnl, 2),
        "P/L %": round(100 * tot_pnl / init_cap, 2),
        "Trades": len(trades_df),
    }
    return trades_df, results

###############################################################################
# 12. BUY & HOLD
###############################################################################

def buy_and_hold_spy(spy_daily_df, start_date, end_date, initial_capital=100000):
    """
    Use the same SPY daily DataFrame that the backtest used, with debug prints
    to see if/why it becomes empty or lacks 'Close' data.
    """
    print(Fore.YELLOW + "\n--- DEBUG: Entering buy_and_hold_spy ---" + Style.RESET_ALL)
    if spy_daily_df.empty:
        print("DEBUG: spy_daily_df is EMPTY. Returning None.")
        return None

    print("DEBUG: spy_daily_df shape:", spy_daily_df.shape)
    print(
        "DEBUG: spy_daily_df date range => min:",
        spy_daily_df.index.min(),
        " / max:",
        spy_daily_df.index.max(),
    )
    if "Close" not in spy_daily_df.columns:
        print("DEBUG: 'Close' not in spy_daily_df columns =>", spy_daily_df.columns)
        return None

    sub = spy_daily_df.loc[start_date:end_date]
    print("DEBUG: sub shape after slicing =>", sub.shape)
    if sub.empty:
        print("DEBUG: sub is empty after slicing. Start/End =>", start_date, end_date)
        return None
    if "Close" not in sub.columns:
        print("DEBUG: 'Close' not found in sub columns =>", sub.columns)
        return None

    entry = sub["Close"].iloc[0]
    finalp = sub["Close"].iloc[-1]
    if pd.isna(entry) or pd.isna(finalp):
        print("DEBUG: The first/last close is NaN => entry:", entry, " finalp:", finalp)
        return None

    shares = initial_capital / entry
    final_val = shares * finalp
    total_pnl = final_val - initial_capital

    print("DEBUG: B&H => entry px:", entry, " final px:", finalp)
    print("DEBUG: shares =>", shares, " final_val =>", final_val)

    return {
        "Initial Capital": initial_capital,
        "Final Capital": round(final_val, 2),
        "Total P/L": round(total_pnl, 2),
        "P/L %": round(100 * total_pnl / initial_capital, 2),
    }

###############################################################################
# 13. PARAMETER TUNING MENU
###############################################################################

def tune_hmm_states_main():
    if not HMM_AVAILABLE:
        print("hmmlearn not available.")
        return
    # example => tune on SPY daily
    spy = load_timeframe_data("SPY","5y","1d")
    if spy.empty:
        print("No SPY => skip hmm states.")
        return
    spy = compute_indicators(spy)
    best_model,best_states = tune_hmm_states(spy,2,4)
    if best_model is None:
        print("No best hmm model found.")
    else:
        print(f"Best #states => {best_states}")

def tune_xgb_main():
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed => skip xgb tune.")
        return
    spy = load_timeframe_data("SPY","3y","1d")
    spy = compute_indicators(spy)
    if spy.empty:
        print("No SPY => cannot tune XGB.")
        return
    X,y,_,_ = create_features_for_ml(spy)
    if len(X)<200:
        print("Not enough SPY data => skip xgb tune.")
        return
    model,best = optuna_xgb_tuner(X,y,n_trials=25)
    print("Best xgb =>", best)

def run_parameter_tuning(config):
    print("\n=== Parameter Tuning Menu ===")
    print("1) Tune HMM states for SPY")
    print("2) Tune XGBoost hyperparams (Optuna)")
    print("3) Return")

    c = input("Enter choice (1-3): ").strip()
    if c=="1":
        tune_hmm_states_main()
    elif c=="2":
        tune_xgb_main()
    elif c=="3":
        return
    else:
        print("Invalid choice")

###############################################################################
# 14. MAIN MENU
###############################################################################

def display_menu():
    print(Fore.CYAN + "\n=== Multi-Asset, Rolling Rebalance Algo ===" + Style.RESET_ALL)
    print("1. Run Backtest")
    print("2. Run Parameter Tuning")
    print("3. Exit")

def run_full_backtest(config):
    periods = {"1":"1m","2":"3m","3":"1y","4":"2y","5":"5y"}
    print("\nSelect backtesting period:")
    for k,v in periods.items():
        print(f"{k}) {v}")
    c = input("Enter choice (1-5): ").strip()
    if c not in periods:
        print("Invalid choice.")
        return
    selection = periods[c]
    now = datetime.today()
    if selection=="1m":
        start_dt = now - relativedelta(months=1)
    elif selection=="3m":
        start_dt = now - relativedelta(months=3)
    elif selection=="1y":
        start_dt = now - relativedelta(years=1)
    elif selection=="2y":
        start_dt = now - relativedelta(years=2)
    else:
        start_dt = now - relativedelta(years=5)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = now.strftime("%Y-%m-%d")

    # load macro
    macro_df = pd.DataFrame()
    if FRED_AVAILABLE and "FRED_API_KEY" in config.get("API_KEYS",{}):
        base_freds = config.get("FRED_SERIES",["CPIAUCSL","UNRATE"])
        add = ["T10Y2Y","UMCSENT"]
        series_list = list(set(base_freds + add))
        macro_df = load_macro_data(series_list, config["API_KEYS"]["FRED_API_KEY"], start_str, end_str)

    tickers = config.get("TICKER_LIST", ["SPY","QQQ","IWM","VIX"])
    ticker_data_map = {}
    for sym in tickers:
        print(f"Loading data for {sym}...")
        ticker_data_map[sym] = load_symbol_timeframes(sym, config)

    trades, results = run_backtest(ticker_data_map, macro_df, config, start_str, end_str)

    if not trades.empty and results:
        print(Fore.MAGENTA + "\n=== FULL TRADE HISTORY ===" + Style.RESET_ALL)
        print(trades.to_string(index=False))
        outn = f"trades_log_{selection}.csv"
        trades.to_csv(outn, index=False)
        print(Fore.GREEN + f"Trade history => {outn}" + Style.RESET_ALL)

    print(Fore.GREEN + f"\n=== {selection} Rolling Rebalance Results ===" + Style.RESET_ALL)
    if not results or "P/L %" not in results:
        if results:
            for k,v in results.items():
                print(f"{k}: {v}")
        print(Fore.RED + "No valid backtest results => skipping buy & hold comparison." + Style.RESET_ALL)
        return
    else:
        for k,v in results.items():
            print(f"{k}: {v}")

    # Attempt the same final SPY daily data for B&H
    # The user might not have "SPY" in TICKER_LIST, but let's assume it is there
    spy_final_df = ticker_data_map.get("SPY",{}).get("daily", pd.DataFrame())
    bh = buy_and_hold_spy(spy_final_df, start_str, end_str, config["ALGO_SETTINGS"]["initial_capital"])

    print(Fore.CYAN + f"\n=== {selection} Buy & Hold SPY ===" + Style.RESET_ALL)
    if bh is None:
        print("No SPY data => could not do buy & hold. Possibly empty or missing columns.")
        return
    for kk,vv in bh.items():
        print(f"{kk}: {vv}")

    if results["P/L %"]>bh["P/L %"]:
        print(Fore.GREEN + "Strategy beat SPY." + Style.RESET_ALL)
    else:
        print(Fore.RED + "Strategy did NOT beat SPY." + Style.RESET_ALL)

def main():
    ensure_data_dir()
    config = load_config(CONFIG_FILE)
    logging.info("Loaded config.")
    while True:
        display_menu()
        c = input("Enter choice (1-3): ").strip()
        if c=="1":
            run_full_backtest(config)
        elif c=="2":
            run_parameter_tuning(config)
        elif c=="3":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__=="__main__":
    main()
