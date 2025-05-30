#!/usr/bin/env python3
"""
SPXL / SPXS – Intraday-Reversal Model (XGBoost)
==============================================

• 5-minute bars via yfinance
• Detects a *reversal event* = strong trend during the last 60 min
  that flips by ≥ 0.35 % (underlying) inside the next 30 min.
• Trades opposite to the current trend:
      up-trend → expect down-reversal → long SPXS
      down-trend → expect up-reversal → long SPXL
• Risk: 1 % of equity, stop = 0.35 % or 1×ATR, TP = 0.60 %
• Works as:  (i) trainer / back-tester,  (ii) live decision engine
"""
import argparse, datetime as dt, os, pickle, sys
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression      # 🔸 NEW
import optuna

# ---------------------- Hyper-parameters --------------------------------------
INTERVAL        = "5m"
LOOKBACK        = 12          # 🔸 past bars (≈ 60 min) to define “trend”
FWD_WINDOW      = 6           # 🔸 future bars (≈ 30 min) to test for reversal
TREND_TH        = 0.0020      # 🔸 trend strength ≥ 0.20 % (underlying)
REV_TH          = 0.0035      # 🔸 opposite move ≥ 0.35 %
TP_PCT          = 0.0060      # take-profit 0.60 %
STOP_PCT        = 0.0035      # hard stop 0.35 %
ATR_WIN         = 14
# PROB_TH         = 0.60  # Deprecated: Thresholds are now per-regime and stored in the model bundle.
START_DATE      = (dt.date.today() - dt.timedelta(days=50)).isoformat()
END_DATE        = dt.date.today().isoformat()
ETF_LONG, ETF_SH = "SPXL", "SPXS"
CAPITAL, RISK_PCT, SLIP_BP    = 100_000.0, 0.01, 5
MODEL_PATH      = "xgb_reversal.model"
REG_PCTL        = 0.75   # 75-th percentile of atr_pct defines “high-vol” regime

# ---------------------- Data utilities ----------------------------------------
def download(tkr: str) -> pd.DataFrame:
    df = yf.download(tkr, start=START_DATE, end=END_DATE,
                     interval=INTERVAL, progress=False)
    if df.empty: raise ValueError(f"No data for {tkr}")
    df = df.tz_localize(None)
    # If columns are MultiIndex (e.g., ('SPXL', 'Open')), flatten them for single ticker
    if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels[0]) == 1:
        df.columns = df.columns.droplevel(0) # Drops the top level (ticker name)
    return df

def rsi(series: pd.Series, n: int = 2) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = -delta.clip(upper=0).rolling(n).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

# ---------------------- Feature engineering -----------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    new_cols = []
    for col in d.columns:
        if isinstance(col, tuple):
            # If the second part of tuple is empty or if the first part is a common name
            if len(col) > 1 and (col[1] == '' or col[0] in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']):
                new_cols.append(col[0])
            else: # Otherwise, join all parts of the tuple
                new_cols.append('_'.join(str(c).strip() for c in col))
        else: # If not a tuple, it's already a string
            new_cols.append(str(col).strip())
    d.columns = new_cols

    if d.empty: # Check after potential column name changes if d could become empty (unlikely here)
        expected_cols = ["log_ret", "cum_ret60", "trend_sign", "vwap", "dist_vwap",
                         "bb_z", "rsi2", "atr", "atr_pct", "spread_pct"] + \
                        [f"log_ret_l{l}" for l in (1,2,3,4,5)] + \
                        ["Close", "High", "Low", "Volume", "Open"]
        empty_df = pd.DataFrame(columns=expected_cols, index=d.index)
        for col_name_expected in empty_df.columns:
             if col_name_expected in ["Volume"]:
                 empty_df[col_name_expected] = pd.Series(dtype='int64')
             else:
                 empty_df[col_name_expected] = pd.Series(dtype='float64')
        return empty_df

    # Ensure index is unique and sorted
    if not d.index.is_unique:
        print("Warning: DataFrame index is not unique. Grouping by index and taking first.")
        d = d.groupby(d.index).first()
    if not d.index.is_monotonic_increasing:
        print("Warning: DataFrame index is not sorted. Sorting index.")
        d = d.sort_index()

    # NEW: Ensure core OHLCV columns are Series
    for col_name in ["Open", "High", "Low", "Close", "Volume"]:
        if col_name in d.columns:
            if isinstance(d[col_name], pd.DataFrame) and d[col_name].shape[1] == 1:
                d[col_name] = d[col_name].iloc[:, 0] # Convert single-column DataFrame to Series
            elif not isinstance(d[col_name], pd.Series):
                print(f"Warning: Column {col_name} is not a Series or single-column DataFrame. Type: {type(d[col_name])}")
        else:
            # This case might indicate a problem with data download or column name changes
            print(f"Warning: Expected column {col_name} not found in DataFrame.")


    d["log_ret"]   = np.log(d["Close"] / d["Close"].shift(1))
    d["cum_ret60"] = d["Close"] / d["Close"].shift(LOOKBACK) - 1  # 🔸past hour
    d["trend_sign"] = np.sign(d["cum_ret60"])

    # VWAP calculation using numpy arrays for robustness
    typical_price_values = d[["High","Low","Close"]].mean(axis=1).values
    volume_values = d["Volume"].values # Should be a Series now due to above loop

    if typical_price_values.ndim > 1: typical_price_values = typical_price_values.squeeze()
    if volume_values.ndim > 1: volume_values = volume_values.squeeze()
    # VWAP calculation using numpy arrays for robustness (debug prints removed)
    typical_price_values = d[["High","Low","Close"]].mean(axis=1).values
    volume_values = d["Volume"].values

    if typical_price_values.ndim > 1: typical_price_values = typical_price_values.squeeze()
    if volume_values.ndim > 1: volume_values = volume_values.squeeze()

    vol_price_values = volume_values * typical_price_values
    cum_vol_price_values = vol_price_values.cumsum()
    cum_volume_values    = volume_values.cumsum()
    cum_vol_price_values = np.nan_to_num(cum_vol_price_values)
    cum_volume_values    = np.nan_to_num(cum_volume_values)
    vwap_values = cum_vol_price_values / (cum_volume_values + 1e-12)
    vwap_series = pd.Series(vwap_values, index=d.index, name="vwap_calc")
    d["vwap"] = vwap_series.ffill().fillna(0)

    # Force numpy array operations for other features
    close_val = d["Close"].values
    if close_val.ndim > 1: close_val = close_val.squeeze()
    vwap_val = d["vwap"].values
    if vwap_val.ndim > 1: vwap_val = vwap_val.squeeze()

    dist_vwap_values = (close_val - vwap_val) / (vwap_val + 1e-12)
    d["dist_vwap"] = pd.Series(dist_vwap_values, index=d.index)

    ma20_val = d["Close"].rolling(20).mean().values
    if ma20_val.ndim > 1: ma20_val = ma20_val.squeeze()
    std20_val = d["Close"].rolling(20).std().fillna(0).values # fillna(0) for std if it's NaN (e.g. single point)
    if std20_val.ndim > 1: std20_val = std20_val.squeeze()

    bb_z_values = (close_val - ma20_val) / (std20_val + 1e-12) # Epsilon for std20
    d["bb_z"] = pd.Series(bb_z_values, index=d.index)

    d["rsi2"]      = rsi(d["Close"], 2)

    high_val = d["High"].values
    if high_val.ndim > 1: high_val = high_val.squeeze()
    low_val = d["Low"].values
    if low_val.ndim > 1: low_val = low_val.squeeze()

    atr_series = (pd.Series(high_val, index=d.index) - pd.Series(low_val, index=d.index)).rolling(ATR_WIN).mean()
    d["atr"] = atr_series.ffill().fillna(0)

    atr_val = d["atr"].values
    if atr_val.ndim > 1: atr_val = atr_val.squeeze()

    d["atr_pct"] = pd.Series(atr_val / (close_val + 1e-12), index=d.index) # Use close_val
    d["spread_pct"] = pd.Series((high_val - low_val) / (close_val + 1e-12), index=d.index) # Use close_val

    log_ret_series = d["log_ret"] # d["log_ret"] should be a Series
    if isinstance(log_ret_series, pd.DataFrame) and log_ret_series.shape[1] == 1:
        log_ret_series = log_ret_series.iloc[:,0] # Ensure it's a series for .shift()
    for l in (1, 2, 3, 4, 5):
        d[f"log_ret_l{l}"] = log_ret_series.shift(l)
    d.dropna(inplace=True)
    return d

def triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """
    +1 if TP hit first inside FWD_WINDOW
    -1 if SL hit first
     0 otherwise  (no trade)
    """
    close = df["Close"].values
    targets = np.zeros(len(df))
    for i in range(len(df) - FWD_WINDOW):
        tp_lvl = close[i] * (1 + TP_PCT)
        sl_lvl = close[i] * (1 - SL_PCT)
        window = close[i + 1 : i + 1 + FWD_WINDOW]
        try:
            tp_hit = np.where(window >= tp_lvl)[0][0]
        except IndexError:
            tp_hit = FWD_WINDOW + 1
        try:
            sl_hit = np.where(window <= sl_lvl)[0][0]
        except IndexError:
            sl_hit = FWD_WINDOW + 1
        if tp_hit < sl_hit:
            targets[i] = 1
        elif sl_hit < tp_hit:
            targets[i] = -1
    return pd.Series(targets, index=df.index)

def dataset() -> pd.DataFrame:
    base   = download(ETF_LONG)
    feats  = engineer(base)
    feats["target"] = triple_barrier_labels(feats)
    feats = feats[feats["target"] != 0]          # only decisive bars
    return feats

# ---------------------- Label construction ------------------------------------
def label_reversals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy() # df is the output of engineer()

    # Ensure inputs are 1D numpy arrays for robustness
    close_values = d["Close"].values
    if close_values.ndim > 1: close_values = close_values.squeeze()

    cum_ret60_values = d["cum_ret60"].values
    if cum_ret60_values.ndim > 1: cum_ret60_values = cum_ret60_values.squeeze()

    # Calculate fut_ret using numpy arrays where possible, ensure Series for shift
    # Using .loc to avoid SettingWithCopyWarning if d["Close"] was a slice
    fut_ret_series = d.loc[:, "Close"].shift(-FWD_WINDOW) / (d.loc[:, "Close"] + 1e-12) - 1
    fut_ret_values = fut_ret_series.fillna(0).values # fillna(0) before .values
    if fut_ret_values.ndim > 1: fut_ret_values = fut_ret_values.squeeze()

    cond_strong_values = abs(cum_ret60_values) >= TREND_TH

    sign_mult = np.sign(cum_ret60_values) * np.sign(fut_ret_values)
    cond_flip_values = (sign_mult == -1) & (abs(fut_ret_values) >= REV_TH)

    target_values = (cond_strong_values & cond_flip_values).astype(int)
    d["target"] = pd.Series(target_values, index=d.index)

    d.dropna(inplace=True)
    return d

def dataset() -> pd.DataFrame:
    base = download(ETF_LONG)          # we learn on SPXL only – cheaper
    return label_reversals(engineer(base))

# ------------------- regime split helper --------------------------------------
def split_by_regime(feats: pd.DataFrame):
    thresh = feats["atr_pct"].quantile(REG_PCTL)
    high   = feats[feats["atr_pct"] >= thresh]
    low    = feats[feats["atr_pct"] <  thresh]
    return {"high": high, "low": low}, thresh

def choose_regime(row, vol_thresh):
    atr_pct_val = row["atr_pct"]
    if isinstance(atr_pct_val, pd.Series): # Should not happen with row-wise apply if columns are simple
        atr_pct_val = atr_pct_val.item() if len(atr_pct_val) == 1 else np.nan # Or handle error

    if pd.isna(atr_pct_val): # Handle case where atr_pct is NaN
        return "low"
    if vol_thresh is None or pd.isna(vol_thresh): # Handle missing vol_thresh
        print("Warning: vol_thresh is None or NaN in choose_regime. Defaulting to 'low'.")
        return "low"
    return "high" if atr_pct_val >= vol_thresh else "low"

# Placed before train_regime, e.g., after dataset() functions or where other ML helpers are.
def train_single(X_tr, y_tr, X_val, y_val):
    def _optuna(trial):
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda":  trial.suggest_float("lambda", 0, 5),
            "alpha":   trial.suggest_float("alpha", 0, 5),
            "n_estimators": 600, "objective": "binary:logistic",
            "eval_metric": "logloss", "n_jobs": -1, "random_state": 42} # Added random_state for reproducibility
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr) # No eval_set or early stopping for simplicity as per snippet
        return -f1_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction="minimize") # Removed show_progress_bar
    study.optimize(_optuna, n_trials=3, timeout=60) # Faster testing parameters

    best = study.best_params
    best.update({"n_estimators": 600, "objective": "binary:logistic",
                 "eval_metric": "logloss", "n_jobs": -1, "random_state": 42}) # Added random_state
    mdl = xgb.XGBClassifier(**best).fit(X_tr, y_tr)

    raw_val_prob = mdl.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val_prob, y_val)
    calib_prob_val = iso.transform(raw_val_prob)

    if y_val.mean() == 0 or y_val.mean() == 1:
        thr = 0.5 if y_val.mean() == 0 else 0.001
        print(f"Warning: y_val mean is {y_val.mean()} for threshold calculation. Using fallback threshold: {thr}")
    elif len(np.unique(calib_prob_val)) == 1:
        thr = np.unique(calib_prob_val)[0]
        print(f"Warning: All calibrated probabilities in validation are identical ({thr}). Using this as threshold.")
    else:
        thr = np.quantile(calib_prob_val, 1 - y_val.mean())

    return mdl, iso, thr

# ---------------------- Model training ----------------------------------------
# def _optuna_objective_original_backup(trial, X_tr, y_tr, X_val, y_val):
#     params = {
#         "eta":          trial.suggest_float("eta", 0.01, 0.2, log=True),
#         "max_depth":    trial.suggest_int("max_depth", 3, 9),
#         "subsample":    trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "lambda":       trial.suggest_float("lambda", 0.0, 5.0),
#         "alpha":        trial.suggest_float("alpha", 0.0, 5.0),
#         "n_estimators": 600,
#         "objective":    "binary:logistic",
#         "eval_metric":  "logloss",
#         "n_jobs":       -1,
#     }
#     model = xgb.XGBClassifier(**params)
#     model.fit(X_tr, y_tr)
#     preds = model.predict(X_val)
#     return -f1_score(y_val, preds)               # maximise F1

# def _train_original_backup(df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, IsotonicRegression, float, List[str]]:
#     feats = [c for c in df.columns if c not in ("target",)]
#     X, y  = df[feats], (df["target"] == 1).astype(int)   # 1 = TP side
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, shuffle=False)
#
#     study = optuna.create_study(direction="minimize", show_progress_bar=False)
#     study.optimize(lambda t: optuna_objective(t, X_tr, y_tr, X_val, y_val), # This would now call the backup
#                    n_trials=60, timeout=900)
#
#     best_params = study.best_params
#     best_params.update({"n_estimators": 600, "objective": "binary:logistic",
#                         "eval_metric": "logloss", "n_jobs": -1})
#     model = xgb.XGBClassifier(**best_params)
#     model.fit(X_tr, y_tr)
#
#     # ----- probability calibration -------------------------------------------
#     raw_val_prob = model.predict_proba(X_val)[:, 1]
#     iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val_prob, y_val)
#     calib_prob   = iso.transform(raw_val_prob)
#
#     # choose threshold that keeps historical win-rate ≥ 55 %
#     thresh = np.quantile(calib_prob, 1 - y_val.mean())
#
#     # report
#     preds = (calib_prob > thresh).astype(int)
#     acc   = accuracy_score(y_val, preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary")
#     print(f"ACC {acc:.2f}  PREC {prec:.2f}  REC {rec:.2f}  F1 {f1:.2f}")
#
#     # persist three objects in one pickle
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump({"xgb": model, "iso": iso, "thresh": thresh, "feats": feats}, f)
#
#     return model, iso, thresh, feats

# Placed after train_single, likely replacing the old 'train' function.
def train_regime(feats_df: pd.DataFrame) -> dict:
    regimes, vol_thresh = split_by_regime(feats_df)

    feature_cols = [c for c in feats_df.columns if c != "target"]

    bundle = {"vol_thresh": vol_thresh, "feats": feature_cols,
              "models": defaultdict(dict)}

    print(f"Feature columns for training: {feature_cols}")
    print(f"Volatility threshold (atr_pct for regime split): {vol_thresh:.6f}")

    for reg_name, df_sub in regimes.items():
        if df_sub.empty or len(df_sub) < 50:
            print(f"[{reg_name}] Skipping training: insufficient data ({len(df_sub)} samples). Min 50 required.")
            bundle["models"][reg_name] = {"xgb": None, "iso": None, "thr": np.nan, "error": "Insufficient data"}
            continue

        X, y = df_sub[feature_cols], (df_sub["target"] == 1).astype(int)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        if len(X_tr) < 20 or len(X_val) < 10:
            print(f"[{reg_name}] Skipping training: insufficient data after split (train {len(X_tr)}, val {len(X_val)}). Min 20/10 required.")
            bundle["models"][reg_name] = {"xgb": None, "iso": None, "thr": np.nan, "error": "Insufficient data after split"}
            continue

        if len(np.unique(y_val)) < 2 and len(y_val) > 0 :
            print(f"[{reg_name}] Warning: y_val for regime '{reg_name}' has only one class. Calibration might be suboptimal.")

        print(f"Training model for regime: {reg_name} with {len(X_tr)} train samples, {len(X_val)} validation samples.")
        try:
            mdl, iso, thr = train_single(X_tr, y_tr, X_val, y_val)
            bundle["models"][reg_name] = {"xgb": mdl, "iso": iso, "thr": thr}
            print(f"[{reg_name}] Trained. Threshold={thr:.4f}. Samples used (total for regime): {len(y)}")
        except Exception as e:
            print(f"[{reg_name}] Error during training: {e}")
            bundle["models"][reg_name] = {"xgb": None, "iso": None, "thr": np.nan, "error": str(e)}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"  Volatility Threshold (atr_pct): {bundle.get('vol_thresh', 'N/A')}")
    print(f"  Feature columns: {bundle.get('feats', 'N/A')}")
    for reg_name, models_dict in bundle.get("models", {}).items():
        error_msg = models_dict.get("error")
        if error_msg:
            print(f"  Regime '{reg_name}': Training failed or skipped. Error: {error_msg}")
        else:
            # Ensure thr is formatted only if it's a number (including numpy float types)
            thr_val = models_dict.get('thr')
            thr_display = f"{thr_val:.4f}" if isinstance(thr_val, (int, float, np.floating)) and not np.isnan(thr_val) else "N/A"
            print(f"  Regime '{reg_name}': Has model components: xgb={models_dict.get('xgb') is not None}, iso={models_dict.get('iso') is not None}, threshold={thr_display}")
    return bundle

# ---------------------- Back-test engine --------------------------------------
class Backtest:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.vol_thresh = bundle.get("vol_thresh") # Use .get for safety
        self.feature_cols = bundle.get("feats")   # Use .get for safety
        self.regime_models = bundle.get("models", defaultdict(dict)) # Default to empty defaultdict
        self.cap, self.equity = CAPITAL, [] # CAPITAL is a global constant

        if not self.regime_models: # Check if models dict is empty or None
            print("Warning: No models found in the bundle for Backtest.")
        if self.vol_thresh is None or (isinstance(self.vol_thresh, float) and np.isnan(self.vol_thresh)):
            print(f"Warning: Invalid volatility threshold ({self.vol_thresh}) in bundle for Backtest.")
        if not self.feature_cols:
            print("Warning: No feature columns found in the bundle for Backtest.")

    @staticmethod
    def _trade_eligible(row: pd.Series) -> bool:
        bb_z_val = row.get("bb_z")
        spread_pct_val = row.get("spread_pct")

        if isinstance(bb_z_val, pd.Series):
            bb_z_val = bb_z_val.item() if len(bb_z_val) == 1 and not bb_z_val.empty else np.nan
        if isinstance(spread_pct_val, pd.Series):
            spread_pct_val = spread_pct_val.item() if len(spread_pct_val) == 1 and not spread_pct_val.empty else np.nan

        if pd.isna(bb_z_val) or pd.isna(spread_pct_val):
            return False

        boll_cond = abs(bb_z_val) >= 1.5
        sprd_cond = spread_pct_val >= 0.0015 # Ensure spread_pct_val is float after potential .item()
        return boll_cond and sprd_cond

    def run(self, df: pd.DataFrame) -> None:
        d = df.copy() # df should have 'atr_pct', 'bb_z', 'spread_pct', 'cum_ret60' from engineer()

        if self.vol_thresh is None or np.isnan(self.vol_thresh) or not self.feature_cols or not self.regime_models:
            print("Backtest run aborted: Volatility threshold, features, or models invalid/missing in bundle.")
            ret = (self.cap / CAPITAL - 1) * 100; trades = 0; wr = 0
            print(f"Return {ret:.2f}% | Win-rate {wr:.2%} | Trades {trades}")
            return

        # 1. Determine regime for all rows
        # choose_regime function should be accessible (it's global)
        # The lambda now more robustly handles row["atr_pct"]
        d["regime"] = d.apply(
            lambda row: choose_regime(row, self.vol_thresh), axis=1
        )
        # Old lambda: lambda row: choose_regime(row, self.vol_thresh) if pd.notna(row["atr_pct"]) else "low"
        # choose_regime now handles NaN atr_pct internally.

        # 2. Vectorized calculation for eligibility
        d["eligible"] = d.apply(Backtest._trade_eligible, axis=1)

        d["cal_prob"] = np.nan # Initialize column

        # 3. Calculate calibrated probability per regime (vectorized per regime)
        for regime_name_iter in d["regime"].unique():
            if regime_name_iter is None: continue

            model_components = self.regime_models.get(regime_name_iter)
            if not model_components or not model_components.get("xgb") or not model_components.get("iso"):
                # print(f"No valid model for regime {regime_name_iter}, trades will not be generated for it.")
                continue

            xgb_model = model_components["xgb"]
            iso_model = model_components["iso"]

            regime_mask = (d["regime"] == regime_name_iter)
            if regime_mask.sum() == 0: continue

            features_for_regime = d.loc[regime_mask, self.feature_cols]
            if features_for_regime.empty: continue

            # Ensure features are in the correct order and no NaNs for XGBoost
            features_for_regime = features_for_regime[self.feature_cols]
            if features_for_regime.isnull().values.any():
                print(f"Warning: NaN values found in features for regime {regime_name_iter}. Filling with 0 for prediction.")
                features_for_regime = features_for_regime.fillna(0)


            raw_probs_regime = xgb_model.predict_proba(features_for_regime.values)[:, 1]
            cal_probs_regime = iso_model.transform(raw_probs_regime)
            d.loc[regime_mask, "cal_prob"] = cal_probs_regime

        # 4. Signal generation (vectorized per regime)
        d["signal"] = 0 # Default no signal
        for regime_name_iter in d["regime"].unique():
            if regime_name_iter is None: continue
            model_components = self.regime_models.get(regime_name_iter)
            if not model_components or model_components.get("thr") is None or np.isnan(model_components.get("thr")):
                # print(f"No valid threshold for regime {regime_name_iter}, signals will not be generated.")
                continue

            regime_thresh = model_components["thr"]
            regime_mask = (d["regime"] == regime_name_iter)

            # Apply signal only to eligible rows within the regime that meet the threshold
            d.loc[regime_mask & d["eligible"], "signal"] = \
               (d.loc[regime_mask & d["eligible"], "cal_prob"] >= regime_thresh).astype(int)

        # 5. Trend determination
        # The engineer function creates "trend_sign" column.
        d["trend"] = d["trend_sign"]

        # 6. Existing trade execution logic (uses d["signal"] and d["trend"])
        position, entry_px, qty, side, bars_in_trade = 0, 0, 0, 0, 0
        self.equity = [] # Reset equity for each run

        for i, row in d.iterrows():
            px = row["Open"]
            if position == 0 and row["signal"] == 1:
                side = -1 if row["trend"] == 1 else 1
                slip = px * (1 + SLIP_BP / 1e4)
                entry_px = slip
                qty = (self.cap * RISK_PCT) / entry_px
                position, bars_in_trade = side, 0
            elif position != 0:
                bars_in_trade += 1
                cur_px = px
                change = (cur_px - entry_px) * position / entry_px

                stop_level_pct = STOP_PCT # Default stop
                if row["Close"] != 0 and pd.notna(row["atr"]) and row["Close"] != 0 : # Check for non-zero Close and non-NaN atr
                    stop_level_pct = max(STOP_PCT, row["atr"] / row["Close"])

                if change >= TP_PCT or change <= -stop_level_pct or bars_in_trade >= FWD_WINDOW:
                    exit_px = px * (1 - SLIP_BP / 1e4)
                    pnl = qty * (exit_px - entry_px) * position
                    self.cap += pnl
                    position = 0
            self.equity.append(self.cap)

        # Final metrics
        ret = (self.cap / CAPITAL - 1) * 100
        wins = (np.diff(self.equity) > 0).sum() if len(self.equity) > 1 else 0
        losses = (np.diff(self.equity) < 0).sum() if len(self.equity) > 1 else 0
        trades = wins + losses
        wr = wins / trades if trades else 0
        print(f"Return {ret:.2f}% | Win-rate {wr:.2%} | Trades {trades}")

# ---------------------- Live decision helper ----------------------------------
# Assumes engineer, choose_regime, ETF_LONG, INTERVAL are defined globally
def get_live_prediction_data(bundle: dict):
    vol_thresh = bundle.get("vol_thresh")
    feature_cols = bundle.get("feats")
    regime_models = bundle.get("models")

    if vol_thresh is None or np.isnan(vol_thresh) or feature_cols is None or regime_models is None:
        print(f"{dt.datetime.now()} Live: Bundle is missing essential components (vol_thresh, feats, or models).")
        return None

    try:
        df_raw = yf.download(ETF_LONG, period="3d", interval=INTERVAL, progress=False, timeout=10)
        if df_raw.empty:
            print(f"{dt.datetime.now()} Live: No data downloaded from yfinance.")
            return None
        if df_raw.index.tz is not None:
            df_raw = df_raw.tz_localize(None)
    except Exception as e:
        print(f"{dt.datetime.now()} Live: Error downloading data: {e}")
        return None

    df_eng = engineer(df_raw)

    if df_eng.empty:
        print(f"{dt.datetime.now()} Live: Data became empty after feature engineering.")
        return None

    latest_row_df = df_eng.iloc[-1:].copy()
    if latest_row_df.empty:
        print(f"{dt.datetime.now()} Live: No latest row available after engineering.")
        return None

    latest_row_series = latest_row_df.iloc[0]

    if pd.isna(latest_row_series.get("atr_pct")):
        print(f"{dt.datetime.now()} Live: Cannot determine regime, atr_pct is NaN for latest row. Timestamp: {latest_row_series.name}")
        return None
    current_regime = choose_regime(latest_row_series, vol_thresh)

    model_components = regime_models.get(current_regime)
    if not model_components or not model_components.get("xgb") or \
       not model_components.get("iso") or model_components.get("thr") is None or \
       np.isnan(model_components.get("thr")):
        print(f"{dt.datetime.now()} Live: Model for regime '{current_regime}' is missing, invalid, or has no threshold. Timestamp: {latest_row_series.name}")
        return None

    xgb_model = model_components["xgb"]
    iso_model = model_components["iso"]
    regime_thr = model_components["thr"]

    missing_feats = [col for col in feature_cols if col not in latest_row_df.columns]
    if missing_feats:
        print(f"{dt.datetime.now()} Live: Missing feature columns in latest_row_df: {missing_feats}. Timestamp: {latest_row_series.name}")
        return None

    latest_features_values = latest_row_df[feature_cols].values
    if np.isnan(latest_features_values).any():
        nan_features = latest_row_df[feature_cols].columns[np.isnan(latest_features_values).any(axis=0)].tolist()
        print(f"{dt.datetime.now()} Live: NaN values found in features for XGBoost: {nan_features}. Timestamp: {latest_row_series.name}")
        return None

    raw_prob_latest = xgb_model.predict_proba(latest_features_values)[:, 1]
    cal_prob_latest = iso_model.transform(raw_prob_latest)[0]

    trend_latest = int(latest_row_series["trend_sign"])
    ts_latest = latest_row_df.index[-1]

    return {
        "cal_prob": cal_prob_latest,
        "regime_thr": regime_thr,
        "trend": trend_latest,
        "timestamp": ts_latest,
        "latest_row_data": latest_row_series
    }

# def _latest_features_original_backup(model, feats):
#     df = yf.download(ETF_LONG, period="3d", interval=INTERVAL, progress=False)
#     df = engineer(df).tail(LOOKBACK+1)          # ensure sufficient history
#     df = label_reversals(df)                    # will create cum_ret etc.
#     if df.empty: return None
#     x = df.iloc[-1:][feats]
#     prob = model.predict_proba(x)[:, 1][0]
#     trend = int(df.iloc[-1]["trend_sign"])
#     return prob, trend, df.index[-1]

# ---------------------- CLI ---------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--bt",    action="store_true")
    p.add_argument("--live",  action="store_true")
    args = p.parse_args()

    if args.train or not os.path.exists(MODEL_PATH):
        print("Training new regime-based models...")
        bundle = train_regime(dataset())
        print("Training complete. Model bundle generated.")
    else:
        print(f"Loading existing model bundle from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        print("Model bundle loaded.")

    print("\nLoaded bundle structure summary:")
    vol_thresh_display = f"{bundle.get('vol_thresh', 'N/A'):.6f}" if isinstance(bundle.get('vol_thresh'), (int, float)) else "N/A"
    print(f"  Volatility Threshold: {vol_thresh_display}")
    print(f"  Features: {bundle.get('feats', 'N/A')}")
    if 'models' in bundle:
        for reg_name, reg_model_data in bundle['models'].items():
            error_msg = reg_model_data.get("error")
            if error_msg:
                print(f"  Regime '{reg_name}': Skipped/Failed - {error_msg}")
            else:
                thr_val = reg_model_data.get('thr')
                thr_display = f"{thr_val:.4f}" if isinstance(thr_val, (int, float, np.floating)) and not np.isnan(thr_val) else "N/A"
                print(f"  Regime '{reg_name}': threshold={thr_display}, has xgb={reg_model_data.get('xgb') is not None}, has iso={reg_model_data.get('iso') is not None}")
    else:
        print("  No models found in bundle.")

    # TODO: The Backtest and latest_features calls below will likely fail as they expect
    # a single model, iso, THRESH, and feats structure, not a regime-based bundle.
    # This needs to be addressed in a subsequent step/subtask.
    if args.bt:
        print("\nStarting backtest with regime-based models...")
        if 'models' not in bundle or not bundle['models'] or \
           bundle.get('vol_thresh') is None or np.isnan(bundle.get('vol_thresh')) or \
           'feats' not in bundle:
            print("Cannot run backtest: model bundle is incomplete (missing models, vol_thresh, or feats).")
        else:
            backtest_data = dataset()
            # Check if essential columns are present in the data from dataset()
            # These columns are produced by engineer()
            expected_eng_cols = ['atr_pct', 'bb_z', 'spread_pct', 'trend_sign', 'Open', 'Close', 'atr', 'cum_ret60'] # Added cum_ret60 for safety, though trend_sign is derived from it.
            # bundle['feats'] contains the model features.
            # Ensure all columns in bundle['feats'] and expected_eng_cols are in backtest_data.columns
            all_required_cols = list(set(bundle['feats'] + expected_eng_cols)) # Use set to avoid duplicates
            missing_cols = [col for col in all_required_cols if col not in backtest_data.columns]

            if backtest_data.empty:
                print("Backtest data is empty. Aborting backtest.")
            elif missing_cols:
                print(f"Backtest data is missing required columns: {missing_cols}. Aborting backtest.")
            else:
                Backtest(bundle).run(backtest_data)
        print("Backtest finished.")

    if args.live:
        print(f"\n{dt.datetime.now()} Starting live signal generation...")
        if not bundle or 'models' not in bundle or not bundle['models'] or \
           bundle.get('vol_thresh') is None or np.isnan(bundle.get('vol_thresh')) or \
           'feats' not in bundle:
            print(f"{dt.datetime.now()} Live: Cannot run: model bundle is incomplete or invalid.")
            return

        live_data_parts = get_live_prediction_data(bundle)

        if live_data_parts is None:
            print(f"{dt.datetime.now()} Live: Failed to get valid prediction data. No signal.")
            return

        cal_prob = live_data_parts["cal_prob"]
        regime_thr = live_data_parts["regime_thr"]
        trend = live_data_parts["trend"]
        ts = live_data_parts["timestamp"]
        latest_row = live_data_parts["latest_row_data"]

        bb_z_val = latest_row.get("bb_z")
        spread_pct_val = latest_row.get("spread_pct")

        if bb_z_val is None or pd.isna(bb_z_val) or \
           spread_pct_val is None or pd.isna(spread_pct_val):
            print(f"{ts} Live: bb_z or spread_pct missing/NaN in latest data. Cannot check eligibility.")
            print(f"    Data: bb_z={bb_z_val}, spread_pct={spread_pct_val}, atr_pct={latest_row.get('atr_pct')}")
            return

        is_eligible = Backtest._trade_eligible(latest_row)

        current_regime_str = "Unknown"
        atr_pct_val = latest_row.get("atr_pct")
        # Check atr_pct_val for NaN before passing to choose_regime
        if atr_pct_val is not None and not pd.isna(atr_pct_val) and bundle.get('vol_thresh') is not None and not np.isnan(bundle.get('vol_thresh')):
            current_regime_str = choose_regime(latest_row, bundle.get('vol_thresh'))

        # Ensure formatting is safe for potentially None or NaN values if checks above were less strict
        atr_pct_display = f"{atr_pct_val:.4f}" if atr_pct_val is not None and not pd.isna(atr_pct_val) else 'N/A'
        bb_z_display = f"{bb_z_val:.2f}" if bb_z_val is not None and not pd.isna(bb_z_val) else 'N/A'
        spread_pct_display = f"{spread_pct_val:.4f}" if spread_pct_val is not None and not pd.isna(spread_pct_val) else 'N/A'


        print(f"{ts} Live data: Regime={current_regime_str}, CalibProb={cal_prob:.3f} (Thr={regime_thr:.3f}), Trend={trend}, Eligible={is_eligible}")
        details = f"    bb_z={bb_z_display}, spread_pct={spread_pct_display}, atr_pct={atr_pct_display}"
        print(details)

        if is_eligible and cal_prob >= regime_thr:
            side = -1 if trend == 1 else 1
            etf_to_trade = ETF_SH if side == -1 else ETF_LONG
            action = "BUY"
            print(f"TRADE SIGNAL: {ts} --> {action} {etf_to_trade} (Prob={cal_prob:.3f}, Trend={trend}, Regime={current_regime_str})")
        else:
            reason = []
            if not is_eligible: reason.append("not eligible")
            if cal_prob < regime_thr: reason.append(f"prob {cal_prob:.3f} < thr {regime_thr:.3f}")
            print(f"{ts} Live: No trade signal. ({'; '.join(reason)})")

if __name__ == "__main__":
    main()
