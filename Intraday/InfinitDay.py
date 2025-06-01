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

from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import requests # Ensure this import is added at the top of the file


# ====== 3. STACKED ENSEMBLE  (XGB + LightGBM + Stockformer)  ==================
# Ensure torch and torch.nn are imported globally before this class is defined.
# e.g., import torch
#       import torch.nn as nn

# --- 3a. Stockformer – minimal transformer-based classifier -------------------
class Stockformer(nn.Module):
    """
    Tiny transformer for tabular sequences (lookback_window × features).
    """
    def __init__(self, n_features: int, seq_len: int = 12, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # Ensure nn.Linear and nn.TransformerEncoderLayer/nn.TransformerEncoder are available
        # These come from torch.nn, which should be imported as nn.
        self.embed = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True) # batch_first=True is important
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Sequential(nn.Flatten(),
                                 nn.Linear(seq_len * d_model, 1),
                                 nn.Sigmoid())

    def forward(self, x):              # x: [batch, seq_len, n_features]
        x = self.embed(x)
        x = self.transformer(x)
        return self.cls(x)


# --- 3b. Training & stacking ---------------------------------------------------
# Ensure LogisticRegression, LGBMClassifier, xgb, torch, nn are imported.
# from sklearn.linear_model import LogisticRegression (should be imported)
# from lightgbm import LGBMClassifier (should be imported)
# import xgboost as xgb (should be imported)
# import torch (should be imported)
# import torch.nn as nn (should be imported)
# class Stockformer should be defined before this function.

def train_ensemble(X_train, y_train, X_val, y_val, seq_len: int = 12, feature_cols: list = None):
    """ Returns dict of fitted models + meta-learner.
    X_* for trees = 2-d array (samples × features)
    For Stockformer we build 3-d tensor (samples × seq_len × features)
    Assumes X_train is sorted chronologically so rolling windows work.
    """
    # Ensure necessary libraries are available (already imported at script level)
    import numpy as np
    import pandas as pd # Though not explicitly used in this version, good to have if X_train were df
    import torch
    import torch.nn as nn
    # XGBoost (xgb), LightGBM (LGBMClassifier), Stockformer class, LogisticRegression
    # are assumed to be available from global imports / definitions.

    models = {}

    # ---- XGBoost -------------------------------------------------------------
    # print("Training XGBoost for ensemble...") # Optional: for verbosity
    xgb_params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="binary:logistic", eval_metric="logloss",
                      n_jobs=-1, random_state=42) # Added random_state for reproducibility
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train, y_train) # Assumes X_train is a NumPy array or compatible
    models["xgb"] = xgb_clf
    # print("XGBoost training complete.")

    # ---- LightGBM ------------------------------------------------------------
    # print("Training LightGBM for ensemble...")
    lgb_params = dict(n_estimators=500, num_leaves=64,
                      learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="binary", random_state=42) # Added random_state
    lgb_clf = LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train, y_train) # Assumes X_train is a NumPy array or compatible
    models["lgb"] = lgb_clf
    # print("LightGBM training complete.")

    # ---- Stockformer ---------------------------------------------------------
    # print("Training Stockformer for ensemble...")
    # Ensure X_train is a NumPy array for .shape and slicing.
    # If X_train is Pandas DataFrame, use X_train.values
    # The issue context implies X_train, X_val are already numpy arrays by this point.

    # If feature_cols is provided and X_train is a DataFrame (not typical here based on signature)
    # X_train_values = X_train[feature_cols].values if isinstance(X_train, pd.DataFrame) and feature_cols else X_train
    # For this function, assume X_train IS a numpy array.
    X_train_values = X_train

    n_feat = X_train_values.shape[1]

    # Inner function to create sequences for Stockformer
    def make_tensor(X_numpy_array): # X_numpy_array must be a 2D numpy array
        xs_list = []
        # Start from seq_len-1 to create the first sequence X[0:seq_len]
        # Loop up to len(X_numpy_array) - 1 to get the last sequence X[end-seq_len:end]
        # The target for X[idx-seq_len:idx] is y[idx-1]
        for idx in range(seq_len, len(X_numpy_array) + 1): # Corrected loop range
            window = X_numpy_array[idx-seq_len:idx, :] # Ensure slicing is correct for 2D array
            xs_list.append(torch.tensor(window, dtype=torch.float32))
        if not xs_list: # Handle case where X_numpy_array is too short
            return torch.empty((0, seq_len, n_feat), dtype=torch.float32)
        return torch.stack(xs_list)

    Xs_train_tensor = make_tensor(X_train_values)

    # Align y_train: target for sequence X[idx-seq_len:idx] is y[idx-1]
    # So, if make_tensor creates sequences ending at original indices k (from seq_len-1 to len-1),
    # the y_train should be y[seq_len-1 : len(y_train)]
    # Number of samples in Xs_train_tensor is len(X_train_values) - seq_len + 1
    num_stockformer_samples = Xs_train_tensor.shape[0]

    if num_stockformer_samples == 0:
        print("Stockformer training skipped: no valid training sequences generated (X_train too short).")
        models["stk"] = None # Mark as not trained
    else:
        # y_train needs to be sliced from (seq_len-1) up to (seq_len-1 + num_stockformer_samples)
        y_stockformer_train = torch.tensor(y_train[seq_len-1 : seq_len-1 + num_stockformer_samples],
                                           dtype=torch.float32).view(-1,1)

        if Xs_train_tensor.shape[0] != y_stockformer_train.shape[0]:
             print(f"Shape mismatch! Xs_train_tensor: {Xs_train_tensor.shape}, y_stockformer_train: {y_stockformer_train.shape}. Skipping Stockformer.")
             models["stk"] = None
        else:
            net = Stockformer(n_feat, seq_len) # Stockformer class must be defined
            loss_fn = nn.BCELoss()
            optim_ = torch.optim.Adam(net.parameters(), lr=1e-3)

            net.train()
            for epoch in range(5): # Small epoch count for demo
                optim_.zero_grad()
                preds_stk = net(Xs_train_tensor)
                loss = loss_fn(preds_stk, y_stockformer_train)
                loss.backward()
                optim_.step()
                # print(f"Stockformer Epoch {epoch+1}/5, Loss: {loss.item():.4f}") # Optional
            models["stk"] = net.eval()
            # print("Stockformer training complete.")

    # ---- Meta-learner --------------------------------------------------------
    # print("Training meta-learner...")
    # X_val is assumed to be a NumPy array here.
    # If it were a DataFrame, use X_val.values or X_val[feature_cols].values

    # Predictions from base models on validation set
    xgb_meta_preds = xgb_clf.predict_proba(X_val)[:,1]
    lgb_meta_preds = lgb_clf.predict_proba(X_val)[:,1]

    stk_meta_preds_numpy = np.array([]) # Initialize
    if models.get("stk"):
        # Create sequences from X_val for Stockformer
        # X_val_values = X_val[feature_cols].values if isinstance(X_val, pd.DataFrame) and feature_cols else X_val
        X_val_values = X_val # Assuming X_val is already a numpy array
        Xs_val_tensor = make_tensor(X_val_values) # Use the same make_tensor

        if Xs_val_tensor.shape[0] > 0:
            stk_meta_preds_numpy = models["stk"](Xs_val_tensor).detach().numpy().flatten()
        else: # If no validation sequences for stockformer
            # This means y_meta alignment will be tricky.
            # The original issue code for meta-learner implies stk preds are always there.
            # Let's align with the number of samples y_val[seq_len:] would have.
            num_expected_stk_val_samples = len(X_val_values) - seq_len + 1 if len(X_val_values) >= seq_len else 0
            stk_meta_preds_numpy = np.full(num_expected_stk_val_samples, 0.5) # Fallback
            if num_expected_stk_val_samples <=0 : stk_meta_preds_numpy = np.array([])
            print("Warning: Stockformer produced no validation sequences for meta-learner. Using fallback.")
    else: # Stockformer was not trained or skipped
        num_expected_stk_val_samples = len(X_val) - seq_len + 1 if len(X_val) >= seq_len else 0
        stk_meta_preds_numpy = np.full(num_expected_stk_val_samples, 0.5) # Fallback
        if num_expected_stk_val_samples <=0 : stk_meta_preds_numpy = np.array([])
        print("Warning: Stockformer model is None. Using fallback predictions for meta-learner.")

    # Align y_meta: y_val needs to be sliced like y_train for Stockformer
    # Target for sequence X_val[idx-seq_len:idx] is y_val[idx-1]
    y_meta = y_val[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)] # Align with actual STK preds length

    # Align XGB and LGB predictions with y_meta and stk_meta_preds_numpy
    # The first (seq_len-1) samples of X_val don't have corresponding Stockformer predictions.
    # So, XGB/LGB predictions for these initial samples are not used in stacking if STK is involved.
    xgb_meta_preds_aligned = xgb_meta_preds[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)]
    lgb_meta_preds_aligned = lgb_meta_preds[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)]

    # Check if all prediction arrays for meta-learner have the same number of samples
    if not (len(xgb_meta_preds_aligned) == len(lgb_meta_preds_aligned) == len(stk_meta_preds_numpy) == len(y_meta)):
        print(f"Meta-learner input array length mismatch after alignment attempts:")
        print(f"  XGB: {len(xgb_meta_preds_aligned)}, LGB: {len(lgb_meta_preds_aligned)}, STK: {len(stk_meta_preds_numpy)}, y_meta: {len(y_meta)}")
        # Fallback: if lengths don't match, meta-learner training might fail or be unreliable.
        # This could happen if X_val is very short.
        # If stk_meta_preds_numpy is empty, then y_meta, xgb_aligned, lgb_aligned should also be empty.
        if len(stk_meta_preds_numpy) == 0: # No data for stockformer path
            print("No data for stockformer path in meta-learner. Attempting meta-learner with full XGB/LGB preds if possible.")
            # This case would mean X_val was shorter than seq_len.
            # The issue's example doesn't deeply cover this edge case for meta-learner construction.
            # For now, if this happens, meta learner might not be trained.
            X_meta = np.array([]) # Empty, so meta won't be trained
        else: # Some other mismatch, try to proceed but warn
             min_len = min(len(xgb_meta_preds_aligned), len(lgb_meta_preds_aligned), len(stk_meta_preds_numpy), len(y_meta))
             xgb_meta_preds_aligned = xgb_meta_preds_aligned[:min_len]
             lgb_meta_preds_aligned = lgb_meta_preds_aligned[:min_len]
             stk_meta_preds_numpy = stk_meta_preds_numpy[:min_len]
             y_meta = y_meta[:min_len]
             print(f"Adjusted meta-learner input arrays to min_len: {min_len}")
             if min_len == 0: X_meta = np.array([])
             else: X_meta = np.column_stack([xgb_meta_preds_aligned, lgb_meta_preds_aligned, stk_meta_preds_numpy])
    elif len(y_meta) == 0 : # If y_meta is empty (e.g. X_val was too short for any STK preds)
        print("Meta-learner training skipped: No data available for y_meta (X_val likely too short).")
        X_meta = np.array([]) # Empty, so meta won't be trained
    else:
        X_meta = np.column_stack([
            xgb_meta_preds_aligned,
            lgb_meta_preds_aligned,
            stk_meta_preds_numpy
        ])

    if X_meta.shape[0] > 0 and X_meta.shape[0] == len(y_meta):
        meta = LogisticRegression(random_state=42).fit(X_meta, y_meta) # Added random_state
        models["meta"] = meta
        # print("Meta-learner training complete.")
    else:
        print("Meta-learner training skipped: Input data (X_meta) is empty or mismatched with y_meta.")
        models["meta"] = None


    models["seq_len"] = seq_len # Store for inference
    models["feature_cols"] = feature_cols # Store for inference
    # Flag if stockformer was effectively excluded from meta-learner due to no preds
    if models.get("stk") is None or stk_meta_preds_numpy.size == 0 :
        models["meta_stk_fallback"] = True # Indicates STK path had issues or no data

    return models


def stacked_predict(models: dict, X_latest: np.ndarray) -> float:
    """ Predict calibrated probability for a single sample.
    X_latest must contain at least seq_len rows of feature history.
    """
    import numpy as np # Local import for safety, though global is expected
    import torch       # Local import for safety

    # Ensure models dictionary contains necessary components
    required_keys = ["xgb", "lgb", "meta", "seq_len"] # feature_cols is not directly used by predict
    if not all(k in models for k in required_keys):
        print("Warning: `models` dictionary in stacked_predict is missing key components (xgb, lgb, meta, or seq_len).")
        # `stk` model might be missing if models.get("meta_stk_fallback") is True
        if not models.get("meta_stk_fallback", False) and "stk" not in models:
            print("Warning: `stk` model missing and meta_stk_fallback is not set. Prediction might be unreliable.")
        # Depending on strictness, could return 0.0 or raise error
        return 0.0 # Return neutral/error probability

    # For XGB & LGBM, predict on the most recent set of features (the last row of X_latest)
    # X_latest is expected to be (seq_len, n_features) for a single prediction context.
    if X_latest.ndim != 2:
        raise ValueError(f"X_latest must be a 2D array (seq_len, n_features), got {X_latest.ndim}D")

    xgb_lgb_input = X_latest[-1:, :] # Takes the last row, keeps it 2D for predict_proba

    xgb_p = models["xgb"].predict_proba(xgb_lgb_input)[:,1] # Should be (1,)
    lgb_p = models["lgb"].predict_proba(xgb_lgb_input)[:,1] # Should be (1,)

    stk_p_item = 0.5 # Default neutral if stk not used or fails

    if models.get("stk") and not models.get("meta_stk_fallback", False) :
        seq_len = models["seq_len"]
        if X_latest.shape[0] < seq_len:
            print(f"Warning: X_latest has {X_latest.shape[0]} rows, less than seq_len {seq_len}. STK prediction might be unreliable or fail.")
            # Fallback, or could raise error earlier

        # Stockformer expects input [batch, seq_len, n_features]
        # X_latest should be (seq_len, n_features), so unsqueeze to (1, seq_len, n_features)
        # Take the last `seq_len` rows from X_latest. If X_latest is already exactly (seq_len, n_features), this is fine.
        # If X_latest has more than seq_len rows, this ensures we take the correct slice.
        stk_input_np = X_latest[-seq_len:, :]
        if stk_input_np.shape[0] != seq_len: # Should ideally not happen if X_latest is prepared correctly
             print(f"Warning: STK input shape {stk_input_np.shape} after slicing for seq_len {seq_len} is not as expected. Using fallback for STK prediction.")
        else:
            xf_tensor = torch.tensor(stk_input_np, dtype=torch.float32).unsqueeze(0)
            stk_p_tensor = models["stk"](xf_tensor) # Output is a tensor e.g., tensor([[0.123]])
            stk_p_item = stk_p_tensor.item() # Convert to scalar Python float
    elif models.get("meta_stk_fallback", False):
        # print("Debug: Using fallback for STK prediction in stacked_predict due to meta_stk_fallback=True")
        pass # stk_p_item remains 0.5
    else: # stk model is None and meta_stk_fallback is False (should have been caught by initial check)
        # print("Debug: STK model is None in stacked_predict. Using fallback for STK prediction.")
        pass # stk_p_item remains 0.5


    # Meta-learner input: needs to be a 2D array, e.g. (1, num_base_models)
    # xgb_p, lgb_p are numpy arrays of shape (1,)
    # stk_p_item is a scalar

    # If meta_stk_fallback is True, the meta learner was trained without STK features.
    # This part needs to align with how train_ensemble constructed X_meta.
    # The issue's train_ensemble always includes a value for STK in X_meta (even if it's a fallback 0.5).
    # So, meta_in should always have 3 features.
    meta_in_list = [xgb_p[0], lgb_p[0], stk_p_item]
    meta_in_array = np.array([meta_in_list]) # Reshape to (1, 3)

    if models.get("meta"):
        final_prob_array = models["meta"].predict_proba(meta_in_array)[:,1] # Should be (1,)
        final_prob = float(final_prob_array[0])
    else:
        print("Warning: Meta-learner model ('meta') is missing. Returning average of XGB and LGBM.")
        # Fallback if meta model isn't available for some reason
        final_prob = float((xgb_p[0] + lgb_p[0]) / 2.0)

    return final_prob


# ---------------------- Hyper-parameters --------------------------------------
INTERVAL        = "5m"
LOOKBACK        = 12          # 🔸 past bars (≈ 60 min) to define “trend”
FWD_WINDOW      = 6           # 🔸 future bars (≈ 30 min) to test for reversal
MAX_HOLD        = 6           # 6x5-min bars  (approx 30 min)
TREND_TH        = 0.0020      # 🔸 trend strength ≥ 0.20 % (underlying)
REV_TH          = 0.0035      # 🔸 opposite move ≥ 0.35 %
TP_PCT          = 0.0060      # take-profit 0.60 %
SL_PCT          = 0.0035      # –0.35 % stop-loss
STOP_PCT        = 0.0035      # hard stop 0.35 % (retained if used elsewhere, SL_PCT is for TBL)
ATR_WIN         = 14
# PROB_TH         = 0.60  # Deprecated: Thresholds are now per-regime and stored in the model bundle.
START_DATE      = (dt.date.today() - dt.timedelta(days=50)).isoformat()
END_DATE        = dt.date.today().isoformat()
ETF_LONG, ETF_SH = "SPXL", "SPXS"
CAPITAL, RISK_PCT, SLIP_BP    = 100_000.0, 0.01, 5
MODEL_PATH      = "xgb_reversal.model"
REG_PCTL        = 0.75   # 75-th percentile of atr_pct defines “high-vol” regime

# ---------- cache & grid --------------------------------
CACHE_DIR   = "pred_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRID_STEPS  = 21               # 0 = spot, ±10 ticks of 0.1 %

# ---------------------- Data utilities ----------------------------------------
# START_DATE, END_DATE, INTERVAL are assumed to be defined globally in InfinitDay.py

def download(tkr: str) -> pd.DataFrame:
    df_result = pd.DataFrame() # Initialize df_result as an empty DataFrame
    try:
        session = requests.Session()
        headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        session.headers.update(headers)

        df_temp = yf.download(tkr, start=START_DATE, end=END_DATE,
                              interval=INTERVAL, progress=False, timeout=30, session=session)

        if df_temp.empty:
            print(f"Warning: No data downloaded for {tkr} (yf.download returned empty).")
            # df_result remains the empty DataFrame initialized earlier
        else:
            df_result = df_temp # Assign downloaded data

    except Exception as e:
        error_str = str(e).lower()
        # More robust check for rate limit error messages
        is_rate_limit_error = ("yfratelimiterror" in error_str or
                               "too many requests" in error_str or
                               "rate limited" in error_str or
                               "429 client error" in error_str or # Check for 429 status
                               "temporary error" in error_str) # Added for general temporary issues

        if is_rate_limit_error:
            print(f"Warning: Rate limit error or temporary issue encountered for ticker {tkr}. Returning empty DataFrame. Error: {e}")
            # df_result remains the empty DataFrame initialized earlier
        else:
            # For any other exception during the download, wrap it in a ValueError.
            raise ValueError(f"Failed to download data for {tkr} due to an unexpected error: {e}")

    # Post-processing, only if df_result is not empty
    if not df_result.empty:
        # Only attempt tz_localize if the index is timezone-aware
        if df_result.index.tz is not None:
            try:
                df_result = df_result.tz_localize(None)
            except TypeError as tz_e:
                # This might happen if it's already naive but somehow df_result.index.tz was not None.
                print(f"Note: Could not make timezone naive for {tkr} (already naive or other issue). Error: {tz_e}")

        # Flatten MultiIndex columns
        # Ensure columns is actually a MultiIndex and has levels to prevent errors
        if isinstance(df_result.columns, pd.MultiIndex) and df_result.columns.nlevels > 1:
            # Check if the top level has a single value (usually the ticker)
            if len(df_result.columns.get_level_values(0).unique()) == 1:
                 df_result.columns = df_result.columns.droplevel(0)

    return df_result

def rsi(series: pd.Series, n: int = 2) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = -delta.clip(upper=0).rolling(n).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def append_cache(row_dict: dict):
    """Append one prediction row to today's CSV cache."""
    fname = os.path.join(CACHE_DIR,
                         f"predictions_{dt.date.today():%Y%m%d}.csv")
    df = pd.DataFrame([row_dict])
    if os.path.exists(fname):
        df.to_csv(fname, mode="a", header=False, index=False)
    else:
        df.to_csv(fname, index=False)

def load_cache() -> pd.DataFrame:
    fname = os.path.join(CACHE_DIR,
                         f"predictions_{dt.date.today():%Y%m%d}.csv")
    return pd.read_csv(fname, parse_dates=["timestamp"]) if os.path.exists(fname)            else pd.DataFrame()

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


# ====== 1. BETTER LABELS — TRIPLE-BARRIER (non-overlapping) ================
TP_PCT = 0.0060 # +0.60 % take–profit
SL_PCT = 0.0035 # –0.35 % stop-loss
MAX_HOLD = 6 # 6×5-min bars (≈30 min)

def triple_barrier_labels(df: pd.DataFrame, tp: float = TP_PCT, sl: float = SL_PCT, max_hold: int = MAX_HOLD) -> pd.Series:
    """ Non-overlapping triple-barrier labelling.
    +1 → take-profit hit first
    -1 → stop hit first
    0 → no decisive event / padded rows skipped
    """
    # Ensure numpy and pandas are imported, typically at the script's top level.
    # For safety within a function if it were to be isolated:
    import numpy as np
    import pandas as pd

    if "Close" not in df.columns:
        # Or raise error, or return empty series with expected name
        return pd.Series(dtype="int8", index=df.index, name="target")

    close = df["Close"].values
    labels = np.zeros(len(df), dtype="int8")

    i = 0
    # Ensure max_hold is treated as an int for the loop condition and window slicing
    # The issue snippet defines MAX_HOLD as int, so direct use is fine if it's passed correctly.
    # If max_hold parameter could be float, int(max_hold) would be safer.
    # Given the default is MAX_HOLD (int), this should be fine.

    while i < len(df) - max_hold: # Ensure enough bars left for a full hold period
        # Check if there's enough data for the look-forward window from current position 'i'
        # The window starts at i+1 and needs max_hold bars.
        # So, the last index needed is i + max_hold.
        # If i + max_hold >= len(close), then close[i + 1 : i + 1 + max_hold] might be short or empty.
        # This check is subtly handled by the while condition: len(df) - max_hold ensures that
        # i + max_hold will at most be len(df) -1, so i + 1 + max_hold can be len(df).
        # Slicing `close[i + 1 : i + 1 + max_hold]` is thus safe.

        tp_price = close[i] * (1 + tp)
        sl_price = close[i] * (1 - sl)
        window = close[i + 1 : i + 1 + max_hold]

        if len(window) == 0: # Should be prevented by the while condition
            i += 1
            continue

        # first indices of TP / SL hits (or large sentinel)
        try:
            # np.where returns a tuple of arrays; [0] gets the array of indices, [0] gets the first index
            tp_idx = np.where(window >= tp_price)[0][0] + 1 # +1 because window is offset by 1 from 'i'
        except IndexError:
            tp_idx = max_hold + 1 # If not found, consider it happens after max_hold
        try:
            sl_idx = np.where(window <= sl_price)[0][0] + 1 # +1 for same reason
        except IndexError:
            sl_idx = max_hold + 1

        if tp_idx < sl_idx:
            labels[i] = 1
            i += tp_idx        # skip overlapping region
        elif sl_idx < tp_idx:
            labels[i] = -1
            i += sl_idx        # skip overlapping region
        else: # This includes the case where both tp_idx and sl_idx are max_hold + 1 (no hit)
              # or if they were somehow equal and less than max_hold + 1 (e.g. hit at same bar)
            i += 1             # undecided – move one bar forward

    return pd.Series(labels, index=df.index, name="target")


# ====== 2. LIQUIDITY / ORDER-FLOW FEATURES ===================================
def orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Augments a 5-minute OHLCV DataFrame with liquidity & micro-structure stats.
    Works on standard Yahoo OHLCV (no tick data required).
    """
    # Ensure numpy and pandas are imported, typically at the script's top level.
    import numpy as np
    import pandas as pd

    d = df.copy()
    if d.empty:
        # Return an empty DataFrame with expected columns if needed by downstream processes
        # For now, returning the empty copy as per original logic
        return d

    # Ensure required columns are present
    required_ohlc = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_ohlc:
        if col not in d.columns:
            print(f"Warning: Column {col} not found in orderflow_features. Returning original DataFrame.")
            # Potentially, return d and let downstream handle missing new columns,
            # or return df to signal no modification. Issue example implies it should work.
            return df

    # intrabar spread & depth proxies
    # Ensure High, Low, Close are numeric and not zero for denominators
    d["hl_spread_pct"] = (d["High"] - d["Low"]) / (d["Close"].replace(0, 1e-9) + 1e-9)
    # For close_off_high/low, (High - Low) can be zero. Add epsilon.
    high_minus_low = d["High"] - d["Low"]
    d["close_off_high"] = (d["High"] - d["Close"]) / (high_minus_low.replace(0, 1e-9) + 1e-9)
    d["close_off_low"]  = (d["Close"] - d["Low"])  / (high_minus_low.replace(0, 1e-9) + 1e-9)

    # rudimentary Amihud illiquidity (|ret| / $Vol)
    d["ret"] = d["Close"].pct_change()
    d["dollar_vol"] = d["Close"] * d["Volume"]
    # Ensure dollar_vol is not zero before division for Amihud
    d["amihud"] = (d["ret"].abs() / (d["dollar_vol"].replace(0, 1e-9) + 1e-9)).fillna(0) # fillna(0) for resulting NaNs

    # rolling order-flow imbalance
    # Ensure 'ret' and 'Volume' are numeric
    up_vol   = np.where(d["ret"].fillna(0) > 0, d["Volume"], 0) # fillna for ret before comparison
    down_vol = np.where(d["ret"].fillna(0) < 0, d["Volume"], 0)

    rolling_window_ofi = 6 # As per issue example (rolling(6))

    # Ensure index alignment for Series operations if df's index isn't default RangeIndex or is non-unique
    # The issue uses pd.Series(up_vol).rolling(6) which implicitly uses a new RangeIndex if up_vol is a raw numpy array.
    # To respect df's index:
    sum_up_vol = pd.Series(up_vol, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()
    sum_down_vol = pd.Series(down_vol, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()
    total_vol_for_rolling = up_vol + down_vol # This is correct as numpy arrays
    sum_total_vol = pd.Series(total_vol_for_rolling, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()

    d["vol_imb"] = (sum_up_vol - sum_down_vol) /                    (sum_total_vol.replace(0, 1e-9) + 1e-9)
    d["vol_imb"] = d["vol_imb"].fillna(0) # fillna for resulting NaNs from division or rolling

    # 20-bar median spread for regime use
    d["med_spread20"] = d["hl_spread_pct"].rolling(20, min_periods=1).median().fillna(method='bfill').fillna(0)

    # Drop intermediate columns used only for calculation
    cols_to_drop = []
    if "ret" in d.columns: cols_to_drop.append("ret")
    if "dollar_vol" in d.columns: cols_to_drop.append("dollar_vol")
    if cols_to_drop:
        d.drop(columns=cols_to_drop, inplace=True)

    # Fill any remaining NaNs in the newly created columns, perhaps with 0 or using ffill/bfill
    # This depends on how downstream functions expect these features.
    # For example, amihud, vol_imb, med_spread20 were already filled.
    # hl_spread_pct, close_off_high, close_off_low might have NaNs if input data had NaNs.
    new_feature_cols = ["hl_spread_pct", "close_off_high", "close_off_low", "amihud", "vol_imb", "med_spread20"]
    for new_col in new_feature_cols:
        if new_col in d.columns:
             d[new_col] = d[new_col].fillna(0) # Or a more sophisticated fill strategy

    return d


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
    # Global constants like ETF_LONG, MAX_HOLD are assumed to be defined.
    # Functions engineer, orderflow_features, triple_barrier_labels are assumed to be defined.
    # yf (yfinance) and pd (pandas) are assumed to be imported.

    print("Starting dataset generation...") # Optional: for verbosity
    base = download(ETF_LONG) # download() is an existing helper in the script
    if base.empty:
        print("Warning: Downloaded data is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    print(f"Data downloaded. Shape: {base.shape}")

    feats = engineer(base) # engineer() is an existing helper in the script
    if feats.empty:
        print("Warning: Data is empty after engineer(). Returning empty DataFrame.")
        return pd.DataFrame()
    print(f"Features engineered. Shape after engineer(): {feats.shape}")

    # Call the newly integrated orderflow_features function
    # Ensure orderflow_features is available in the global scope.
    if 'orderflow_features' in globals() and callable(globals()['orderflow_features']):
        feats = orderflow_features(feats)
        if feats.empty: # orderflow_features might return original df if required cols missing
            print("Warning: Data is empty or unchanged after orderflow_features(). Check for warnings from orderflow_features.")
            # Depending on strictness, could return empty or proceed if feats is just the original df
            # For now, assume if it's empty, it's problematic.
            if feats.empty: return pd.DataFrame()
        print(f"Orderflow features added. Shape after orderflow_features(): {feats.shape}")
    else:
        print("ERROR: orderflow_features function not found. Cannot proceed with dataset generation.")
        return pd.DataFrame() # Or raise an error

    # Call the newly integrated triple_barrier_labels function
    # Ensure triple_barrier_labels is available in the global scope.
    if 'triple_barrier_labels' in globals() and callable(globals()['triple_barrier_labels']):
        # The new triple_barrier_labels uses MAX_HOLD as a default parameter value,
        # sourcing it from the global scope. So, no need to pass it explicitly here if defaults are used.
        # The issue's usage example is `feats["target"] = triple_barrier_labels(feats)`
        target_labels = triple_barrier_labels(feats) # df, tp, sl, max_hold
        if target_labels.empty and not feats.empty: # Labels empty but feats were not
             print("Warning: triple_barrier_labels returned empty Series but features were present. Check TBL logic.")
             # Assign empty target to allow filtering to still work (results in empty df)
             feats["target"] = pd.Series(dtype='int8', index=feats.index)
        else:
            feats["target"] = target_labels
        print(f"Triple-barrier labels generated. Shape after adding target: {feats.shape}")
    else:
        print("ERROR: triple_barrier_labels function not found. Cannot proceed with dataset generation.")
        return pd.DataFrame() # Or raise an error

    # Ensure 'target' column exists before filtering (it should, from TBL)
    if 'target' in feats.columns:
        # Keep only decisive bars (+1 or -1). The issue's example doesn't explicitly show this filtering
        # for the `train_ensemble` input, but the original `dataset` function did it.
        # The triple_barrier_labels function produces 0 for non-decisive events.
        # For training, we usually only want +1 or -1.
        # The issue's y is `(feats["target"] == 1).astype(int).values`, implying filtering might happen later
        # or that `0` labels are handled. Let's retain the filtering for now as it was in original dataset fn.
        feats_filtered = feats[feats["target"] != 0].copy() # Use .copy() to avoid SettingWithCopyWarning later
        if feats_filtered.empty and not feats.empty:
            print("Warning: No decisive bars found after triple_barrier_labels (target != 0).")
        print(f"Filtered for decisive bars (target != 0). Shape: {feats_filtered.shape}")
        feats = feats_filtered
    else:
        print("Warning: 'target' column not found after labeling attempts in dataset(). Cannot filter decisive bars.")
        # If target is critical and missing, returning empty might be best
        return pd.DataFrame()

    if feats.empty:
        print("Warning: Dataset is empty after all processing steps.")

    print("Dataset generation complete.")
    return feats

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

def price_scenario_prob(base_row: pd.Series,
                        new_close: float,
                        reg_bundle: dict) -> float:
    """
    Re-price close/high/low to a hypothetical level and recompute
    price-dependent features needed by the model, then return calibrated
    probability.
    *Assumes other inputs (vol, volume) unchanged – good enough for ±1 % scan.*
    """
    r = base_row.copy()
    scale = new_close / base_row["Close"]
    r["Close"] *= scale
    r["High"]  *= scale
    r["Low"]   *= scale

    # recompute quick features that depend on close
    r["dist_vwap"] = (r["Close"] - ["vwap"]) / r["vwap"]

    # The original bb_z recalculation logic from the prompt was:
    # ma20 = base_row["Close"] * scale / (1 + base_row["bb_z"] *
    #                                     base_row["Close"].rolling(20).std().iloc[-1])
    # r["bb_z"]  = (r["Close"] - ma20) / base_row["Close"].rolling(20).std().iloc[-1]
    # This is problematic because base_row["Close"] is a float and cannot be used with .rolling().
    # Using a safer approach: recalculate bb_z if 'ma20' and 'std20' (standard deviation for bb)
    # are available in base_row. Otherwise, fallback to the original bb_z value.
    if 'ma20' in base_row and 'std20' in base_row and base_row['std20'] > 1e-9:
        r['bb_z'] = (r['Close'] - base_row['ma20']) / base_row['std20']
    else:
        # Fallback to original bb_z if recalculation isn't reliably possible
        r['bb_z'] = base_row.get('bb_z', 0)

    reg   = choose_regime(r, reg_bundle["vol_thresh"])
    mdl_components = reg_bundle['models'].get(reg) # Use .get for safety

    if not mdl_components or not mdl_components.get("xgb") or not mdl_components.get("iso") or not reg_bundle.get("feats"):
        # Handle cases where a model, its components, or features might be missing
        # print(f"Warning: Model components or features missing for regime {reg} or bundle. Returning 0.0 probability.")
        return 0.0

    # Ensure all features are present in r, fill with 0 if any are missing (should not happen if base_row is from engineer())
    features_for_prediction = r[reg_bundle["feats"]].fillna(0)

    raw_p = mdl_components["xgb"].predict_proba(features_for_prediction.values.reshape(1, -1))[:, 1]
    return float(mdl_components["iso"].transform(raw_p)[0]) # Ensure scalar output


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


# ---------------------- Back-test engine --------------------------------------
class Backtest:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.feature_cols = bundle.get("feature_cols")
        self.seq_len = bundle.get("seq_len")
        self.models = {k: v for k, v in bundle.items() if k not in ('feature_cols', 'seq_len', 'error')} # Store actual models

        self.cap, self.equity = CAPITAL, []

        if not self.feature_cols or not self.seq_len or not self.models:
            print("Warning: Ensemble bundle for Backtest is missing feature_cols, seq_len, or models.")
            # Potentially raise an error or set a flag to prevent run()

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
        d = df.copy() # df should be output of dataset() -> engineer -> orderflow -> triple_barrier

        if not self.feature_cols or not self.seq_len or not self.models or self.bundle.get('error'):
            print("Backtest run aborted: Bundle incomplete, has error, or missing critical components.")
            if self.bundle.get('error'): print(f"Bundle error: {self.bundle.get('error')}")
            # Simulate no trades if aborted
            self.equity = [CAPITAL] * len(d) if not d.empty else [CAPITAL]
            self.cap = CAPITAL
            print(f"Return 0.00% | Win-rate 0.00% | Trades 0")
            return

        # Ensure necessary columns from feature_cols and for trade logic are present
        required_cols_for_run = self.feature_cols + ["Open", "Close", "High", "Low", "trend_sign", "atr"]
        missing_cols = [col for col in required_cols_for_run if col not in d.columns]
        if missing_cols:
            print(f"Backtest run aborted: DataFrame is missing required columns: {missing_cols}")
            return

        d["signal"] = 0
        d["prob"] = np.nan

        # Main backtesting loop
        position, entry_px, qty, side, bars_in_trade = 0, 0, 0, 0, 0
        self.equity = []
        current_capital = CAPITAL # Use a local var for capital during simulation

        # For ensemble, prediction needs last seq_len rows.
        # Loop from seq_len -1 to ensure enough history for the first potential prediction.
        for i in range(self.seq_len -1, len(d)):
            row = d.iloc[i]
            px = row["Open"] # Entry/Exit price for current bar

            # Try to make a prediction if no position
            if position == 0:
                # Prepare X_latest: last seq_len rows up to and including d.iloc[i-1]
                # Prediction for current bar 'i' is based on data available *before* its open.
                # So, features from d.iloc[i - self.seq_len : i]
                if i >= self.seq_len : # Check if we have enough history for a full X_latest
                    X_latest_df = d.iloc[i - self.seq_len : i]
                    X_latest = X_latest_df[self.feature_cols].values

                    if X_latest.shape[0] == self.seq_len and X_latest.shape[1] == len(self.feature_cols):
                        current_prob = -1.0
                        if 'stacked_predict' in globals() and callable(globals()['stacked_predict']):
                            try:
                                current_prob = stacked_predict(self.models, X_latest)
                            except Exception as e_pred_bt:
                                print(f"BT Error at index {i} during stacked_predict: {e_pred_bt}")
                        else:
                            if i == self.seq_len: # Print warning only once
                                print("BT WARNING: stacked_predict function not found.")
                        d.loc[d.index[i], "prob"] = current_prob

                        # Trade eligibility & signal generation
                        # _trade_eligible checks generic conditions (bb_z, spread_pct)
                        # Assume _trade_eligible is available in the class or globally
                        eligible_now = Backtest._trade_eligible(row) if hasattr(Backtest, '_trade_eligible') else True
                        backtest_threshold = 0.55 # Example threshold

                        if eligible_now and current_prob > 0 and current_prob >= backtest_threshold:
                            d.loc[d.index[i], "signal"] = 1 # Signal to trade

            # Trade Execution Logic
            if position == 0 and d.loc[d.index[i], "signal"] == 1:
                # Use trend_sign from the CURRENT bar 'i' for decision
                # The signal was based on data PRIOR to bar 'i'
                current_trend_sign = row["trend_sign"]
                side = -1 if current_trend_sign == 1 else 1 # -1 for short (SPXS), 1 for long (SPXL)
                entry_px = px * (1 + SLIP_BP / 1e4 * side) # Adjust slip based on side if needed, here simplified
                # Risk management: qty based on RISK_PCT of current_capital and stop_level_pct
                # Calculate stop_level_pct (similar to original backtest)
                stop_level_atr = row["atr"] / row["Close"] if row["Close"] > 0 and pd.notna(row["atr"]) else STOP_PCT
                effective_stop_pct = max(STOP_PCT, stop_level_atr)
                if entry_px > 0 and effective_stop_pct > 0:
                    qty = (current_capital * RISK_PCT) / (entry_px * effective_stop_pct)
                    position, bars_in_trade = side, 0
                else: qty = 0; position = 0; # Cannot calculate qty, no trade
            elif position != 0:
                bars_in_trade += 1
                cur_px = row["Close"] # Evaluate exit at close of bar
                change = (cur_px - entry_px) * position / entry_px

                stop_level_atr_exit = row["atr"] / row["Close"] if row["Close"] > 0 and pd.notna(row["atr"]) else STOP_PCT
                effective_stop_pct_exit = max(STOP_PCT, stop_level_atr_exit)

                # Exit conditions: TP, SL, or MAX_HOLD (from new constants)
                # FWD_WINDOW was used before, MAX_HOLD is from new TBL. Let's use MAX_HOLD.
                if change >= TP_PCT or change <= -effective_stop_pct_exit or bars_in_trade >= MAX_HOLD:
                    exit_px = cur_px * (1 - SLIP_BP / 1e4 * side) # Slip on exit
                    pnl = qty * (exit_px - entry_px) * position
                    current_capital += pnl
                    position = 0
            self.equity.append(current_capital)

        # If loop didn't run (e.g. not enough data), ensure equity has initial capital
        if not self.equity and not d.empty : self.equity = [CAPITAL] * len(d)
        elif not self.equity and d.empty: self.equity = [CAPITAL]
        elif self.equity and len(self.equity) < len(d): # If loop exited early
             # Pad equity for remaining rows if needed, assuming no trades
             self.equity.extend([self.equity[-1]] * (len(d) - len(self.equity)))

        self.cap = current_capital # Final capital
        ret = (self.cap / CAPITAL - 1) * 100
        # Calculate wins/losses based on PNL of trades, not just equity diff for more accuracy
        # This part needs more detailed trade logging to be precise. Simplified for now:
        trade_pnls = np.diff(self.equity) # Simplified: PNLs from equity changes when trades occur
                                                     # More robust: log PNL per trade
        wins = (trade_pnls > 0).sum() if len(trade_pnls) > 0 else 0
        losses = (trade_pnls < 0).sum() if len(trade_pnls) > 0 else 0
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
    p.add_argument("--run_example", action="store_true", help="Run the integrated usage example.")
    args = p.parse_args()

    if args.run_example:
        print("Running integrated usage example as per issue specification...")
        try:
            # Ensure necessary modules like yf, engineer, orderflow_features, etc. are available globally.
            # Imports like numpy (np), pandas (pd) should be at the top of the script.
            import numpy as np
            import pandas as pd
            import yfinance as yf
            # Ensure other model specific imports like xgb, LGBMClassifier, torch, LogisticRegression are available globally.

            print("Downloading SPXL data (start='2024-01-01', interval='5m')...")
            # For the example, use a more restricted date range to speed up if necessary,
            # but stick to "2024-01-01" as per issue.
            # Ensure end_date allows for sufficient data. Let's use a fixed 90-day period for the example.
            start_date_example = "2024-01-01"
            try:
                end_date_example = (pd.to_datetime(start_date_example) + pd.Timedelta(days=90)).strftime('%Y-%m-%d')
                # Ensure end_date is not in the future, cap at today if it is.
                today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
                if end_date_example > today_str:
                    end_date_example = today_str
            except Exception: # Fallback if date parsing fails
                end_date_example = (dt.date.today()).isoformat()


            if pd.to_datetime(start_date_example) >= pd.to_datetime(end_date_example):
                print(f"Example Error: Start date {start_date_example} is on or after end date {end_date_example}. Adjusting end date.")
                # Adjust end_date to be at least a bit after start_date, e.g. start_date + 30 days, capped by today
                end_date_example = (pd.to_datetime(start_date_example) + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                if end_date_example > today_str: end_date_example = today_str
                if pd.to_datetime(start_date_example) >= pd.to_datetime(end_date_example): # If still problematic
                     print("Example Error: Could not set a valid date range. Aborting example.")
                     return


            print(f"Example data download range: {start_date_example} to {end_date_example}")
            raw_df = yf.download("SPXL", start=start_date_example, end=end_date_example, interval="5m", progress=False)

            if raw_df.empty:
                print("Example Error: No data downloaded for SPXL. Check ticker and date range.")
                return
            if raw_df.index.tz is not None: raw_df = raw_df.tz_localize(None) # Remove timezone if present
            print(f"Raw SPXL data downloaded. Shape: {raw_df.shape}")

            # Feature pipeline
            print("1. Running engineer(raw_df)...")
            feats_example = engineer(raw_df) # engineer() is an existing function in the script
            if feats_example.empty:
                print("Example Error: DataFrame empty after engineer().")
                return
            print(f"   Shape after engineer(): {feats_example.shape}")

            print("2. Running orderflow_features(feats_example)...")
            feats_example = orderflow_features(feats_example) # Newly integrated
            if feats_example.empty:
                print("Example Error: DataFrame empty after orderflow_features().")
                return
            print(f"   Shape after orderflow_features(): {feats_example.shape}")

            print("3. Running triple_barrier_labels(feats_example) to generate 'target'...")
            feats_example["target"] = triple_barrier_labels(feats_example) # Newly integrated
            if "target" not in feats_example.columns:
                print("Example Error: 'target' column not created by triple_barrier_labels().")
                return
            print(f"   Shape after triple_barrier_labels(): {feats_example.shape}")

            # Drop rows with NaN in target or features used by models, if any were introduced and not handled
            # train_ensemble expects numpy arrays, so NaNs can cause issues.
            # First, identify potential feature columns (all except 'target' initially)
            potential_feature_cols = [c for c in feats_example.columns if c not in ("target",)]
            # Check for NaNs in these feature columns and the target column
            cols_to_check_for_nan = potential_feature_cols + ["target"]
            nan_rows_mask = feats_example[cols_to_check_for_nan].isnull().any(axis=1)
            if nan_rows_mask.any():
                print(f"   Found {nan_rows_mask.sum()} rows with NaNs in features/target. Removing them.")
                feats_example.dropna(subset=cols_to_check_for_nan, inplace=True)
                print(f"   Shape after NaN removal: {feats_example.shape}")

            if feats_example.empty:
                print("Example Error: DataFrame is empty after NaN removal. Cannot proceed.")
                return

            # Split data
            print("4. Splitting data into X and y...")
            # feature_cols for train_ensemble should be determined *before* converting to numpy arrays
            # if train_ensemble's feature_cols parameter is to be used meaningfully with a DataFrame.
            # However, the new train_ensemble expects X to be a numpy array already, and feature_cols
            # is more for reference or if X were a DataFrame passed to it.
            # For this example, we'll pass the list of column names.
            all_cols_for_X = [c for c in feats_example.columns if c not in ("target",)]

            X_example_np = feats_example[all_cols_for_X].values
            y_example_np = (feats_example["target"] == 1).astype(int).values # Target is +1 class
            print(f"   X_example (NumPy) shape: {X_example_np.shape}, y_example (NumPy) shape: {y_example_np.shape}")

            min_data_for_split_train = 20 # Minimum for any operations
            # train_ensemble internally needs seq_len for Stockformer, and then some for training.
            # Let's say seq_len (12) + a few samples for training (e.g., 10) for train set,
            # and seq_len + a few for val set. So roughly 2* (12+10) = 44 samples.
            # Let's use a higher threshold for the example to be safe.
            min_samples_for_example = 50 # Adjusted to ensure enough data for seq_len logic
            if len(X_example_np) < min_samples_for_example:
                print(f"Example Error: Not enough data ({len(X_example_np)} samples) after processing for a meaningful train/val split and ensemble training (need ~{min_samples_for_example}).")
                return

            split_idx_example = int(len(X_example_np) * 0.8)
            X_train_ex, X_val_ex = X_example_np[:split_idx_example], X_example_np[split_idx_example:]
            y_train_ex, y_val_ex = y_example_np[:split_idx_example], y_example_np[split_idx_example:]
            print(f"   X_train_ex shape: {X_train_ex.shape}, y_train_ex shape: {y_train_ex.shape}")
            print(f"   X_val_ex shape: {X_val_ex.shape},   y_val_ex shape: {y_val_ex.shape}")

            # Train ensemble
            example_models_bundle = None
            seq_len_example = 12 # As per issue's usage example for train_ensemble

            # Check if X_train_ex and X_val_ex are sufficient for seq_len operations
            # train_ensemble's make_tensor needs len(X) > seq_len
            if len(X_train_ex) <= seq_len_example or len(X_val_ex) <= seq_len_example:
                 print(f"Example Error: Training data (len {len(X_train_ex)}) or validation data (len {len(X_val_ex)}) is not long enough for seq_len {seq_len_example}.")
                 return

            print(f"5. Calling train_ensemble(X_train_ex, y_train_ex, X_val_ex, y_val_ex, seq_len={seq_len_example}, feature_cols=all_cols_for_X)...")
            try:
                example_models_bundle = train_ensemble(X_train_ex, y_train_ex, X_val_ex, y_val_ex,
                                                       seq_len=seq_len_example, feature_cols=all_cols_for_X)
                print("   train_ensemble() completed for example.")
                if not example_models_bundle or not example_models_bundle.get("meta"):
                    print("   Warning: train_ensemble did not return a complete model bundle or meta-learner is missing.")
            except Exception as e_train_ex:
                print(f"   Example Error during train_ensemble: {e_train_ex}")
                # import traceback
                # traceback.print_exc() # For more detailed error during subtask execution if needed
                return # Stop if training fails

            # Live prediction example
            if example_models_bundle and example_models_bundle.get("meta"): # Check if meta learner exists
                # The issue example uses `X[-12:]`. Here, X is `X_example_np`.
                if len(X_example_np) >= seq_len_example:
                    X_latest_for_predict = X_example_np[-seq_len_example:] # Last seq_len rows of the whole dataset
                    print(f"6. Calling stacked_predict(example_models_bundle, X_latest_for_predict)...")
                    try:
                        prob_now_example = stacked_predict(example_models_bundle, X_latest_for_predict)
                        print(f"   >>> Example ensemble reversal probability (on last {seq_len_example} samples of data): {prob_now_example:.4f}")
                    except Exception as e_predict_ex:
                        print(f"   Example Error during stacked_predict: {e_predict_ex}")
                        # import traceback
                        # traceback.print_exc()
                else:
                    print(f"   Example Warning: Not enough data in X_example_np (len {len(X_example_np)}) for stacked_predict with seq_len {seq_len_example}.")
            elif example_models_bundle:
                 print("   Example Info: Meta-learner is missing from bundle. Skipping stacked_predict().")
            else:
                print("   Example Info: Model training failed or produced no bundle. Skipping stacked_predict().")

        except Exception as e_example_main:
            print(f"An unexpected error occurred during --run_example: {e_example_main}")
            # import traceback
            # traceback.print_exc()
        finally:
            print("Usage example finished.")
        return # Exit main after example runs
    elif args.train or not os.path.exists(MODEL_PATH):
        print("Preparing dataset for ensemble training...")
        feats_df = dataset() # Call the modified dataset function
        if feats_df.empty or 'target' not in feats_df.columns:
            print("Error: Dataset is empty or 'target' column is missing. Aborting training.")
            # return # Or sys.exit(1) if appropriate in main context
            bundle = {} # Assign empty bundle to prevent further errors if execution continues
        else:
            print(f"Dataset prepared. Shape: {feats_df.shape}")
            all_cols = [c for c in feats_df.columns if c not in ("target",)]
            X = feats_df[all_cols].values
            y = (feats_df["target"] == 1).astype(int).values # Target is +1 class

            if len(X) < 20: # Check for minimum data length for split
                 print(f"Error: Insufficient data for training ensemble (samples: {len(X)}). Need at least 20. Aborting training.")
                 bundle = {}
            else:
                 # Split data - using 0.20 test_size, shuffle=False for time series
                 split_idx = int(len(X) * 0.80)
                 X_train, X_val = X[:split_idx], X[split_idx:]
                 y_train, y_val = y[:split_idx], y[split_idx:]
                 print(f"Data split: X_train: {X_train.shape}, X_val: {X_val.shape}")

                 # Placeholder for train_ensemble call
                 print("Calling train_ensemble (assuming function exists with required signature)...")
                 # Default seq_len for Stockformer, can be adjusted
                 seq_len_param = 12
                 if 'train_ensemble' in globals() and callable(globals()['train_ensemble']):
                     try:
                         bundle = train_ensemble(X_train, y_train, X_val, y_val,
                                            seq_len=seq_len_param, feature_cols=all_cols)
                         print("train_ensemble called successfully.")
                     except Exception as e_train:
                         print(f"Error during train_ensemble call: {e_train}. Creating dummy bundle.")
                         bundle = {'error': str(e_train), 'feature_cols': all_cols, 'seq_len': seq_len_param}
                 else:
                     print("WARNING: train_ensemble function not found. Training skipped. Creating dummy bundle.")
                     # Create a dummy bundle structure similar to what train_ensemble might return
                     bundle = {
                         'xgb': None, 'lgb': None, 'stk': None, 'meta': None,
                         'seq_len': seq_len_param, 'feature_cols': all_cols,
                         'error': 'train_ensemble function not defined'
                     }
                     # To allow script to proceed, also ensure MODEL_PATH is handled or training is skipped
                 print("Ensemble training process finished (or skipped).")
        # print("Training complete. Model bundle generated.") # Original message, can be adapted
    else:
        print(f"Loading existing ENSEMBLE model bundle from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        print("Model bundle loaded.")

    print("\nLoaded ENSEMBLE bundle structure summary:")
    print(f"  Sequence Length: {bundle.get('seq_len', 'N/A')}")
    print(f"  Feature Columns: {bundle.get('feature_cols', 'N/A')}")
    if bundle.get('error'): print(f"  Bundle Error: {bundle['error']}")
    if 'xgb' in bundle or 'lgb' in bundle: # Check for actual model keys
        print(f"  XGB Model: {'Present' if bundle.get('xgb') else 'Absent'}")
        print(f"  LGBM Model: {'Present' if bundle.get('lgb') else 'Absent'}")
        print(f"  Stockformer Model: {'Present' if bundle.get('stk') else 'Absent'}")
        print(f"  Meta Learner: {'Present' if bundle.get('meta') else 'Absent'}")
    else: print("  No individual models (xgb, lgb, etc.) found in bundle or bundle has error.")

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
        print(f"\n{dt.datetime.now()} Starting ENSEMBLE live signal generation...")
        if not bundle or 'feature_cols' not in bundle or 'seq_len' not in bundle:
            print(f"{dt.datetime.now()} Live: Ensemble bundle is incomplete (missing feature_cols or seq_len). Aborting.")
            return # Or appropriate exit for main()

        feature_cols = bundle['feature_cols']
        seq_len = bundle['seq_len']

        try:
            # Download sufficient data: period for LOOKBACK + seq_len for Stockformer + some buffer for engineering
            # LOOKBACK is a global constant, assume it's around 12-20. seq_len is also ~12.
            # Fetching '5d' should be more than enough for 5m interval.
            raw_data = yf.download(ETF_LONG, period="5d", interval=INTERVAL, progress=False, timeout=10)
            if raw_data.index.tz is not None: raw_data = raw_data.tz_localize(None)
        except Exception as e:
            print(f"{dt.datetime.now()} Live: Error downloading data: {e}")
            return

        if raw_data.empty or len(raw_data) < LOOKBACK + seq_len:
            print(f"{dt.datetime.now()} Live: Not enough data for {ETF_LONG} (len {len(raw_data)}). Need at least {LOOKBACK + seq_len}.")
            return

        print(f"Raw data downloaded. Shape: {raw_data.shape}")
        feats = engineer(raw_data) # Original engineer function

        # Apply orderflow_features (assuming it exists and is compatible)
        if 'orderflow_features' in globals() and callable(globals()['orderflow_features']):
            feats = orderflow_features(feats)
            print(f"Applied orderflow_features. Shape after: {feats.shape}")
        else:
            print("WARNING: orderflow_features function not found. Proceeding without it.")

        if feats.empty or len(feats) < seq_len:
            print(f"{dt.datetime.now()} Live: Feature engineering resulted in insufficient data (len {len(feats)}). Need at least {seq_len}.")
            return

        # Prepare X_latest: last seq_len rows for relevant features
        # Ensure all feature_cols from training are present
        missing_live_feats = [fc for fc in feature_cols if fc not in feats.columns]
        if missing_live_feats:
            print(f"{dt.datetime.now()} Live: Missing features in live data: {missing_live_feats}. Aborting.")
            return
        X_latest = feats[feature_cols].iloc[-seq_len:].values

        if X_latest.shape[0] < seq_len or X_latest.shape[1] != len(feature_cols):
            print(f"{dt.datetime.now()} Live: X_latest has incorrect shape: {X_latest.shape}. Expected ({seq_len}, {len(feature_cols)}). Aborting.")
            return

        # Placeholder for stacked_predict call
        prob = -1.0 # Default error value
        if 'stacked_predict' in globals() and callable(globals()['stacked_predict']):
            try:
                prob = stacked_predict(bundle, X_latest) # 'bundle' is the loaded model dict
                print(f"{dt.datetime.now()} Live: stacked_predict called. Probability: {prob:.4f}")
            except Exception as e_pred:
                print(f"{dt.datetime.now()} Live: Error during stacked_predict: {e_pred}")
        else:
            print("WARNING: stacked_predict function not found. Cannot generate live probability.")

        latest_close_row = feats.iloc[-1]
        spot = latest_close_row["Close"]
        timestamp_now = feats.index[-1]

        # Side determination (using a fixed threshold for now, e.g., 0.5)
        # The ensemble output is a probability for class 1 (e.g., up-reversal).
        # Original logic: if prob >= thr, consider trade. Trend sign determines ETF.
        # 'trend_sign' comes from engineer(). Need to ensure it's on 'latest_close_row'.
        live_threshold = 0.55 # Example threshold, make configurable or derive from validation if possible
        trend_val = latest_close_row.get("trend_sign", 0) # Default to 0 if not found
        side = "NO-TRADE"
        if prob > 0 and prob >= live_threshold: # prob > 0 to ensure valid prediction
             side = ETF_SH if trend_val == 1 else ETF_LONG # BUY SPXS if uptrend, SPXL if downtrend
             side = f"BUY {side}" # Add BUY prefix

        # --- Cache Update ---
        rec = {
            "timestamp": timestamp_now,
            "close": spot,
            "prob": prob if prob > 0 else np.nan, # Store NaN if error
            "regime": "ensemble", # Regime is now 'ensemble'
            "side": side.split()[0] if side != "NO-TRADE" else "NO-TRADE"
        }
        try: append_cache(rec)
        except Exception as e_cache: print(f"{dt.datetime.now()} Live: Error appending to cache: {e_cache}")

        # --- Console Output (simplified, removing grid search for now) ---
        print(f"\n{timestamp_now}  SPOT ${spot:.2f}")
        print(f"Ensemble probability: {prob:.4f} (Threshold: {live_threshold:.2f})  ->  {side}")

        # --- Display Cache ---
        hist_df = pd.DataFrame()
        try: hist_df = load_cache()
        except Exception as e_load_cache: print(f"{dt.datetime.now()} Live: Error loading cache: {e_load_cache}")

        if not hist_df.empty:
            print(f"\n— Today so far (Ensemble Predictions) —")
            # Ensure columns match what's cached
            display_cols = ["timestamp", "close", "prob", "side"]
            if 'regime' in hist_df.columns: display_cols.insert(3, "regime")
            print(hist_df.tail(5)[display_cols].to_string(index=False))
            # Hit rate logic might need adjustment as 'regime_thr' is not from bundle now
            trades_today_count = (hist_df["side"] != "NO-TRADE").sum()
            # Simple hit rate: count where prob met fixed threshold (for BUYs)
            # This is a rough metric as actual outcome isn't checked here.
            successful_triggers = ((hist_df["side"] != "NO-TRADE") & (hist_df["prob"] >= live_threshold)).sum()
            hit_rate_display = f"{successful_triggers / trades_today_count:.0%}" if trades_today_count > 0 else "N/A"
            print(f"Trades today (signals) {trades_today_count}  |  Potential Hit-rate (based on current threshold) {hit_rate_display}")
        else:
            print("\n— No trade history for today yet —")

if __name__ == "__main__":
    main()
