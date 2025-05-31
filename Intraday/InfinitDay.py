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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model*4, # Common practice
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
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

def train_ensemble(X_train_df, y_train_np, X_val_df, y_val_np, # Changed to accept DataFrames for X for .values access
                   seq_len: int = 12, feature_cols: list = None):
    """
    Returns dict of fitted models + meta-learner.
    X_* for trees = 2-d array (samples × features) -> now expects DataFrames to use .values
    For Stockformer we build 3-d tensor (samples × seq_len × features)
    Assumes X_train_df is sorted chronologically so rolling windows work.
    y_train_np and y_val_np are expected to be numpy arrays.
    """
    import numpy as np # Local import for safety, though global is expected
    import pandas as pd # Local import for safety
    import torch # Local import for safety
    import torch.nn as nn # Local import for safety
    # Ensure xgb, LGBMClassifier, LogisticRegression are available from global imports

    models = {}

    # Convert X DataFrames to numpy arrays for XGBoost and LightGBM
    X_train_np = X_train_df[feature_cols].values if isinstance(X_train_df, pd.DataFrame) and feature_cols else X_train_df
    X_val_np = X_val_df[feature_cols].values if isinstance(X_val_df, pd.DataFrame) and feature_cols else X_val_df

    if not feature_cols:
        if isinstance(X_train_df, pd.DataFrame):
            feature_cols = X_train_df.columns.tolist()
        elif isinstance(X_train_np, np.ndarray) and X_train_np.ndim == 2:
             print("Warning: feature_cols not provided to train_ensemble, deriving from X_train shape if possible or using generic names.")
             # Cannot get names from numpy array directly, this list will be for reference in bundle
             feature_cols = [f'feat_{i}' for i in range(X_train_np.shape[1])]
        else:
            raise ValueError("feature_cols must be provided if X_train is not a DataFrame.")


    # ---- XGBoost -------------------------------------------------------------
    print("Training XGBoost for ensemble...")
    # Ensure xgb is imported and available
    if 'xgb' not in globals(): raise NameError("xgboost (xgb) not imported or not in global scope.")

    xgb_params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="binary:logistic", eval_metric="logloss",
                      n_jobs=-1, random_state=42) # Added random_state
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train_np, y_train_np)
    models["xgb"] = xgb_clf
    print("XGBoost training complete.")

    # ---- LightGBM ------------------------------------------------------------
    print("Training LightGBM for ensemble...")
    # Ensure LGBMClassifier is imported and available
    try: from lightgbm import LGBMClassifier # Check if it was imported
    except ImportError: raise NameError("LGBMClassifier not imported.")

    lgb_params = dict(n_estimators=500, num_leaves=64,
                      learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="binary", random_state=42) # Added random_state
    lgb_clf = LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train_np, y_train_np)
    models["lgb"] = lgb_clf
    print("LightGBM training complete.")

    # ---- Stockformer ---------------------------------------------------------
    print("Training Stockformer for ensemble...")
    # Ensure Stockformer class is defined
    if 'Stockformer' not in globals(): raise NameError("Stockformer class not defined.")

    # Expects X_train_df to be a DataFrame for .values, or X_train_np if DataFrame not passed
    # The input X for make_tensor should be a numpy array.
    # If X_train_df was passed, use its .values. If X_train_np was the original input, use that.
    source_X_for_stockformer_train = X_train_df[feature_cols].values if isinstance(X_train_df, pd.DataFrame) else X_train_np

    n_feat = source_X_for_stockformer_train.shape[1]

    def make_tensor_train(X_numpy_array, sequence_len): # Renamed to avoid conflict if make_tensor is global
        # X_numpy_array should be the .values from the DataFrame or the numpy array directly
        xs_list = []
        # Start from sequence_len to ensure enough lookback
        for idx_t in range(sequence_len, len(X_numpy_array) +1): # +1 to include last possible window
            # Slice from idx_t - sequence_len to idx_t
            window = X_numpy_array[idx_t - sequence_len : idx_t]
            if window.shape[0] == sequence_len: # Ensure window is of correct length
                 xs_list.append(torch.tensor(window, dtype=torch.float32))
            # else: print(f"Skipping window at idx_t {idx_t} due to insufficient length {window.shape[0]}") # Debug
        if not xs_list:
            # This case means no valid sequences could be formed, e.g. if len(X_numpy_array) < sequence_len
            # Return an empty tensor or handle as an error
            print(f"Warning: No sequences created for Stockformer. Input length {len(X_numpy_array)}, seq_len {sequence_len}")
            return torch.empty((0, sequence_len, n_feat), dtype=torch.float32) # n_feat must be defined
        return torch.stack(xs_list)

    Xs_train_tensor = make_tensor_train(source_X_for_stockformer_train, seq_len)
    # Align y_train: it should correspond to the *end* of each sequence in Xs_train_tensor
    # If Xs_train_tensor has N samples, these correspond to original indices seq_len-1 to N+seq_len-2
    # So y_train_np should be sliced from index seq_len-1 onwards.
    # Example: if seq_len=12, first sequence is X[0:12], target is y[11]
    # make_tensor_train starts making sequences for target at index `seq_len-1` in original data
    # So, y_train_np needs to be sliced from `seq_len-1` up to `len(y_train_np)`
    # Number of samples in Xs_train_tensor is len(source_X_for_stockformer_train) - seq_len + 1

    num_stockformer_samples = Xs_train_tensor.shape[0]
    if num_stockformer_samples == 0:
        print("Stockformer training skipped: no valid training sequences generated.")
        models["stk"] = None # Mark as not trained
    else:
        # y_train_np needs to be sliced from (seq_len -1) up to (seq_len -1 + num_stockformer_samples)
        # Example: X is 100 samples, seq_len is 10.
        # First sequence is X[0:10], target y[9]. Last sequence X[90:100], target y[99].
        # make_tensor_train loop: range(10, 101). idx_t=10 -> X[0:10]. Target y[9].
        # So y_stockformer_train = y_train_np[seq_len-1 : len(y_train_np)] -> this gives all possible targets
        # We need to align it with number of samples from make_tensor_train
        y_stockformer_train = torch.tensor(y_train_np[seq_len-1 : seq_len-1 + num_stockformer_samples], dtype=torch.float32).view(-1,1)

        if Xs_train_tensor.shape[0] != y_stockformer_train.shape[0]:
             print(f"Shape mismatch! Xs_train: {Xs_train_tensor.shape}, ys_train_stk: {y_stockformer_train.shape}. Skipping Stockformer.")
             models["stk"] = None
        else:
            net = Stockformer(n_feat, seq_len) # Stockformer class must be defined
            loss_fn = nn.BCELoss() # BCELoss expects target to be float
            optim_ = torch.optim.Adam(net.parameters(), lr=1e-3)

            net.train() # Set model to training mode
            for epoch in range(5): # Small epoch count for demo
                optim_.zero_grad()
                preds_stk = net(Xs_train_tensor)
                loss = loss_fn(preds_stk, y_stockformer_train)
                loss.backward()
                optim_.step()
                # print(f"Stockformer Epoch {epoch+1}/5, Loss: {loss.item():.4f}") # Optional print
            models["stk"] = net.eval() # Set model to evaluation mode
            print("Stockformer training complete.")

    # ---- Meta-learner --------------------------------------------------------
    print("Training meta-learner...")
    # Ensure LogisticRegression is imported
    try: from sklearn.linear_model import LogisticRegression # Check if it was imported
    except ImportError: raise NameError("LogisticRegression not imported.")

    # Build out-of-fold probabilities for validation set
    # X_val_np is for xgb and lgb
    # For Stockformer, X_val_df (or X_val_np if df not passed) is used to make tensor
    source_X_for_stockformer_val = X_val_df[feature_cols].values if isinstance(X_val_df, pd.DataFrame) else X_val_np

    # Predictions from base models
    xgb_meta_preds = xgb_clf.predict_proba(X_val_np)[:,1]
    lgb_meta_preds = lgb_clf.predict_proba(X_val_np)[:,1]

    stk_meta_preds_list = []
    if models.get("stk"):
        Xs_val_tensor = make_tensor_train(source_X_for_stockformer_val, seq_len) # Use same make_tensor
        if Xs_val_tensor.shape[0] > 0:
            stk_meta_preds_list = models["stk"](Xs_val_tensor).detach().numpy().flatten()
        else: # If no validation sequences for stockformer
            stk_meta_preds_list = np.array([0.5] * len(X_val_np)) # Fallback: neutral probability
            # This needs alignment if stk_meta_preds_list has different length
            # The y_meta will be y_val_np[seq_len-1 : seq_len-1 + Xs_val_tensor.shape[0]]
            # If Xs_val_tensor is empty, this means we can't use stockformer predictions for meta-learner
            # A more robust solution would be to only stack models that trained successfully and produced predictions
            print("Warning: Stockformer produced no validation sequences. Meta-learner might be suboptimal.")
    else: # If Stockformer wasn't trained
        # Fallback: If stk model doesn't exist, use a neutral prediction (0.5) for stacking
        # The length must match other meta predictions (xgb_meta_preds, lgb_meta_preds)
        # This path is problematic if we need to align y_meta.
        # For simplicity, if stk is None, we might exclude it from stacking or use this placeholder.
        # Let's assume for now, if stk is None, we may need to adjust X_meta and y_meta alignment.
        # The issue example assumes stk is always present in the meta learner.
        # So if models['stk'] is None, this step will likely fail or produce poor results.
        # A robust implementation would handle this by conditional stacking.
        # Given the y_meta slicing below, we'll assume stk_meta_preds need to be of length len(y_val_np[seq_len-1:])
        num_samples_for_stk_val_y = len(y_val_np) - (seq_len -1) if len(y_val_np) >= seq_len else 0
        stk_meta_preds_list = np.full(num_samples_for_stk_val_y, 0.5) if num_samples_for_stk_val_y > 0 else np.array([])


    # Align y_meta: y_val_np needs to be sliced like y_train_np for Stockformer
    # This means y_meta corresponds to the targets for which Stockformer could make predictions
    # If Stockformer produced predictions on Xs_val_tensor (num_stk_val_samples),
    # then y_meta should be y_val_np[seq_len-1 : seq_len-1 + num_stk_val_samples]
    num_stk_val_samples = Xs_val_tensor.shape[0] if 'Xs_val_tensor' in locals() and Xs_val_tensor.shape[0] > 0 else 0

    if num_stk_val_samples > 0 :
        y_meta = y_val_np[seq_len-1 : seq_len-1 + num_stk_val_samples]
        # Align other model predictions to this length
        xgb_meta_preds_aligned = xgb_meta_preds[seq_len-1 : seq_len-1 + num_stk_val_samples]
        lgb_meta_preds_aligned = lgb_meta_preds[seq_len-1 : seq_len-1 + num_stk_val_samples]
        # stk_meta_preds_list should already be this length from Xs_val_tensor
        if len(stk_meta_preds_list) != num_stk_val_samples: # Should not happen if logic is correct
            print(f"STK meta preds length mismatch: {len(stk_meta_preds_list)} vs {num_stk_val_samples}. Readjusting.")
            # This is a fallback, ideally make_tensor_train for val and y_val slicing are robust
            stk_meta_preds_list = np.full(num_stk_val_samples, 0.5) # Fallback to neutral

        X_meta = np.column_stack([
            xgb_meta_preds_aligned,
            lgb_meta_preds_aligned,
            stk_meta_preds_list # This should be correctly sized from Xs_val_tensor
        ])
    else: # Case where Stockformer produced no valid predictions (e.g. X_val too short)
        print("Warning: Stockformer produced no valid validation predictions. Meta-learner will use only XGB and LGBM.")
        # Use full X_val_np for XGB and LGBM, and full y_val_np
        y_meta = y_val_np
        X_meta = np.column_stack([
            xgb_meta_preds, # Full length
            lgb_meta_preds  # Full length
        ])
        # Note: If this path is taken, stacked_predict will also need to know to only use xgb, lgb

    if X_meta.shape[0] == 0: # If no data for meta learner
        print("Meta-learner training skipped: No data available for X_meta.")
        models["meta"] = None
    else:
        meta = LogisticRegression(random_state=42).fit(X_meta, y_meta) # Added random_state
        models["meta"] = meta
        print("Meta-learner training complete.")

    models["seq_len"] = seq_len # Store for inference
    models["feature_cols"] = feature_cols # Store for inference
    if num_stk_val_samples == 0 : # Flag if stockformer was excluded from meta
        models["meta_excluded_stk"] = True

    return models


def stacked_predict(models: dict, X_latest_np: np.ndarray) -> float:
    """
    Predict calibrated probability for a single new sample or a batch.
    X_latest_np must contain at least `seq_len` rows of feature history if Stockformer is used.
    If predicting for a single new instance, X_latest_np would be (seq_len, n_features).
    The prediction is for the time step *after* the last row in X_latest_np.
    """
    import numpy as np # Local import for safety
    import torch # Local import for safety

    # Ensure models dictionary contains necessary components
    if not all(k in models for k in ["xgb", "lgb", "meta", "seq_len", "feature_cols"]):
        # Stockformer ('stk') might be optional if 'meta_excluded_stk' is True
        if not (models.get("meta_excluded_stk") and 'stk' not in models):
             print("Warning: `models` dictionary in stacked_predict is missing key components.")
             return 0.0 # Return neutral/error probability

    seq_len = models["seq_len"]
    # For XGB & LGBM, we predict on the most recent set of features, i.e., the last row of X_latest_np
    # X_latest_np is expected to be (num_samples_for_prediction, n_features) or (seq_len, n_features)
    # If X_latest_np has more rows than 1 (e.g. seq_len rows for stockformer), take last for xgb/lgb

    # If X_latest_np is shaped (seq_len, n_features), it means we're predicting for ONE instance
    # after this sequence. So, xgb/lgb use the features from X_latest_np[-1].
    # If X_latest_np is shaped (num_instances > 1, n_features), it means we're batch predicting.
    # For now, let's assume X_latest_np is for ONE new prediction, so it's (seq_len, n_features)
    # or just (1, n_features) if seq_len=1 (though ensemble implies seq_len > 1 for stockformer)

    if X_latest_np.ndim == 2: # e.g. (seq_len, n_features) or (1, n_features)
        xgb_lgb_input = X_latest_np[-1:, :] # Takes the last row, keeps it 2D
    elif X_latest_np.ndim == 3: # e.g. (batch_size, seq_len, n_features)
        # This is more for batch prediction with Stockformer.
        # For xgb/lgb, we'd take the last feature set from each sequence in the batch.
        # xgb_lgb_input = X_latest_np[:, -1, :]
        print("Warning: stacked_predict received 3D X_latest_np. Assuming xgb/lgb predict on last slice. Untested path.")
        xgb_lgb_input = X_latest_np[:, -1, :]
    else:
        raise ValueError(f"X_latest_np has unsupported ndim: {X_latest_np.ndim}")

    xgb_p = models["xgb"].predict_proba(xgb_lgb_input)[:,1]
    lgb_p = models["lgb"].predict_proba(xgb_lgb_input)[:,1]

    stk_p_val = 0.5 # Default neutral if stk not used or fails
    if models.get("stk") and not models.get("meta_excluded_stk"):
        # Stockformer expects input of shape [batch, seq_len, n_features]
        # If X_latest_np was (seq_len, n_features) for one prediction, add batch dim
        if X_latest_np.ndim == 2 and X_latest_np.shape[0] == seq_len:
            xf_tensor = torch.tensor(X_latest_np, dtype=torch.float32).unsqueeze(0)
        elif X_latest_np.ndim == 3 and X_latest_np.shape[1] == seq_len: # Already batched
            xf_tensor = torch.tensor(X_latest_np, dtype=torch.float32)
        else: # Shape not suitable for stockformer with current seq_len
            print(f"Warning: X_latest_np shape {X_latest_np.shape} not directly usable for Stockformer with seq_len {seq_len}. Using fallback for stk_p.")
            # This might cause issues if meta learner expects stk_p
            # Fallback to a neutral prediction for stk_p. Length should match xgb_p, lgb_p.
            # This path indicates a potential issue in how data is passed or if seq_len doesn't match.
            stk_p = np.full_like(xgb_p, 0.5) # Match length of other preds
            # If stk_p needs to be a scalar for single prediction:
            stk_p_val = 0.5

        if 'xf_tensor' in locals(): # If tensor was created
            stk_p = models["stk"](xf_tensor).detach().numpy().flatten() # .item() if single, .flatten() if batch
            # If xgb_p is scalar (single prediction), stk_p should also be scalar
            if xgb_p.size == 1 and stk_p.size == 1:
                stk_p_val = stk_p[0]
            elif xgb_p.size == stk_p.size : # Batch prediction, sizes match
                 stk_p_val = stk_p # Keep as array
            else: # Size mismatch, fallback
                print(f"Warning: stk_p size {stk_p.size} mismatch with xgb_p size {xgb_p.size}. Using fallback for stk_p_val.")
                stk_p_val = np.full_like(xgb_p, 0.5) if xgb_p.size > 1 else 0.5


    # Prepare input for meta-learner
    if models.get("meta_excluded_stk"):
        meta_in = np.column_stack([xgb_p, lgb_p])
    else: # Stockformer was included
        # Ensure stk_p_val is correctly shaped (scalar or array matching xgb_p, lgb_p)
        if isinstance(stk_p_val, float) and isinstance(xgb_p, np.ndarray): # Convert scalar stk_p_val to array
            stk_p_val_final = np.full_like(xgb_p, stk_p_val)
        elif isinstance(stk_p_val, np.ndarray) and stk_p_val.shape != xgb_p.shape: # Reshape if necessary (e.g. (1,) vs (1,1))
            stk_p_val_final = stk_p_val.reshape(xgb_p.shape) if stk_p_val.size == xgb_p.size else np.full_like(xgb_p, 0.5)
        else:
            stk_p_val_final = stk_p_val

        meta_in = np.column_stack([xgb_p, lgb_p, stk_p_val_final])

    final_prob = models["meta"].predict_proba(meta_in)[:,1]

    # If predicting for a single instance, return a float
    return float(final_prob[0]) if final_prob.size == 1 else final_prob


# ---------------------- Hyper-parameters --------------------------------------
INTERVAL        = "5m"
LOOKBACK        = 12          # 🔸 past bars (≈ 60 min) to define “trend”
FWD_WINDOW      = 6           # 🔸 future bars (≈ 30 min) to test for reversal
MAX_HOLD        = 6           # 6x5-min bars  (approx 30 min)
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

# ---------- cache & grid --------------------------------
CACHE_DIR   = "pred_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRID_STEPS  = 21               # 0 = spot, ±10 ticks of 0.1 %

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


# ====== 1. BETTER LABELS — TRIPLE-BARRIER  (non-overlapping)  ================
# Constants like TP_PCT, SL_PCT, MAX_HOLD are assumed to be defined globally in InfinitDay.py
# For default parameters, we reference them. If they are not global when this function
# is defined, Python will raise NameError. They are defined in the Hyper-parameters section.

def triple_barrier_labels(df: pd.DataFrame,
                          tp: float = TP_PCT,  # Assumes TP_PCT is global
                          sl: float = SL_PCT,  # Assumes SL_PCT is global
                          max_hold: int = MAX_HOLD) -> pd.Series: # Assumes MAX_HOLD is global
    """
    Non-overlapping triple-barrier labelling.
    +1 -> take-profit hit first
    -1 -> stop hit first
     0 -> no decisive event / padded rows skipped
    """
    import numpy as np
    import pandas as pd

    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype="int8", index=df.index, name="target")

    close = df["Close"].values
    labels = np.zeros(len(df), dtype="int8")

    i = 0
    # Ensure max_hold is treated as an int for the loop condition and window slicing
    max_hold_int = int(max_hold)

    while i < len(df) - max_hold_int:
        # Check if there's enough data for the look-forward window from current position 'i'
        # The window starts at i+1 and needs max_hold_int bars.
        # So, the last index needed is i + max_hold_int.
        # If i + max_hold_int >= len(close), then close[i + 1 : i + 1 + max_hold_int] might be short or empty.
        if i + max_hold_int >= len(close): # Corrected boundary condition
            break

        tp_price = close[i] * (1 + tp)
        sl_price = close[i] * (1 - sl)

        # Window is from bar i+1 up to i+1+max_hold_int-1
        window = close[i + 1 : i + 1 + max_hold_int]

        if len(window) == 0: # Should be prevented by the while condition and boundary check
            i += 1
            continue

        try:
            # np.where returns a tuple of arrays; [0][0] gets the first index from the first array
            tp_idx = np.where(window >= tp_price)[0][0] + 1 # +1 because window is offset by 1 from 'i'
        except IndexError:
            tp_idx = max_hold_int + 1 # If not found, consider it happens after max_hold_int
        try:
            sl_idx = np.where(window <= sl_price)[0][0] + 1 # +1 for same reason
        except IndexError:
            sl_idx = max_hold_int + 1

        if tp_idx < sl_idx:
            labels[i] = 1
            i += tp_idx        # skip overlapping region
        elif sl_idx < tp_idx:
            labels[i] = -1
            i += sl_idx        # skip overlapping region
        else: # This includes the case where both tp_idx and sl_idx are max_hold_int + 1 (no hit)
            i += 1             # undecided – move one bar forward

    return pd.Series(labels, index=df.index, name="target")


# ====== 2. LIQUIDITY / ORDER-FLOW FEATURES  ===================================

def orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments a 5-minute OHLCV DataFrame with liquidity & micro-structure stats.
    Works on standard Yahoo OHLCV (no tick data required).
    """
    import numpy as np
    import pandas as pd

    d = df.copy()
    if d.empty:
        return d

    # Ensure required columns are present
    required_ohlc = ["Open", "High", "Low", "Close", "Volume"] # Adjusted to common case
    for col in required_ohlc:
        if col not in d.columns:
            print(f"Warning: Column {col} not found in orderflow_features. Returning original DataFrame.")
            return df # Return original df if essential columns are missing

    # intrabar spread & depth proxies
    d["hl_spread_pct"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-9) # Added epsilon for safety
    d["close_off_high"] = (d["High"] - d["Close"]) / (d["High"] - d["Low"] + 1e-9)
    d["close_off_low"]  = (d["Close"] - d["Low"])  / (d["High"] - d["Low"] + 1e-9)

    # rudimentary Amihud illiquidity (|ret| / $Vol)
    d["ret"] = d["Close"].pct_change()
    d["dollar_vol"] = d["Close"] * d["Volume"]
    # Ensure dollar_vol is not zero before division for Amihud
    d["amihud"] = (d["ret"].abs() / (d["dollar_vol"].replace(0, 1e-9) + 1e-9)) # replace 0s in dollar_vol too

    # rolling order-flow imbalance
    up_vol   = np.where(d["ret"] > 0, d["Volume"], 0)
    down_vol = np.where(d["ret"] < 0, d["Volume"], 0)

    rolling_window_ofi = 6

    # Ensure index alignment for Series operations if df's index isn't default RangeIndex
    sum_up_vol = pd.Series(up_vol, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()
    sum_down_vol = pd.Series(down_vol, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()
    # Ensure up_vol and down_vol are arrays for direct addition before pd.Series conversion
    total_vol_for_rolling = up_vol + down_vol
    sum_total_vol = pd.Series(total_vol_for_rolling, index=d.index).rolling(rolling_window_ofi, min_periods=1).sum()

    d["vol_imb"] = (sum_up_vol - sum_down_vol) / (sum_total_vol.replace(0, 1e-9) + 1e-9) # replace 0s

    # 20-bar median spread for regime use
    d["med_spread20"] = d["hl_spread_pct"].rolling(20, min_periods=1).median()

    # Fill NaNs that may have been created (e.g. from pct_change, rolling ops at the beginning)
    # It's often better to fill NaNs at the very end or let them propagate if subsequent logic handles them.
    # For these specific features, 0 or bfill/ffill might be appropriate.
    cols_to_fill_na = ["amihud", "vol_imb", "med_spread20", "ret"]
    for col_ff in cols_to_fill_na:
        if col_ff in d.columns:
            if col_ff == "med_spread20": # Median spread can be backfilled then zero-filled
                d[col_ff] = d[col_ff].fillna(method='bfill').fillna(0)
            elif col_ff == "ret": # pct_change makes first row NaN
                 d[col_ff] = d[col_ff].fillna(0) # Fill first NaN ret with 0
            else: # amihud, vol_imb can be 0 if not calculable
                d[col_ff] = d[col_ff].fillna(0)

    # Drop intermediate columns used only for calculation
    # Check if 'ret' and 'dollar_vol' are in columns before dropping to avoid KeyError
    cols_to_drop = []
    if "ret" in d.columns: cols_to_drop.append("ret")
    if "dollar_vol" in d.columns: cols_to_drop.append("dollar_vol")
    if cols_to_drop:
        d.drop(columns=cols_to_drop, inplace=True)

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
    base   = download(ETF_LONG)
    feats  = engineer(base)
    # The following functions are assumed to be defined elsewhere as per plan:
    # feats = orderflow_features(feats)
    # feats["target"] = triple_barrier_labels(feats, max_hold=MAX_HOLD) # Pass MAX_HOLD
    # For now, to make the script runnable without these functions defined,
    # we will call the existing/renamed functions or placeholder logic.
    # This will need to be manually updated when the actual functions are available.

    # Attempt to call orderflow_features if it exists, else skip with a message
    if 'orderflow_features' in globals() and callable(globals()['orderflow_features']):
        feats = orderflow_features(feats)
    else:
        print("WARNING: orderflow_features function not found, skipping this step in dataset().")

    # Attempt to call new triple_barrier_labels if it exists, else use old or label_reversals
    if 'triple_barrier_labels' in globals() and callable(globals()['triple_barrier_labels']):
        import inspect
        try:
            sig = inspect.signature(globals()['triple_barrier_labels'])
            if 'max_hold' in sig.parameters: # Check for new signature
                feats["target"] = triple_barrier_labels(feats, max_hold=MAX_HOLD) # Call new one
            elif '_old_triple_barrier_labels' in globals() and callable(globals()['_old_triple_barrier_labels']):
                print("WARNING: New triple_barrier_labels signature mismatch, using _old_triple_barrier_labels in dataset().")
                feats["target"] = _old_triple_barrier_labels(feats)
            else: # Fallback to label_reversals if new TBL has wrong sig and old doesn't exist
                print("WARNING: New triple_barrier_labels signature mismatch and _old_triple_barrier_labels not found, using label_reversals in dataset().")
                feats = label_reversals(feats) # label_reversals already creates 'target'
        except (ValueError, TypeError): # Handle cases where signature cannot be inspected
            print("WARNING: Could not inspect signature of triple_barrier_labels, using label_reversals in dataset().")
            feats = label_reversals(feats) # label_reversals already creates 'target'
    elif '_old_triple_barrier_labels' in globals() and callable(globals()['_old_triple_barrier_labels']):
        print("WARNING: New triple_barrier_labels not found, using _old_triple_barrier_labels in dataset().")
        feats["target"] = _old_triple_barrier_labels(feats)
    else: # Fallback to label_reversals if no triple_barrier_labels defined
        print("WARNING: triple_barrier_labels (new or old) not defined, falling back to label_reversals for target creation in dataset().")
        feats = label_reversals(feats) # This function creates 'target' column

    # Ensure 'target' column exists before filtering
    if 'target' in feats.columns:
        feats = feats[feats["target"] != 0]          # only decisive bars
    else:
        print("WARNING: 'target' column not found after labeling attempts in dataset(). Cannot filter decisive bars.")
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
    r["dist_vwap"] = (r["Close"] - r["vwap"]) / r["vwap"]

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
        print("Running integrated usage example...")
        try:
            # Ensure yf is available (should be imported as yf)
            if 'yf' not in globals(): print("Error: yfinance (yf) not imported."); return

            print("Downloading SPXL data for example (2024-01-01 onwards, 5m interval)...")
            # Using a slightly shorter period for quicker example run if needed, e.g., last 60-90 days
            # For consistency with issue, using 2024-01-01. If too long, adjust START_DATE_EXAMPLE
            START_DATE_EXAMPLE = "2024-01-01"
            # END_DATE_EXAMPLE = dt.date.today().isoformat() # Or a fixed recent date for consistency
            # To avoid issues if today is too close to start_date for enough data:
            end_date_obj = dt.datetime.strptime(START_DATE_EXAMPLE, '%Y-%m-%d') + dt.timedelta(days=90)
            END_DATE_EXAMPLE = end_date_obj.isoformat()
            print(f'Example data range: {START_DATE_EXAMPLE} to {END_DATE_EXAMPLE}')

            raw_df = yf.download("SPXL", start=START_DATE_EXAMPLE, end=END_DATE_EXAMPLE, interval="5m", progress=False)
            if raw_df.empty:
                print("Example Error: No data downloaded for SPXL.")
                return
            if raw_df.index.tz is not None: raw_df = raw_df.tz_localize(None)
            print(f"Raw data shape: {raw_df.shape}")

            # Feature pipeline
            print("Running engineer()...")
            feats_example = engineer(raw_df) # engineer() is an existing function
            print(f"Shape after engineer(): {feats_example.shape}")

            if 'orderflow_features' in globals() and callable(globals()['orderflow_features']):
                print("Running orderflow_features()...")
                feats_example = orderflow_features(feats_example)
                print(f"Shape after orderflow_features(): {feats_example.shape}")
            else: print("Skipping orderflow_features (not found). Example will be incomplete.")

            if 'triple_barrier_labels' in globals() and callable(globals()['triple_barrier_labels']):
                import inspect
                try:
                    sig = inspect.signature(globals()['triple_barrier_labels'])
                    if 'max_hold' in sig.parameters: # Check if it's the new one
                        print("Running new triple_barrier_labels()...")
                        feats_example["target"] = triple_barrier_labels(feats_example)
                        print(f"Shape after triple_barrier_labels(): {feats_example.shape}")
                    else: raise ValueError("Not the new TBL function (missing max_hold param).")
                except Exception as e_tbl_check:
                    print(f"Skipping new triple_barrier_labels ({e_tbl_check}). Example will be incomplete or use fallback.")
                    if '_old_triple_barrier_labels' in globals() and callable(globals()['_old_triple_barrier_labels']):
                        print("Using _old_triple_barrier_labels as fallback for example...")
                        feats_example["target"] = _old_triple_barrier_labels(feats_example)
                    elif 'label_reversals' in globals() and callable(globals()['label_reversals']):
                        print("Using label_reversals as ultimate fallback for example...")
                        feats_example = label_reversals(feats_example) # Creates 'target'
                    else: print("No suitable labeling function found for example.")
            else: print("Skipping triple_barrier_labels (not found). Example will be incomplete.")

            if feats_example.empty or 'target' not in feats_example.columns:
                print("Example Error: Feature DataFrame is empty or 'target' is missing after pipeline.")
                return

            # Split
            all_cols_example = [c for c in feats_example.columns if c not in ("target",)]
            X_example = feats_example[all_cols_example].values
            y_example = (feats_example["target"] == 1).astype(int).values
            print(f"X_example shape: {X_example.shape}, y_example shape: {y_example.shape}")

            if len(X_example) < 20: # Min data for split and train_ensemble
                print(f"Example Error: Not enough data after processing ({len(X_example)} samples) for training example.")
                return

            split_idx_example = int(len(X_example) * 0.8)
            X_train_ex, X_val_ex = X_example[:split_idx_example], X_example[split_idx_example:]
            y_train_ex, y_val_ex = y_example[:split_idx_example], y_example[split_idx_example:]
            print(f"X_train_ex: {X_train_ex.shape}, X_val_ex: {X_val_ex.shape}")

            example_models = None
            if 'train_ensemble' in globals() and callable(globals()['train_ensemble']):
                print("Running train_ensemble() for example...")
                try:
                    example_models = train_ensemble(X_train_ex, y_train_ex, X_val_ex, y_val_ex,
                                                 seq_len=12, feature_cols=all_cols_example)
                    print("train_ensemble() completed for example.")
                except Exception as e_train_ex:
                    print(f"Error during example train_ensemble: {e_train_ex}")
            else: print("Skipping train_ensemble (not found). Example cannot complete training/prediction part.")

            if example_models and 'stacked_predict' in globals() and callable(globals()['stacked_predict']):
                if len(X_val_ex) >= example_models.get('seq_len', 12): # Use X_val_ex for prediction example
                    print("Running stacked_predict() for example...")
                    # Predict on the first part of X_val_ex that has enough history
                    # For this example, let's take last seq_len from X_train_ex to predict first of X_val_ex or use X_val_ex itself
                    # The issue example uses X[-12:], which means last 12 of entire X.
                    # Let's use last seq_len of X_example for simplicity of demo here.
                    seq_len_ex = example_models.get('seq_len', 12)
                    if len(X_example) >= seq_len_ex:
                       prob_now_example = stacked_predict(example_models, X_example[-seq_len_ex:])
                       print(f"Example ensemble reversal probability (on last {seq_len_ex} samples of data): {prob_now_example:.4f}")
                    else: print(f"Not enough data in X_example for stacked_predict with seq_len {seq_len_ex}")
                else: print("Not enough validation data for stacked_predict example or seq_len missing in bundle.")
            elif example_models: print("Skipping stacked_predict (not found). Example cannot complete prediction part.")
            else: print("Skipping stacked_predict as models were not trained.")

        except Exception as e_example:
            print(f"An error occurred during --run_example: {e_example}")
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
