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
import argparse, datetime as dt, os, pickle, time
from typing import List, Tuple
# Removed: sys, collections.defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
# import xgboost as xgb # Moved into functions
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression      # 🔸 NEW
import optuna

# from lightgbm import LGBMClassifier # Moved into functions
# import torch # Moved into functions
# import torch.nn as nn # Moved into functions
from sklearn.linear_model import LogisticRegression
import requests # Ensure this import is added at the top of the file


# ====== 3. STACKED ENSEMBLE  (XGB + LightGBM + Stockformer)  ==================
# Ensure torch and torch.nn are imported globally before this class is defined.
# e.g., import torch
#       import torch.nn as nn

# --- 3a. Stockformer – minimal transformer-based classifier -------------------
class Stockformer(torch.nn.Module if 'torch' in globals() and 'nn' in globals() else object): # Make conditional for when torch is not loaded
    """
    Tiny transformer for tabular sequences (lookback_window × features).
    """
    def __init__(self, n_features: int, seq_len: int = 12, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2):
        import torch
        import torch.nn as nn
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
        # x = self.embed(x) # embed is already a torch module
        x_embedded = self.embed(x) # Apply embedding
        x_transformed = self.transformer(x_embedded) # Apply transformer
        return self.cls(x_transformed) # Apply classification head


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
    import numpy as np
    # import pandas as pd # Not strictly needed here with numpy inputs

    # Conditional imports for model training
    _xgb = None
    _LGBMClassifier = None
    _torch = None
    _nn = None

    try: import xgboost as xgb; _xgb = xgb
    except ImportError: print("XGBoost (xgb) not found. Will skip.")
    try: from lightgbm import LGBMClassifier; _LGBMClassifier = LGBMClassifier
    except ImportError: print("LightGBM (LGBMClassifier) not found. Will skip.")
    try: import torch; _torch = torch
    except ImportError: print("PyTorch (torch) not found. Will skip Stockformer.")
    # nn might fail even if torch is present, if torch.nn is somehow not part of the installation
    if _torch:
        try: import torch.nn as nn; _nn = nn
        except ImportError: print("PyTorch (torch.nn) not found. Will skip Stockformer.")

    # Stockformer class and LogisticRegression are assumed to be available from global imports / definitions.

    models = {}

    # ---- XGBoost -------------------------------------------------------------
    if _xgb:
        try:
            xgb_params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="binary:logistic", eval_metric="logloss",
                              n_jobs=-1, random_state=42)
            xgb_clf = _xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train, y_train)
            models["xgb"] = xgb_clf
        except Exception as e: # Catch any error during XGBoost model init or fit
            print(f"Error during XGBoost training: {e}. Skipping XGBoost.")
            models["xgb"] = None
    else:
        # This message is now redundant due to the import message, but kept for clarity if _xgb is None for other reasons
        print("XGBoost not available (import failed or _xgb not set). Skipping XGBoost training.")
        models["xgb"] = None

    # ---- LightGBM ------------------------------------------------------------
    if _LGBMClassifier:
        try:
            lgb_params = dict(n_estimators=500, num_leaves=64,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, objective="binary", random_state=42)
            lgb_clf = _LGBMClassifier(**lgb_params)
            lgb_clf.fit(X_train, y_train)
            models["lgb"] = lgb_clf
        except Exception as e: # Catch any error during LightGBM model init or fit
            print(f"Error during LightGBM training: {e}. Skipping LightGBM.")
            models["lgb"] = None
    else:
        # Redundant message similar to XGBoost's
        print("LightGBM not available (import failed or _LGBMClassifier not set). Skipping LightGBM training.")
        models["lgb"] = None

    # ---- Stockformer ---------------------------------------------------------
    # print("Training Stockformer for ensemble...")
    # Ensure X_train is a NumPy array for .shape and slicing.
    # If X_train is Pandas DataFrame, use X_train.values
    # The issue context implies X_train, X_val are already numpy arrays by this point.

    # If feature_cols is provided and X_train is a DataFrame (not typical here based on signature)
    # X_train_values = X_train[feature_cols].values if isinstance(X_train, pd.DataFrame) and feature_cols else X_train
    # For this function, assume X_train IS a numpy array.
    X_train_values = X_train
    n_feat = X_train_values.shape[1] if X_train_values.ndim > 1 else 0


    # Inner function to create sequences for Stockformer
    # Example:
    # If X_numpy_array is (5 samples, 2 features) and seq_len = 3:
    #   [[1,10], [2,20], [3,30], [4,40], [5,50]]
    # Output will be a tensor of shape (3, 3, 2):
    #   [[[1,10], [2,20], [3,30]],  # Sequence 1
    #    [[2,20], [3,30], [4,40]],  # Sequence 2
    #    [[3,30], [4,40], [5,50]]]  # Sequence 3
    def make_tensor(X_numpy_array): # X_numpy_array must be a 2D numpy array
        # Uses _torch from outer scope if available
        if not _torch:
            # This function should not be called if _torch is None, but as a safeguard:
            raise ImportError("make_tensor requires torch to be imported and available.")
        xs_list = []
        current_n_feat = X_numpy_array.shape[1] if X_numpy_array.ndim > 1 else 0
        for idx in range(seq_len, len(X_numpy_array) + 1): # Corrected loop range
            window = X_numpy_array[idx-seq_len:idx, :]
            xs_list.append(_torch.tensor(window, dtype=_torch.float32))
        if not xs_list:
            return _torch.empty((0, seq_len, current_n_feat), dtype=_torch.float32)
        return _torch.stack(xs_list)

    Xs_train_tensor = None # Initialize
    if not _torch or not _nn or 'Stockformer' not in globals():
        # This condition checks if torch, torch.nn were successfully imported via _torch, _nn
        # and if Stockformer class definition was successful (it might not be if torch.nn was missing at class def time)
        print("PyTorch, torch.nn or Stockformer class not available. Skipping Stockformer training.")
        models["stk"] = None
    elif n_feat == 0:
        print("Stockformer training skipped: X_train has no features.")
        models["stk"] = None
    else:
        try:
            # Attempt to create tensor sequences
            Xs_train_tensor = make_tensor(X_train_values) # make_tensor uses _torch from train_ensemble scope

            if Xs_train_tensor.shape[0] == 0: # No sequences generated
                print("Stockformer training skipped: no valid training sequences generated (X_train too short).")
                models["stk"] = None
            else:
                num_stockformer_samples = Xs_train_tensor.shape[0]
                # Align y_train for Stockformer
                # Ensure y_train has enough samples after slicing for seq_len adjustment
                if len(y_train) >= seq_len -1 + num_stockformer_samples:
                    y_stockformer_train = _torch.tensor(y_train[seq_len-1 : seq_len-1 + num_stockformer_samples],
                                                       dtype=_torch.float32).view(-1,1)

                    if Xs_train_tensor.shape[0] != y_stockformer_train.shape[0]:
                        print(f"Shape mismatch! Xs_train_tensor: {Xs_train_tensor.shape}, y_stockformer_train: {y_stockformer_train.shape}. Skipping Stockformer.")
                        models["stk"] = None
                    else:
                        # Stockformer instantiation and training
                        # Stockformer class itself imports torch and torch.nn internally.
                        # If those imports fail within Stockformer, it will raise an error there.
                        if _torch: # Set seed for reproducibility if torch is available
                            _torch.manual_seed(42)
                        net = Stockformer(n_feat, seq_len) # Uses torch, nn imported by Stockformer
                        loss_fn = _nn.BCELoss() # Uses _nn from the top of train_ensemble
                        optim_ = _torch.optim.Adam(net.parameters(), lr=1e-3) # Uses _torch

                        net.train()
                        for epoch in range(5): # Small epoch count for demo
                            optim_.zero_grad()
                            preds_stk = net(Xs_train_tensor)
                            loss = loss_fn(preds_stk, y_stockformer_train)
                            loss.backward()
                            optim_.step()
                        models["stk"] = net.eval()
                else:
                    print(f"Stockformer training skipped: y_train too short for aligned labels (need {seq_len - 1 + num_stockformer_samples}, got {len(y_train)}).")
                    models["stk"] = None
        except (ImportError, NameError, AttributeError, RuntimeError) as e: # Catch issues from make_tensor or Stockformer
            print(f"Stockformer training failed due to error: {e}. Skipping.")
            models["stk"] = None
            # Xs_train_tensor = None # Already initialized to None, ensure it remains so if error

    # ---- Meta-learner --------------------------------------------------------
    # print("Training meta-learner...")
    # X_val is assumed to be a NumPy array here.
    # If it were a DataFrame, use X_val.values or X_val[feature_cols].values

    # Predictions from base models on validation set
    xgb_meta_preds = np.array([0.5] * len(X_val)) # Default if model not available
    if models.get("xgb"):
        xgb_meta_preds = models["xgb"].predict_proba(X_val)[:,1]

    lgb_meta_preds = np.array([0.5] * len(X_val))
    if models.get("lgb"):
        lgb_meta_preds = models["lgb"].predict_proba(X_val)[:,1]

    # Initialize stk_meta_preds_numpy with a default size based on X_val and seq_len
    num_expected_stk_val_samples = len(X_val) - seq_len + 1 if len(X_val) >= seq_len else 0
    if num_expected_stk_val_samples < 0: num_expected_stk_val_samples = 0 # Ensure non-negative
    stk_meta_preds_numpy = np.full(num_expected_stk_val_samples, 0.5) # Default fallback

    if models.get("stk") and _torch: # Check _torch for general availability (from train_ensemble top)
        X_val_values = X_val # Assuming X_val is already a numpy array
        if X_val_values.ndim > 1 and X_val_values.shape[1] > 0: # Ensure X_val has features
            try:
                Xs_val_tensor = make_tensor(X_val_values) # Uses _torch from train_ensemble scope
                if Xs_val_tensor.shape[0] > 0:
                    # Stockformer's forward pass uses torch components imported within Stockformer itself.
                    # _torch.no_grad() ensures inference mode.
                    with _torch.no_grad():
                        stk_predictions = models["stk"](Xs_val_tensor)
                    # .numpy() implicitly handles .detach() if tensor was on CUDA and requires grad
                    current_stk_preds = stk_predictions.numpy().flatten()

                    if len(current_stk_preds) == num_expected_stk_val_samples:
                        stk_meta_preds_numpy = current_stk_preds
                    else:
                        # This case should ideally not happen if make_tensor is consistent
                        # and Xs_val_tensor.shape[0] implies num_expected_stk_val_samples
                        print(f"Warning: Stockformer validation predictions length ({len(current_stk_preds)}) "
                              f"mismatch with expected ({num_expected_stk_val_samples}). Using fallback.")
                        # stk_meta_preds_numpy remains the pre-filled fallback
                elif num_expected_stk_val_samples > 0 : # No sequences, but expected some
                    print("Warning: Stockformer produced no validation sequences for meta-learner. Using fallback.")
                    # stk_meta_preds_numpy remains the pre-filled fallback
            except (ImportError, NameError, AttributeError, RuntimeError) as e:
                print(f"Stockformer validation prediction failed due to error: {e}. Using fallback.")
                # stk_meta_preds_numpy remains the pre-filled fallback
        elif num_expected_stk_val_samples > 0: # X_val has no features or is 1D, but expected sequences
            print("Warning: X_val for Stockformer has no features or is 1D. Using fallback for STK predictions.")
            # stk_meta_preds_numpy remains the pre-filled fallback
    # else: # Stockformer not trained, or PyTorch (_torch) not available
        # stk_meta_preds_numpy is already pre-filled with fallback
        # Optional: Print a message if STK was expected but unavailable
        # if models.get("stk") is None and _torch: # _torch means torch *could* have been available
        #     if num_expected_stk_val_samples > 0: print("Warning: Stockformer model is None. Using fallback for meta-learner.")
        # elif not _torch: # PyTorch itself was not available from the start
        #     if num_expected_stk_val_samples > 0: print("Warning: PyTorch not available. Using fallback for STK predictions in meta-learner.")

    # Align y_meta: y_val needs to be sliced like y_train for Stockformer
    # The length of y_meta must match stk_meta_preds_numpy's length (which is num_expected_stk_val_samples)
    y_meta = y_val[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)]

    xgb_meta_preds_aligned = xgb_meta_preds[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)]
    lgb_meta_preds_aligned = lgb_meta_preds[seq_len-1 : seq_len-1 + len(stk_meta_preds_numpy)]

    if not (len(xgb_meta_preds_aligned) == len(lgb_meta_preds_aligned) == len(stk_meta_preds_numpy) == len(y_meta)):
        print(f"Meta-learner input array length mismatch after alignment attempts:")
        print(f"  XGB: {len(xgb_meta_preds_aligned)}, LGB: {len(lgb_meta_preds_aligned)}, STK: {len(stk_meta_preds_numpy)}, y_meta: {len(y_meta)}")
        if len(stk_meta_preds_numpy) == 0:
            print("No data for stockformer path in meta-learner.")
            # Decide if meta-learner can be trained with only XGB/LGB if they exist and have enough samples
            # For simplicity, if STK path is empty, assume meta-learner cannot be trained robustly as designed.
            X_meta = np.array([])
        else:
             min_len = min(len(xgb_meta_preds_aligned), len(lgb_meta_preds_aligned), len(stk_meta_preds_numpy), len(y_meta))
             if min_len > 0:
                 xgb_meta_preds_aligned = xgb_meta_preds_aligned[:min_len]
                 lgb_meta_preds_aligned = lgb_meta_preds_aligned[:min_len]
                 stk_meta_preds_numpy = stk_meta_preds_numpy[:min_len]
                 y_meta = y_meta[:min_len]
                 print(f"Adjusted meta-learner input arrays to min_len: {min_len}")
                 X_meta = np.column_stack([xgb_meta_preds_aligned, lgb_meta_preds_aligned, stk_meta_preds_numpy])
             else: # min_len is 0
                 X_meta = np.array([])
    elif len(y_meta) == 0 :
        print("Meta-learner training skipped: No data available for y_meta (X_val likely too short for STK path or STK failed).")
        X_meta = np.array([])
    else:
        X_meta = np.column_stack([xgb_meta_preds_aligned, lgb_meta_preds_aligned, stk_meta_preds_numpy])

    if X_meta.shape[0] > 0 and X_meta.shape[0] == len(y_meta):
        # Ensure LogisticRegression is imported or available
        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(random_state=42).fit(X_meta, y_meta)
        models["meta"] = meta
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
    import numpy as np

    # Ensure models dictionary contains necessary components
    # Check for actual model objects, not just keys that might be None
    required_model_keys = ["xgb", "lgb", "meta"] # STK is optional depending on meta_stk_fallback
    # Check if essential models are present and not None
    essential_models_present = all(models.get(k) is not None for k in required_model_keys)
    if not essential_models_present or "seq_len" not in models:
        print("Warning: `models` dictionary in stacked_predict is missing key components (xgb, lgb, meta are None or seq_len missing).")
        if not models.get("meta_stk_fallback", False) and not models.get("stk"):
             print("Warning: `stk` model missing and meta_stk_fallback is not set. Prediction might be unreliable.")
        return 0.0

    if X_latest.ndim != 2:
        raise ValueError(f"X_latest must be a 2D array (seq_len, n_features), got {X_latest.ndim}D")

    xgb_lgb_input = X_latest[-1:, :]

    xgb_p = models["xgb"].predict_proba(xgb_lgb_input)[:,1] if models.get("xgb") else np.array([0.5])
    lgb_p = models["lgb"].predict_proba(xgb_lgb_input)[:,1] if models.get("lgb") else np.array([0.5])

    stk_p_item = 0.5 # Default
    _torch_available_for_predict = False
    try:
        import torch # Try to import torch for this specific prediction
        _torch_available_for_predict = True
    except ImportError:
        # This message can be noisy if torch is known to be unavailable system-wide.
        # Print only if a Stockformer model exists, implying it was trained and might be expected to work.
        if models.get("stk") and not models.get("meta_stk_fallback", False):
            print("PyTorch not available for stacked_predict. STK path will use fallback.")

    if models.get("stk") and not models.get("meta_stk_fallback", False) and _torch_available_for_predict:
        seq_len = models["seq_len"]
        if X_latest.shape[0] < seq_len:
            print(f"Warning: X_latest has {X_latest.shape[0]} rows, less than seq_len {seq_len}. STK prediction might be unreliable (using fallback).")
            # stk_p_item remains 0.5 (default fallback)
        else:
            stk_input_np = X_latest[-seq_len:, :]
            if stk_input_np.shape[0] == seq_len and stk_input_np.shape[1] > 0: # Also check for features
                try:
                    # Ensure torch.tensor and model call use the locally imported torch
                    xf_tensor = torch.tensor(stk_input_np, dtype=torch.float32).unsqueeze(0)
                    # model["stk"] is a Stockformer instance which has its own torch imports.
                    # torch.no_grad() ensures inference mode.
                    with torch.no_grad():
                         stk_p_tensor = models["stk"](xf_tensor)
                    stk_p_item = stk_p_tensor.item()
                except Exception as e_stk_pred: # Catch any error during STK prediction
                    print(f"Warning: Error during Stockformer prediction: {e_stk_pred}. Using fallback for STK prediction.")
                    # stk_p_item remains 0.5 (default fallback)
            elif stk_input_np.shape[1] == 0 : # No features for STK model
                 print(f"Warning: STK input has no features (shape: {stk_input_np.shape}). Using fallback for STK prediction.")
                 # stk_p_item remains 0.5
            else: # Shape mismatch (e.g. not enough rows after slicing, though outer if should catch this)
                 print(f"Warning: STK input shape {stk_input_np.shape} after slicing for seq_len {seq_len} is not as expected. Using fallback for STK prediction.")
                 # stk_p_item remains 0.5
    elif models.get("stk") and not models.get("meta_stk_fallback", False) and not _torch_available_for_predict:
        # This case is when STK model exists, it's not a meta_fallback case, but torch wasn't even importable here.
        # The earlier ImportError print might have already notified. This is an additional safeguard message if needed.
        # print("Info: Stockformer model exists but PyTorch not available for this prediction. Using fallback for STK.")
        pass # stk_p_item remains 0.5 (default fallback)

    meta_in_list = [xgb_p[0], lgb_p[0], stk_p_item]
    meta_in_array = np.array([meta_in_list])

    if models.get("meta"):
        final_prob_array = models["meta"].predict_proba(meta_in_array)[:,1]
        final_prob = float(final_prob_array[0])
    else: # Meta model is missing (e.g. if training failed or all base models failed)
        print("Warning: Meta-learner model ('meta') is missing. Returning average of available base models (or 0.5 if none).")
        # Average available predictions, or 0.5 if none available
        available_probs = [p[0] for p, m in [(xgb_p, models.get("xgb")), (lgb_p, models.get("lgb"))] if m is not None]
        if stk_p_item != 0.5 and models.get("stk"): available_probs.append(stk_p_item)
        final_prob = float(np.mean(available_probs)) if available_probs else 0.5

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
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10
START_DATE      = (dt.date.today() - dt.timedelta(days=50)).isoformat()
END_DATE        = dt.date.today().isoformat()
ETF_LONG, ETF_SH = "SPXL", "SPXS"
CAPITAL, RISK_PCT, SLIP_BP    = 100_000.0, 0.01, 5
MODEL_PATH      = "xgb_reversal.model" # Note: Ensemble model saving/loading needs review
REG_PCTL        = 0.75   # 75-th percentile of atr_pct defines “high-vol” regime

# ---------- cache & grid --------------------------------
CACHE_DIR   = "pred_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# GRID_STEPS  = 21 # Seems unused

# ---------------------- Data utilities ----------------------------------------
# START_DATE, END_DATE, INTERVAL are assumed to be defined globally in InfinitDay.py

def download(tkr: str) -> pd.DataFrame:
    df_result = pd.DataFrame()
    # Create session once, outside the loop
    session = requests.Session()
    # Standard User-Agent can help avoid some blocking issues
    headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    session.headers.update(headers)

    for attempt_number in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt_number + 1}/{MAX_RETRIES} to download {tkr} from {START_DATE} to {END_DATE} ({INTERVAL} interval)...")
            # Use the session in yf.download
            df_temp = yf.download(tkr, start=START_DATE, end=END_DATE,
                                  interval=INTERVAL, progress=False, timeout=30, session=session)

            if not df_temp.empty:
                df_result = df_temp
                print(f"Successfully downloaded data for {tkr} on attempt {attempt_number + 1}.")
                break  # Success, exit retry loop
            else:
                # yfinance returned empty df without an exception (e.g. no data for period)
                print(f"Warning: No data downloaded for {tkr} (empty DataFrame returned by yfinance) on attempt {attempt_number + 1}. Not retrying for this case.")
                break # Don't retry if yfinance returns empty successfully for the given period

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e_conn_timeout:
            print(f"Warning: Attempt {attempt_number + 1}/{MAX_RETRIES} for {tkr} failed with connection/timeout error: {e_conn_timeout}")
            # Fall through to backoff logic
        except Exception as e_other: # Catch other yfinance or unexpected errors
            error_str = str(e_other).lower()
            # Common strings indicating a rate limit or temporary server issue from yfinance or underlying request mechanisms
            is_retryable_error = ("yfratelimiterror" in error_str or
                                  "too many requests" in error_str or
                                  "rate limited" in error_str or
                                  "429" in error_str or # HTTP 429 Too Many Requests
                                  "temporary error" in error_str or
                                  "service unavailable" in error_str or # HTTP 503
                                  "internal server error" in error_str) # HTTP 500 - sometimes temporary

            if is_retryable_error:
                print(f"Warning: Attempt {attempt_number + 1}/{MAX_RETRIES} for {tkr} failed with potentially retryable error: {e_other}")
                # Fall through to backoff logic if not last attempt
            else: # Non-retryable error
                print(f"Failed to download {tkr} due to a non-retryable error: {e_other}")
                # df_result remains empty, break loop as further retries for this error are futile
                break

        # Backoff logic for retrying
        if attempt_number < MAX_RETRIES - 1:
            delay = INITIAL_BACKOFF_SECONDS * (2 ** attempt_number)
            print(f"Retrying {tkr} after {delay} seconds...")
            time.sleep(delay)
        else: # Last attempt failed
            print(f"All {MAX_RETRIES} retries failed for {tkr}.")
            # df_result remains empty

    # Post-processing, only if df_result is not empty
    if not df_result.empty:
        # Timezone localization: make index timezone-naive if it's aware
        if df_result.index.tz is not None:
            try:
                df_result = df_result.tz_localize(None)
            except TypeError as tz_e: # Specifically catch TypeError if already naive (though .tz should be None then)
                print(f"Note: Could not make timezone naive for {tkr} (possibly already naive or other issue). Error: {tz_e}")
            except Exception as e_gen_tz: # Catch any other unexpected error during tz_localize
                print(f"Warning: Error during tz_localize for {tkr}: {e_gen_tz}. Proceeding with potentially tz-aware index.")

        # Flatten MultiIndex columns if yfinance returns them (e.g. for multiple tickers, though less common for single)
        # Check if the top level has a single, possibly ticker-named, value.
        if isinstance(df_result.columns, pd.MultiIndex) and df_result.columns.nlevels > 1:
            # Check if the first level of columns has only one unique value (often the ticker itself)
            if len(df_result.columns.get_level_values(0).unique()) == 1:
                 df_result.columns = df_result.columns.droplevel(0)
            else:
                 # This case might occur if yfinance changes format or for specific data types.
                 # For now, we'll only flatten if the top level is singular to avoid accidental data loss.
                 print(f"Note: Columns for {tkr} are MultiIndex, but the top level is not singular. Not flattening automatically.")
                 # Potentially, one might want to join levels: df.columns = ['_'.join(col).strip() for col in df.columns.values]
                 # But this requires careful consideration of the expected column structure.

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


    # Ensure 'Close' and its shifted versions are positive for division/log
    close_safe = d["Close"].replace(0, np.nan).fillna(method='ffill').fillna(1e-9) # Fill 0s then ensure positive
    close_shifted_1 = d["Close"].shift(1).replace(0, np.nan).fillna(method='ffill').fillna(1e-9)
    close_shifted_lookback = d["Close"].shift(LOOKBACK).replace(0, np.nan).fillna(method='ffill').fillna(1e-9)

    d["log_ret"]   = np.log(close_safe / close_shifted_1)
    d["cum_ret60"] = close_safe / close_shifted_lookback - 1
    d["trend_sign"] = np.sign(d["cum_ret60"])

    # VWAP calculation using numpy arrays for robustness
    # Example of VWAP calculation:
    # Assume daily data for simplicity in example (though script uses intraday)
    #   High  Low  Close  Volume | Typical Price = (H+L+C)/3 | Vol*TP   | CumVol*TP | CumVol | VWAP
    #0   101   99   100    100    | 100.00                    | 10000.0  | 10000.0   | 100    | 10000.0/100    = 100.00
    #1   102   100  101    200    | 101.00                    | 20200.0  | 30200.0   | 300    | 30200.0/300    = 100.67
    #2   100   98   99     150    |  99.00                    | 14850.0  | 45050.0   | 450    | 45050.0/450    = 100.11
    # Note: .ffill().fillna(0) is applied to the final VWAP series in the code.
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
# Global TP_PCT, SL_PCT, MAX_HOLD (defined earlier) are used as defaults.
# Redundant local definitions are removed.

def triple_barrier_labels(df: pd.DataFrame, tp: float = TP_PCT, sl: float = SL_PCT, max_hold: int = MAX_HOLD) -> pd.Series:
    """ Non-overlapping triple-barrier labelling.
    +1 → take-profit hit first
    -1 → stop hit first
    0 → no decisive event / padded rows skipped

    Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Script's global constants for TP_PCT, SL_PCT, MAX_HOLD are used by default.
    >>> # For a self-contained doctest with specific parameters:
    >>> prices_doctest = [100, 101, 99, 100, 102, 100, 98, 100] # Total 8 bars
    >>> df_doctest = pd.DataFrame({'Close': prices_doctest})
    >>> # Test with TP=1%, SL=1%, Hold=2 bars
    >>> # Bar 0 (100): TP at 101 (1 bar), SL at 99 (2 bars). Window [101, 99]. TP hit first. Label 1. Skip 1 bar. i becomes 1.
    >>> # Bar 1 (101): TP at 102.01, SL at 99.99. Window [99, 100]. SL hit first. Label -1. Skip 1 bar. i becomes 2.
    >>> # Bar 2 (99): TP at 99.99, SL at 98.01. Window [100, 102]. TP hit first. Label 1. Skip 1 bar. i becomes 3.
    >>> # Bar 3 (100): TP at 101, SL at 99. Window [102, 100]. TP hit first. Label 1. Skip 1 bar. i becomes 4.
    >>> # Bar 4 (102): TP at 103.02, SL at 100.98. Window [100, 98]. SL hit first. Label -1. Skip 1 bar. i becomes 5.
    >>> # Bar 5 (100): TP at 101, SL at 99. Window [98, 100]. SL hit first. Label -1. Skip 1 bar. i becomes 6.
    >>> # Bar 6 (98): Loop condition `i < len(df) - max_hold` (6 < 8 - 2 = 6) is false. Loop terminates.
    >>> # Remaining labels (index 6, 7) are 0.
    >>> triple_barrier_labels(df_doctest, tp=0.01, sl=0.01, max_hold=2).tolist()
    [1, -1, 1, 1, -1, -1, 0, 0]
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
    # Example for vol_imb (rolling_window_ofi = 2 for simplicity in this example):
    # Assume d['ret'] and d['Volume'] are:
    # ret: [NaN, 0.1, -0.1, 0.05, -0.02]
    # Vol: [10,  20,   30,   10,    50]
    # up_vol:   [0, 20, 0,  10, 0]
    # down_vol: [0, 0,  30, 0,  50]
    # sum_up_vol (roll 2): [0, 20, 20, 10, 10]
    # sum_down_vol (roll 2): [0, 0, 30, 30, 50]
    # total_vol_for_rolling: [0, 20, 30, 10, 50]
    # sum_total_vol (roll 2): [0, 20, 50, 40, 60]
    # vol_imb = (sum_up - sum_down) / sum_total_vol (eps added to denom):
    #   idx 0: (0-0)/0 (NaN -> fillna(0)) = 0
    #   idx 1: (20-0)/20 = 1.0
    #   idx 2: (20-30)/50 = -0.2
    #   idx 3: (10-30)/40 = -0.5
    #   idx 4: (10-50)/60 = -0.666...

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
        print(f"Error: Could not download base data for {ETF_LONG} after multiple retries. Dataset generation aborted.")
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
    # model = xgb.XGBClassifier(**params) # xgb would need to be imported
#     model.fit(X_tr, y_tr)
#     preds = model.predict(X_val)
#     return -f1_score(y_val, preds)               # maximise F1

# def _train_original_backup(df: pd.DataFrame) -> Tuple[object, IsotonicRegression, float, List[str]]: # Type hint for model changed
#     # import xgboost as xgb # Would be needed here
#     feats = [c for c in df.columns if c not in ("target",)]
#     X, y  = df[feats], (df["target"] == 1).astype(int)   # 1 = TP side
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, shuffle=False)
#
#     # study = optuna.create_study(direction="minimize", show_progress_bar=False)
#     # study.optimize(lambda t: optuna_objective(t, X_tr, y_tr, X_val, y_val), # This would now call the backup
#     #                n_trials=60, timeout=900)
#
#     # best_params = study.best_params
#     # best_params.update({"n_estimators": 600, "objective": "binary:logistic",
#     #                     "eval_metric": "logloss", "n_jobs": -1})
#     # model = xgb.XGBClassifier(**best_params) # This would fail if xgb not imported/installed
#     model = None # Placeholder if XGBoost is not available
#     print("Original backup training function called, but XGBoost part is effectively disabled if not installed.")
#
#
#     # ----- probability calibration -------------------------------------------
#     # raw_val_prob = model.predict_proba(X_val)[:, 1] if model else np.zeros(len(y_val)) # Handle model=None
#     # iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val_prob, y_val)
#     # calib_prob   = iso.transform(raw_val_prob)
#     iso = IsotonicRegression(out_of_bounds="clip").fit(np.array([0,1]), np.array([0,1])) # Dummy if no model
#     calib_prob = np.array([0.5] * len(y_val)) # Dummy if no model
#
#     # choose threshold that keeps historical win-rate ≥ 55 %
#     thresh = np.quantile(calib_prob, 1 - y_val.mean()) if len(y_val) > 0 and not np.all(calib_prob == 0.5) else 0.5
#
#     # report
#     preds = (calib_prob > thresh).astype(int)
#     acc   = accuracy_score(y_val, preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
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

        # Refined equity tracking and PNL calculation
        _equity_over_time = [CAPITAL] * len(d) # Initialize with starting capital for all bars
        current_capital = CAPITAL
        trade_pnls_list = [] # List to store PNL of each closed trade

        # For ensemble, prediction needs last seq_len rows.
        # Loop from seq_len -1 to ensure enough history for the first potential prediction.
        # Decisions and trades impact capital from bar 'i' onwards.
        # Equity before loop start (i.e. for bars 0 to seq_len-2) remains CAPITAL.

        for i in range(self.seq_len -1, len(d)):
            row = d.iloc[i]
            px = row["Open"] # Entry/Exit price for current bar

            # Apply PNL from previous bar's close before making new decisions for current bar 'i'
            if i > 0: # Ensure current_capital reflects state at start of bar i
                 _equity_over_time[i] = current_capital
            else: # For the very first bar of decision making (i == 0, if seq_len was 1)
                 _equity_over_time[i] = CAPITAL


            # Try to make a prediction if no position
            if position == 0:
                if i >= self.seq_len :
                    X_latest_df = d.iloc[i - self.seq_len : i] # Features from i-seq_len to i-1
                    X_latest = X_latest_df[self.feature_cols].values

                    if X_latest.shape[0] == self.seq_len and X_latest.shape[1] == len(self.feature_cols):
                        current_prob = -1.0
                        if 'stacked_predict' in globals() and callable(globals()['stacked_predict']):
                            try:
                                current_prob = stacked_predict(self.models, X_latest)
                            except Exception as e_pred_bt:
                                print(f"BT Error at index {i} during stacked_predict: {e_pred_bt}")
                        else:
                            if i == self.seq_len:
                                print("BT WARNING: stacked_predict function not found.")
                        d.loc[d.index[i], "prob"] = current_prob

                        eligible_now = Backtest._trade_eligible(row) if hasattr(Backtest, '_trade_eligible') else True
                        backtest_threshold = 0.55

                        if eligible_now and current_prob > 0 and current_prob >= backtest_threshold:
                            d.loc[d.index[i], "signal"] = 1

            # Trade Execution Logic
            if position == 0 and d.loc[d.index[i], "signal"] == 1:
                current_trend_sign = row["trend_sign"]
                side = -1 if current_trend_sign == 1 else 1
                entry_px = px * (1 + SLIP_BP / 1e4 * side)
                stop_level_atr = row["atr"] / row["Close"] if row["Close"] > 0 and pd.notna(row["atr"]) else STOP_PCT
                effective_stop_pct = max(STOP_PCT, stop_level_atr)
                if entry_px > 0 and effective_stop_pct > 0:
                    qty = (current_capital * RISK_PCT) / (entry_px * effective_stop_pct)
                    position, bars_in_trade = side, 0
                else: qty = 0; position = 0;
            elif position != 0:
                bars_in_trade += 1
                cur_px = row["Close"]
                change = (cur_px - entry_px) * position / entry_px
                stop_level_atr_exit = row["atr"] / row["Close"] if row["Close"] > 0 and pd.notna(row["atr"]) else STOP_PCT
                effective_stop_pct_exit = max(STOP_PCT, stop_level_atr_exit)

                if change >= TP_PCT or change <= -effective_stop_pct_exit or bars_in_trade >= MAX_HOLD:
                    exit_px = cur_px * (1 - SLIP_BP / 1e4 * side)
                    pnl = qty * (exit_px - entry_px) * position
                    current_capital += pnl
                    trade_pnls_list.append(pnl) # Log PNL for this trade
                    position = 0

            # Update equity for the current bar 'i' after all operations for this bar
            _equity_over_time[i] = current_capital

        self.equity = _equity_over_time
        # Handle case where df might be too short for the loop to run even once
        if not d.empty and not self.equity: # Should not happen if len(d) >= seq_len -1
            self.equity = [CAPITAL] * len(d)
        elif d.empty:
            self.equity = [CAPITAL]


        self.cap = current_capital
        ret = (self.cap / CAPITAL - 1) * 100

        # Refined Win-Rate Calculation
        trades = len(trade_pnls_list)
        wins = sum(p > 0 for p in trade_pnls_list)
        # losses = sum(p < 0 for p in trade_pnls_list) # Not strictly needed for WR
        wr = wins / trades if trades else 0
        print(f"Return {ret:.2f}% | Win-rate {wr:.2%} | Trades {trades}")

# ---------------------- Live decision helper ----------------------------------
# Assumes engineer, choose_regime, ETF_LONG, INTERVAL are defined globally
# NOTE: This function appears to be unused by the current --live logic in main(),
# which uses an ensemble model. This function seems related to an older regime-based model.
# Consider for removal if confirmed obsolete.
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
            # bundle = {} # Assign empty bundle to prevent further errors if execution continues
            # The lines above are from the original code, the check for feats_df.empty is now more robust.
        # else: # This else is removed as the check is now handled below
        #     print(f"Dataset prepared. Shape: {feats_df.shape}")
        #     all_cols = [c for c in feats_df.columns if c not in ("target",)]
        # The lines above are from the original code, modification starts here.

        # Check if dataset is valid for training
        if feats_df.empty or 'target' not in feats_df.columns:
            print("Error: Dataset is empty or 'target' column is missing. Aborting training.")
            bundle = {} # Signal error / incomplete state for subsequent logic
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
                 print("Ensemble training process finished (or skipped).")

            # Save the bundle if training was attempted and produced a bundle
            if bundle and not bundle.get('error'):
                try:
                    with open(MODEL_PATH, "wb") as f:
                        pickle.dump(bundle, f)
                    print(f"Ensemble model bundle successfully saved to {MODEL_PATH}")
                except Exception as e_save:
                    print(f"Error saving ensemble model bundle to {MODEL_PATH}: {e_save}")
            elif bundle and bundle.get('error'):
                print(f"Ensemble training resulted in an error, not saving bundle: {bundle.get('error')}")
            else: # Should ideally not be reached if bundle is always created, but as a safeguard
                print("Ensemble training did not produce a valid bundle, not saving.")
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
        print("\nStarting backtest with ENSEMBLE models...") # Updated comment
        # Refined bundle check for ensemble model
        bundle_is_valid_for_bt = (bundle and
                                  not bundle.get('error') and
                                  bundle.get('feature_cols') and
                                  bundle.get('seq_len') is not None and
                                  (bundle.get('meta') or bundle.get('xgb') or bundle.get('lgb') or bundle.get('stk'))) # Check for at least one model component

        if not bundle_is_valid_for_bt:
            print("Cannot run backtest: model bundle is incomplete, has errors, or is missing critical components (feature_cols, seq_len, or any actual model).")
        else:
            backtest_data = dataset()
            # Check if essential columns are present in the data from dataset()
            # These columns are produced by engineer()
            expected_eng_cols = ['atr_pct', 'bb_z', 'spread_pct', 'trend_sign', 'Open', 'Close', 'atr', 'cum_ret60']
            # Use 'feature_cols' from bundle, not 'feats'
            all_required_cols = list(set(bundle['feature_cols'] + expected_eng_cols))
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
