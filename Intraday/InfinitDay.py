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

# ---------------------- Imports ------------------------------------------------
import argparse, datetime as dt, os, sys
from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------- Hyper-parameters --------------------------------------
INTERVAL        = "5m"
LOOKBACK        = 12          # 🔸 past bars (≈ 60 min) to define “trend”
FWD_WINDOW      = 6           # 🔸 future bars (≈ 30 min) to test for reversal
TREND_TH        = 0.0020      # 🔸 trend strength ≥ 0.20 % (underlying)
REV_TH          = 0.0035      # 🔸 opposite move ≥ 0.35 %
TP_PCT          = 0.0060      # take-profit 0.60 %
STOP_PCT        = 0.0035      # hard stop 0.35 %
ATR_WIN         = 14
PROB_TH         = 0.60
START_DATE      = "2024-01-01"
END_DATE        = dt.date.today().isoformat()
ETF_LONG, ETF_SH = "SPXL", "SPXS"
CAPITAL, RISK_PCT, SLIP_BP    = 100_000.0, 0.01, 5
MODEL_PATH      = "xgb_reversal.model"

# ---------------------- Data utilities ----------------------------------------
def download(tkr: str) -> pd.DataFrame:
    df = yf.download(tkr, start=START_DATE, end=END_DATE,
                     interval=INTERVAL, progress=False)
    if df.empty: raise ValueError(f"No data for {tkr}")
    return df.tz_localize(None)

def rsi(series: pd.Series, n: int = 2) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = -delta.clip(upper=0).rolling(n).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

# ---------------------- Feature engineering -----------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_ret"]   = np.log(d["Close"] / d["Close"].shift(1))
    d["cum_ret60"] = d["Close"] / d["Close"].shift(LOOKBACK) - 1  # 🔸past hour
    d["trend_sign"] = np.sign(d["cum_ret60"])
    d["vwap"]      = (d["Volume"] * (d[["High","Low","Close"]].mean(axis=1))).cumsum() / \
                     d["Volume"].cumsum()
    d["dist_vwap"] = (d["Close"] - d["vwap"]) / d["vwap"]
    ma20           = d["Close"].rolling(20).mean()
    d["bb_z"]      = (d["Close"] - ma20) / d["Close"].rolling(20).std()
    d["rsi2"]      = rsi(d["Close"], 2)
    d["atr"]       = (d["High"] - d["Low"]).rolling(ATR_WIN).mean()
    # lagged returns for pattern context
    for l in (1,2,3,4,5):
        d[f"log_ret_l{l}"] = d["log_ret"].shift(l)
    d.dropna(inplace=True)
    return d

# ---------------------- Label construction ------------------------------------
def label_reversals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fut_ret = d["Close"].shift(-FWD_WINDOW) / d["Close"] - 1
    cond_strong = abs(d["cum_ret60"]) >= TREND_TH               # strong trend
    cond_flip   = (np.sign(d["cum_ret60"]) * np.sign(fut_ret) == -1) & \
                  (abs(fut_ret) >= REV_TH)                      # opposite ≥ 0.35 %
    d["target"] = (cond_strong & cond_flip).astype(int)
    d.dropna(inplace=True)
    return d

def dataset() -> pd.DataFrame:
    base = download(ETF_LONG)          # we learn on SPXL only – cheaper
    return label_reversals(engineer(base))

# ---------------------- Model training ----------------------------------------
def train(df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, List[str]]:
    feats = [c for c in df.columns if c not in ("target",)]
    X, y  = df[feats], df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=-1, objective="binary:logistic"
    )
    model.fit(Xtr, ytr)
    preds  = model.predict(Xte)
    prob   = model.predict_proba(Xte)[:,1]
    acc    = accuracy_score(yte, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary")
    print(f"ACC {acc:.2f}  PREC {prec:.2f}  REC {rec:.2f}  F1 {f1:.2f}")
    model.save_model(MODEL_PATH)
    return model, feats

# ---------------------- Back-test engine --------------------------------------
class Backtest:
    def __init__(self, model, feats):
        self.m, self.f = model, feats
        self.cap, self.equity = CAPITAL, []

    def run(self, df: pd.DataFrame) -> None:
        d = df.copy()
        d["prob"]   = self.m.predict_proba(d[self.f])[:,1]
        d["trend"]  = np.sign(d["cum_ret60"])
        d["signal"] = (d["prob"] >= PROB_TH).astype(int)
        position, entry_px, qty, side, bars_in_trade = 0, 0, 0, 0, 0

        for i, row in d.iterrows():
            px = row["Open"]  # assume fill at bar open
            # ---- open trade --------------------------------------------------
            if position == 0 and row["signal"] == 1:
                side  = -1 if row["trend"] == 1 else 1            # 🔸 opposite
                etf   = ETF_SH if side == -1 else ETF_LONG
                slip  = px * (1 + SLIP_BP/1e4)
                entry_px = slip
                qty   = (self.cap * RISK_PCT) / entry_px
                position, bars_in_trade = side, 0
            # ---- manage open trade -------------------------------------------
            elif position != 0:
                bars_in_trade += 1
                etf = ETF_SH if position == -1 else ETF_LONG
                cur_px  = px
                change  = (cur_px - entry_px) * position / entry_px
                # profit / stop or max holding
                if change >= TP_PCT or change <= -max(STOP_PCT, row["atr"]/row["Close"]) \
                   or bars_in_trade >= FWD_WINDOW:
                    exit_px = px * (1 - SLIP_BP/1e4)
                    pnl     = qty * (exit_px - entry_px) * position
                    self.cap += pnl
                    position = 0
            self.equity.append(self.cap)

        # final metrics
        ret = (self.cap / CAPITAL - 1) * 100
        wins = (np.diff(self.equity) > 0).sum()
        trades = wins + (np.diff(self.equity) < 0).sum()
        wr = wins / trades if trades else 0
        print(f"Return {ret:.2f}% | Win-rate {wr:.2%} | Trades {trades}")

# ---------------------- Live decision helper ----------------------------------
def latest_features(model, feats):
    df = yf.download(ETF_LONG, period="3d", interval=INTERVAL, progress=False)
    df = engineer(df).tail(LOOKBACK+1)          # ensure sufficient history
    df = label_reversals(df)                    # will create cum_ret etc.
    if df.empty: return None
    x = df.iloc[-1:][feats]
    prob = model.predict_proba(x)[:, 1][0]
    trend = int(df.iloc[-1]["trend_sign"])
    return prob, trend, df.index[-1]

# ---------------------- CLI ---------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--bt",    action="store_true")
    p.add_argument("--live",  action="store_true")
    args = p.parse_args()

    if args.train or not os.path.exists(MODEL_PATH):
        model, feats = train(dataset())
    else:
        model = xgb.XGBClassifier(); model.load_model(MODEL_PATH)
        feats = [c for c in dataset().columns if c not in ("target",)]

    if args.bt:
        Backtest(model, feats).run(dataset())

    if args.live:
        out = latest_features(model, feats)
        if out is None:
            print("Not enough data yet.")
            return
        prob, trend, ts = out
        if prob < PROB_TH:
            print(f"{ts}  prob={prob:.2f}  no signal")
            return
        side = -1 if trend == 1 else 1
        etf  = ETF_SH if side == -1 else ETF_LONG
        print(f"{ts}  prob={prob:.2f}  → reversal expected – BUY {etf}")

if __name__ == "__main__":
    main()
