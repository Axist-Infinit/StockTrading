#!/usr/bin/env python3
"""
main.py – Interactive Portfolio / Trade-Manager
===============================================

▲ Requirements
    pip install urwid yfinance tda-api ta xgboost joblib numpy pandas colorama tqdm
▲ One-off TD OAuth
    python -m tda.auth --apikey $TD_API_KEY@AMER.OAUTHAP --redirect-uri http://localhost --token-path td_token.json
"""

from __future__ import annotations
import os, sys, json, time, datetime, joblib, pathlib, warnings
import numpy as np, pandas as pd
import urwid, yfinance as yf
from tqdm import tqdm
from colorama import Fore, Style
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend    import ADXIndicator, CCIIndicator, MACD, EMAIndicator, WMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from tda import auth, client
from tda.client import Client

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent.resolve()
JSON_PATH = ROOT / "watchlist.json"
TOKEN_PATH = ROOT / "td_token.json"

# ─────────────────────────── data helpers ────────────────────────────
def load_watchlist() -> list[dict]:
    if not JSON_PATH.exists():
        print("watchlist.json not found."); sys.exit(1)
    with JSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)

# ---------- yfinance (Portfolio view) ----------
def yf_latest_close(symbols: list[str]) -> dict[str, float]:
    """Batch download last close for all symbols (1 network call)."""
    data = yf.download(
        tickers=" ".join(symbols),
        period="1d",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False,
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                out[sym] = float(data[sym]["Close"].iloc[-1])
            except KeyError:
                pass
    else:  # single symbol
        out[symbols[0]] = float(data["Close"].iloc[-1])
    return out

# ────────────────────────── Trade-Manager stack ──────────────────────
def _tda_login() -> Client:
    api_key = os.getenv("TD_API_KEY")
    if not api_key:
        raise RuntimeError("Export TD_API_KEY env-var first.")
    return auth.easy_client(api_key + "@AMER.OAUTHAP", TOKEN_PATH,
                            lambda: input("Paste redirected URL → "))

def fetch_candles(c: Client, sym: str, days: int,
                  ft: Client.PriceHistory.FrequencyType, freq: int
                 ) -> pd.DataFrame:
    res = c.get_price_history(
        sym,
        period_type=Client.PriceHistory.PeriodType.DAY,
        period=days,
        frequency_type=ft,
        frequency=freq,
        need_extended_hours_data=False
    ).json()
    cds = res.get("candles", [])
    df = pd.DataFrame(cds).rename(columns={
        "open":"Open","high":"High","low":"Low",
        "close":"Close","volume":"Volume"})
    if df.empty: return df
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_localize(None)
    df.set_index("datetime", inplace=True)
    return df

def anchored_vwap(df: pd.DataFrame, lookback: int) -> pd.Series:
    if df.empty: return pd.Series(dtype=float, index=df.index)
    anchor = df.tail(lookback)["Low"].idxmin()
    if pd.isna(anchor): return pd.Series(np.nan, index=df.index)
    after = df.loc[anchor:].copy()
    vwap = (after["Close"]*after["Volume"]).cumsum()/after["Volume"].cumsum()
    out = pd.Series(np.nan, index=df.index); out.loc[vwap.index]=vwap; return out

def add_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty: return df
    w=14
    df[f"RSI_{tf}"]=RSIIndicator(df["Close"],w).rsi()
    adx=ADXIndicator(df["High"],df["Low"],df["Close"],w)
    df[f"ADX_{tf}"]=adx.adx()
    df[f"%K_{tf}"]=StochasticOscillator(df["High"],df["Low"],df["Close"],w,3).stoch()
    df[f"CCI_{tf}"]=CCIIndicator(df["High"],df["Low"],df["Close"]).cci()
    macd=MACD(df["Close"]); df[f"MACD_{tf}"]=macd.macd()
    bb=BollingerBands(df["Close"]); df[f"BBl_{tf}"]=bb.bollinger_lband()
    df[f"ATR_{tf}"]=AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()
    if tf=="d":
        df["EMA50"]=EMAIndicator(df["Close"],50).ema_indicator()
        df["EMA200"]=EMAIndicator(df["Close"],200).ema_indicator()
        hma=2*WMAIndicator(df["Close"],10).wma()-WMAIndicator(df["Close"],3).wma()
        df["HMA"]=hma; df["DI_diff"]=df[f"ADX_{tf}"].diff()
        df["PrevHi"]=df["High"].shift(1); df["PrevLo"]=df["Low"].shift(1)
    return df

def build_features(c: Client, sym: str) -> pd.DataFrame:
    bars={
        "30m": fetch_candles(c,sym,60,Client.PriceHistory.FrequencyType.MINUTE,30),
        "1h":  fetch_candles(c,sym,120,Client.PriceHistory.FrequencyType.MINUTE,60),
        "4h":  fetch_candles(c,sym,180,Client.PriceHistory.FrequencyType.MINUTE,240),
        "d":   fetch_candles(c,sym,380,Client.PriceHistory.FrequencyType.DAILY,1)
    }
    for tf in bars:
        bars[tf]=add_indicators(bars[tf],tf)
        if tf!="d": bars[tf][f"VWAP_{tf}"]=anchored_vwap(bars[tf],200)
    bars["d"]["VWAP"]=anchored_vwap(bars["d"],252)
    # daily snapshots of intraday frames
    snap=lambda df: df.groupby(df.index.date).tail(1).set_index(
        pd.to_datetime(df.groupby(df.index.date).tail(1).index))
    feat=bars["d"].copy(); feat.index.name="Date"
    for tf in ("30m","1h","4h"):
        feat=feat.join(snap(bars[tf]), rsuffix=f"_{tf}")
    feat.dropna(subset=["Close"], inplace=True)
    return feat

def triple_barrier(df: pd.DataFrame, horizon:int=15)->pd.Series:
    cls=df["Close"].values; atr=df["ATR_d"].fillna(0)
    up=cls+2*atr; dn=cls-2*atr; lab=np.ones(len(df),int)
    for i in range(len(df)-horizon):
        w=cls[i+1:i+1+horizon]; a=np.where(w>=up[i])[0]; b=np.where(w<=dn[i])[0]
        if a.size and b.size: lab[i]=2 if a[0]<b[0] else 0
        elif a.size: lab[i]=2
        elif b.size: lab[i]=0
    return pd.Series(lab,index=df.index)

def train_model(feat: pd.DataFrame)->CalibratedClassifierCV:
    y=feat["future"]; X=feat.drop(columns=["future"]).ffill().bfill()
    # balance
    data=pd.concat([X,y],axis=1)
    m=data.future.value_counts().max()
    res=[resample(data[data.future==c],replace=True,n_samples=m,random_state=42)
         for c in (0,1,2)]
    train=pd.concat(res); Xb=train.drop(columns=["future"]); yb=train["future"]
    model=XGBClassifier(objective="multi:softprob",num_class=3,
                        tree_method="hist",device="cuda",
                        max_depth=5,learning_rate=.05,
                        n_estimators=400,subsample=.8,colsample_bytree=.8)
    cal=CalibratedClassifierCV(model,method="isotonic",cv=3)
    cal.fit(Xb,yb)
    return cal

def predict_success(c: Client, sym: str)->float:
    feat=build_features(c,sym)
    if feat.empty: return np.nan
    feat["future"]=triple_barrier(feat)
    model=train_model(feat)
    latest=feat.drop(columns=["future"]).iloc[-1]
    p=model.predict_proba(latest.to_frame().T.values)[0]
    return float(max(p))  # prob of predicted class

# ──────────────────────────── urwid views ────────────────────────────
PALETTE=[
    ('title','white,bold',''),
    ('menu','light cyan,bold',''),
    ('headers','yellow,bold',''),
    ('body','white',''),
    ('positive','dark green',''),
    ('negative','dark red',''),
]

def portfolio_table(data: list[dict], show_prob: bool=False):
    cols=["Symbol","Current","Buy-Zone","Stop","Targets"]
    if show_prob: cols.append("P-Success")
    hdr=''.join(f"{c:>12}" if c!="Symbol" else f"{c:<8}" for c in cols)
    rows=[('headers',hdr+'\n')]
    for rec in data:
        color='positive' if rec["Current"]>=rec["BuyHigh"] else 'negative'
        targets='/'.join(f"{t:.2f}" for t in rec["Targets"])
        parts=[rec["Symbol"].ljust(8),
               f"{rec['Current']:>12.2f}",
               f"{rec['BuyLow']:.2f}-{rec['BuyHigh']:.2f:>4}",
               f"{rec['Stop']:>12.2f}",
               f"{targets:>12}"]
        if show_prob:
            p=f"{rec['P']*100:6.2f}%"
            parts.append(f"{p:>12}")
        line=''.join(parts)
        rows.append((color,line+'\n'))
    return rows

def view_portfolio(holdings: list[dict]):
    syms=[h["ticker"] for h in holdings]
    prices=yf_latest_close(syms)
    table=[]
    for h in holdings:
        low,high=[float(x.replace('$','').strip()) for x in h["buy_zone"].split('–')]
        tgt=[float(t.replace('$','').strip()) for t in h["profit_targets"]]
        stop=float(h["stop_loss"].split()[-1].replace('$',''))
        table.append({
            "Symbol":h["ticker"],
            "Current":prices.get(h["ticker"],np.nan),
            "BuyLow":low,
            "BuyHigh":high,
            "Stop":stop,
            "Targets":tgt,
            "P":np.nan
        })
    body=urwid.Text(portfolio_table(table), align='left')
    layout=urwid.Frame(header=urwid.AttrMap(urwid.Text(" Portfolio (YF quotes) ", 'center'),'title'),
                       body=urwid.AttrMap(urwid.Filler(body,'top'),'body'),
                       footer=urwid.AttrMap(urwid.Text("(Q)uit"),'menu'))
    urwid.MainLoop(layout,PALETTE,unhandled_input=lambda k: k.lower()=='q' and raise_exit()).run()

def view_trade_manager(holdings: list[dict]):
    try:
        c=_tda_login()
    except Exception as e:
        print(e); input("Press Enter"); return
    table=[]
    for h in tqdm(holdings,desc="Predicting"):
        low,high=[float(x.replace('$','').strip()) for x in h["buy_zone"].split('–')]
        tgt=[float(t.replace('$','').strip()) for t in h["profit_targets"]]
        stop=float(h["stop_loss"].split()[-1].replace('$',''))
        cur=float(yf_latest_close([h["ticker"]]).get(h["ticker"],np.nan))
        try:
            p= predict_success(c,h["ticker"])
        except Exception:
            p=np.nan
        table.append({"Symbol":h["ticker"],"Current":cur,"BuyLow":low,
                      "BuyHigh":high,"Stop":stop,"Targets":tgt,"P":p})
    body=urwid.Text(portfolio_table(table,show_prob=True), align='left')
    layout=urwid.Frame(header=urwid.AttrMap(urwid.Text(" Trade Manager (TD + ML) ", 'center'),'title'),
                       body=urwid.AttrMap(urwid.Filler(body,'top'),'body'),
                       footer=urwid.AttrMap(urwid.Text("(Q)uit"),'menu'))
    urwid.MainLoop(layout,PALETTE,unhandled_input=lambda k: k.lower()=='q' and raise_exit()).run()

def raise_exit():
    raise urwid.ExitMainLoop()

# ───────────────────────────────── main menu ─────────────────────────
def main_menu():
    while True:
        os.system('cls' if os.name=='nt' else 'clear')
        print("=== Main Menu ===")
        print("1. View Portfolio")
        print("2. View Trade Manager")
        print("0. Exit")
        choice=input("Select option: ").strip()
        if choice=='1':
            view_portfolio(load_watchlist())
        elif choice=='2':
            view_trade_manager(load_watchlist())
        elif choice=='0':
            break

if __name__=="__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nBye.")
