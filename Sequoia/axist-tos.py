#!/usr/bin/env python3


import os
import sys
import argparse
import datetime
from datetime import timedelta
import json
import joblib, pandas as pd
import numpy as np
# import yfinance as yf # Removed
import urwid
from pathlib import Path
import alpaca_trade_api
from alpaca_trade_api.rest import TimeFrame

from watchlist_utils import load_watchlist, save_watchlist, manage_watchlist
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import CCIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from ta.trend import ADXIndicator, CCIIndicator, MACD, EMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import MFIIndicator  # optional if you want more volume-based features
from ta.volatility import DonchianChannel  # optional, if needed
from colorama import init, Fore, Style
init(autoreset=True)
from fredapi import Fred

import configparser
import warnings
from pandas.errors import PerformanceWarning      # already triggered by pandas
warnings.filterwarnings(
    "ignore",
    message=r"The behavior of array concatenation with empty entries is deprecated",
    category=FutureWarning,
)

PREDICTIONS_FILE = "weekly_signals.json"
DATA_CACHE: dict[tuple, tuple] = {}      # key → (df_5m, df_30m, df_1h, df_2h, df_1d)
ALPACA_API_CLIENT = None

ALPACA_TIMEFRAME_MAP = {
    '1m': TimeFrame.Minute1, '5m': TimeFrame.Minute5, '15m': TimeFrame.Minute15,
    '30m': TimeFrame.Minute30, '1h': TimeFrame.Hour1, '2h': TimeFrame.Hour2, # Added '2h'
    '1d': TimeFrame.Day
}

def get_start_end_dates_for_period(period_str, end_date_dt=None):
    if end_date_dt is None:
        end_date_dt = datetime.datetime.now(datetime.timezone.utc)
    num = int(period_str[:-1])
    unit = period_str[-1].lower()
    if unit == 'd': start_date_dt = end_date_dt - timedelta(days=num)
    elif unit == 'mo': start_date_dt = end_date_dt - timedelta(days=num * 30)
    elif unit == 'y': start_date_dt = end_date_dt - timedelta(days=num * 365)
    else: start_date_dt = end_date_dt - timedelta(days=num)
    return start_date_dt, end_date_dt

def get_or_fetch(ticker: str, start=None, end=None):
    key = (ticker, start, end)
    if key not in DATA_CACHE:
        DATA_CACHE[key] = fetch_data(ticker, start=start, end=end)
    return DATA_CACHE[key]

def load_config():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.ini")
    config.read(config_path)
    fred_api_key = config['FRED'].get('api_key', None)
    alpaca_api_key, alpaca_secret_key, alpaca_base_url = None, None, None
    if 'ALPACA' in config:
        alpaca_api_key = config['ALPACA'].get('api_key', None)
        alpaca_secret_key = config['ALPACA'].get('secret_key', None)
        alpaca_base_url = config['ALPACA'].get('base_url', 'https://paper-api.alpaca.markets')
    return fred_api_key, alpaca_api_key, alpaca_secret_key, alpaca_base_url

def initialize_alpaca_client(api_key, secret_key, base_url):
    global ALPACA_API_CLIENT
    if not api_key or not secret_key or not base_url:
        print(Fore.YELLOW + "Alpaca API credentials incomplete. Cannot initialize client." + Style.RESET_ALL)
        ALPACA_API_CLIENT = None; return
    try:
        api = alpaca_trade_api.REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version='v2')
        account = api.get_account()
        print(Fore.GREEN + "Successfully connected to Alpaca API." + Style.RESET_ALL)
        ALPACA_API_CLIENT = api
    except Exception as e:
        print(Fore.RED + f"Failed to connect to Alpaca API: {e}" + Style.RESET_ALL)
        ALPACA_API_CLIENT = None

def load_predictions():
    if not os.path.isfile(PREDICTIONS_FILE): return []
    try:
        with open(PREDICTIONS_FILE, "r") as fh: data = json.load(fh)
    except Exception: return []
    for rec in data:
        rec.setdefault("status", "Open")
        rec.setdefault("entry_date", datetime.datetime.today().strftime("%Y-%m-%d"))
    return data

def save_predictions(pred_list):
    with open(PREDICTIONS_FILE, "w") as fh: json.dump(pred_list, fh, indent=2)

def _update_positions_status() -> None:
    if ALPACA_API_CLIENT is None:
        print(Fore.RED + "Alpaca API client not initialized. Cannot update positions status." + Style.RESET_ALL); return
    preds = load_predictions(); open_pos = [p for p in preds if p.get("status") == "Open"]
    if not open_pos: return
    symbols = sorted({p["symbol"] for p in open_pos}); price_map = {}
    if not symbols: return
    print(f"Updating status for {len(symbols)} open positions using Alpaca...")
    for sym in symbols:
        try:
            latest_trade = ALPACA_API_CLIENT.get_latest_trade(sym)
            price_map[sym] = latest_trade.price
        except Exception as e:
            print(Fore.YELLOW + f"Could not fetch latest price for {sym} from Alpaca: {e}. Using entry price as fallback." + Style.RESET_ALL)
            pos_for_fallback = next((p for p in open_pos if p["symbol"] == sym), None)
            price_map[sym] = pos_for_fallback["entry_price"] if pos_for_fallback else 0
    today = datetime.date.today().isoformat(); changed = False
    for rec in open_pos:
        sym = rec["symbol"]; now_price = float(price_map.get(sym, rec["entry_price"]))
        if rec["direction"] == "LONG": hit_stop,hit_target = now_price<=rec["stop_loss"], now_price>=rec["profit_target"]
        else: hit_stop,hit_target = now_price>=rec["stop_loss"], now_price<=rec["profit_target"]
        if hit_stop or hit_target:
            rec["status"]="Stop" if hit_stop else "Target"; rec["exit_date"]=today; rec["exit_price"]=round(now_price,2); changed=True
    if changed: save_predictions(preds)

def _alpaca_download_cached(ticker:str,timeframe:TimeFrame,start_iso:str,end_iso:str,cache_dir:str=".ohlcv_cache")->pd.DataFrame:
    Path(cache_dir).mkdir(exist_ok=True)
    fname=Path(cache_dir)/f"{ticker}_alpaca_{timeframe.value}_{start_iso.split('T')[0]}_{end_iso.split('T')[0]}.pkl"
    max_age_seconds = (6*24*3600) if timeframe==TimeFrame.Day else (4*3600)
    if fname.exists():
        try:
            cached_df:pd.DataFrame=joblib.load(fname)
            if not cached_df.empty:
                last_ts_utc=(pd.Timestamp(cached_df.index[-1],tz='America/New_York') if cached_df.index[-1].tzinfo is None else cached_df.index[-1]).tz_convert('UTC')
                if (pd.Timestamp.utcnow()-last_ts_utc).total_seconds()<max_age_seconds: return cached_df.copy()
        except Exception: fname.unlink(missing_ok=True)
    if ALPACA_API_CLIENT is None: print(Fore.RED+f"Alpaca client NA for {ticker}."+Style.RESET_ALL); return pd.DataFrame()
    try:
        bars_df=ALPACA_API_CLIENT.get_bars(symbol_or_symbols=ticker,timeframe=timeframe,start=start_iso,end=end_iso,adjustment='split').df
        if isinstance(bars_df.index,pd.MultiIndex) and 'symbol' in bars_df.index.names:
            bars_df = bars_df.loc[ticker] if ticker in bars_df.index.get_level_values('symbol') else pd.DataFrame()
        if bars_df.empty: return pd.DataFrame()
        bars_df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
        for col in ['Open','High','Low','Close','Volume']:
            if col not in bars_df: bars_df[col]=np.nan
        bars_df.index=(bars_df.index.tz_localize('UTC') if bars_df.index.tz is None else bars_df.index).tz_convert('America/New_York').tz_localize(None)
        bars_df=bars_df[['Open','High','Low','Close','Volume']].sort_index()
        joblib.dump(bars_df,fname); return bars_df.copy()
    except Exception as e: print(Fore.RED+f"Alpaca fetch error {ticker} {timeframe.value}: {e}"+Style.RESET_ALL); return pd.DataFrame()

def preload_alpaca_interval_cache(symbols:list[str],period:str,interval:str,cache_dir:str=".ohlcv_cache")->None:
    symbols=sorted(set(symbols))
    if not symbols or ALPACA_API_CLIENT is None: return
    alpaca_tf=ALPACA_TIMEFRAME_MAP.get(interval)
    if not alpaca_tf: print(Fore.YELLOW+f"Unsupported interval {interval} for preload."+Style.RESET_ALL); return
    start_dt,end_dt=get_start_end_dates_for_period(period,datetime.datetime.now(datetime.timezone.utc))
    start_iso,end_iso=(start_dt.replace(tzinfo=datetime.timezone.utc) if start_dt.tzinfo is None else start_dt).isoformat(), (end_dt.replace(tzinfo=datetime.timezone.utc) if end_dt.tzinfo is None else end_dt).isoformat()
    for sym in symbols: _alpaca_download_cached(sym,alpaca_tf,start_iso,end_iso,cache_dir=cache_dir)

def fetch_data(ticker,start=None,end=None,intervals=None,warmup_days=300):
    if intervals is None:
        intervals={
            '5m':('14d','5m'), '30m':('60d','30m'), '1h':('120d','1h'),
            '2h':('120d','2h'), # Changed to fetch 2h data directly
            '1d':('380d','1d')
        }
    dfs,eval_time_utc={},datetime.datetime.now(datetime.timezone.utc)
    for key,(period_str,interval_str) in intervals.items():
        alpaca_tf_for_fetch = ALPACA_TIMEFRAME_MAP.get(interval_str) # Directly use interval_str from map

        if not alpaca_tf_for_fetch:
            print(Fore.YELLOW+f"Unsupported interval {interval_str} for {key}."+Style.RESET_ALL); dfs[key]=pd.DataFrame(); continue

        current_start_dt,current_end_dt=(pd.to_datetime(start)-timedelta(days=warmup_days), pd.to_datetime(end)) if key=='1d' and start and end else get_start_end_dates_for_period(period_str,eval_time_utc)
        start_iso,end_iso=(current_start_dt.replace(tzinfo=datetime.timezone.utc) if current_start_dt.tzinfo is None else current_start_dt).isoformat(), (current_end_dt.replace(tzinfo=datetime.timezone.utc) if current_end_dt.tzinfo is None else current_end_dt).isoformat()

        df=_alpaca_download_cached(ticker,alpaca_tf_for_fetch,start_iso,end_iso)
        dfs[key]=df # Assign directly, no special resampling for '2h' here

    return (dfs.get(k,pd.DataFrame()) for k in ['5m','30m','1h','2h','1d'])

def compute_indicators(df:pd.DataFrame,timeframe:str="daily")->pd.DataFrame:
    df=df.copy(); req={"Open","High","Low","Close","Volume"}
    if df.empty or not req.issubset(df.columns): return df
    w=14; rsi,adx,stoch,cci,macd,bb,atr=RSIIndicator(df["Close"],w).rsi(),ADXIndicator(df["High"],df["Low"],df["Close"],w),StochasticOscillator(df["High"],df["Low"],df["Close"],w,3),CCIIndicator(df["High"],df["Low"],df["Close"],20,0.015),MACD(df["Close"],26,12,9),BollingerBands(df["Close"],20,2),AverageTrueRange(df["High"],df["Low"],df["Close"],w)
    df[f"RSI_{timeframe}"]=rsi; df[f"ADX_{timeframe}"]=adx.adx(); df[f"ADX_pos_{timeframe}"]=adx.adx_pos(); df[f"ADX_neg_{timeframe}"]=adx.adx_neg(); df[f"STOCHk_{timeframe}"]=stoch.stoch(); df[f"STOCHd_{timeframe}"]=stoch.stoch_signal(); df[f"CCI_{timeframe}"]=cci.cci(); df[f"MACD_{timeframe}"]=macd.macd(); df[f"MACD_signal_{timeframe}"]=macd.macd_signal(); df[f"MACD_hist_{timeframe}"]=macd.macd_diff(); df[f"BB_upper_{timeframe}"]=bb.bollinger_hband(); df[f"BB_lower_{timeframe}"]=bb.bollinger_lband(); df[f"BB_middle_{timeframe}"]=bb.bollinger_mavg(); df[f"ATR_{timeframe}"]=atr.average_true_range()
    if timeframe in {"daily","hourly"}: df[f"EMA50_{timeframe}"]=EMAIndicator(df["Close"],50).ema_indicator(); df[f"EMA200_{timeframe}"]=EMAIndicator(df["Close"],200).ema_indicator()
    if timeframe=="daily":
        def _h(c,n): wmh,wmr=WMAIndicator(c,n//2).wma(),WMAIndicator(c,int(np.sqrt(n))).wma(); return 2*wmh-wmr
        hma=_h(df["Close"],20); df["HMA_daily"],df["HMA_slope_daily"],df["DI_diff_daily"]=hma,hma.diff(),df["ADX_pos_daily"]-df["ADX_neg_daily"]; df["PrevHigh_daily"],df["PrevLow_daily"]=df["High"].shift(1),df["Low"].shift(1)
        mg,hg,dg,ao,vo,hl,sl,uo,lo=df["MACD_hist_daily"]>0,df["HMA_slope_daily"]>0,df["DI_diff_daily"]>0,df["ATR_daily"]>0.50,df["Volume"]>df["Volume"].shift(1)*0.8,df["Close"]>df["Open"],df["Close"]<df["Open"],(df["High"]-df["Close"])/(df["High"]-df["Low"]+1e-9)<0.2,(df["Close"]-df["Low"])/(df["High"]-df["Low"]+1e-9)<0.2
        df["Sequoia_long"]=(mg&hg&dg&hl&uo&vo&ao); df["Sequoia_short"]=(~mg&~hg&~dg&sl&lo&vo&ao)
    return df

def compute_anchored_vwap(df:pd.DataFrame,lookback_bars:int=252)->pd.Series:
    if df.empty or not {"Close","Low","Volume"}.issubset(df.columns): return pd.Series(dtype=float,index=df.index)
    anchor=df.tail(lookback_bars)['Low'].idxmin()
    if pd.isna(anchor): return pd.Series(np.nan,index=df.index)
    after,cum_vol,cum_dollars=df.loc[anchor:].copy(),df.loc[anchor:]['Volume'].cumsum(),(df.loc[anchor:]['Close']*df.loc[anchor:]['Volume']).cumsum()
    out=pd.Series(np.nan,index=df.index); out.loc[after.index]=cum_dollars/cum_vol; out.name='AnchoredVWAP'; return out

def to_daily(intra_df,label):
    if intra_df.empty: return pd.DataFrame()
    daily_data=intra_df.groupby(intra_df.index.date).tail(1); daily_data.index=pd.to_datetime(daily_data.index.date); daily_data.index.name='Date'; return daily_data

def prepare_features(df_5m:pd.DataFrame,df_30m:pd.DataFrame,df_1h:pd.DataFrame,df_2h:pd.DataFrame,df_1d:pd.DataFrame,*,horizon:int=10,drop_recent:bool=True)->pd.DataFrame:
    inds=[compute_indicators(df.copy(),lbl) for df,lbl in zip([df_5m,df_30m,df_1h,df_2h,df_1d],['5m','30m','hourly','2h','daily'])]
    inds[0]['AnchoredVWAP_5m']=compute_anchored_vwap(inds[0],2000)
    inds[1]['AnchoredVWAP_30m']=compute_anchored_vwap(inds[1],200)
    inds[2]['AnchoredVWAP_1h']=compute_anchored_vwap(inds[2],120)
    ind_1d=inds[4]; ind_1d['AnchoredVWAP']=compute_anchored_vwap(ind_1d,252)
    dailies=[to_daily(i,lbl) for i,lbl in zip(inds[:-1],['5m','30m','hourly','2h'])]
    features_df=ind_1d
    for daily_df, suffix in zip(dailies, ['_5m', '_30m', '_1h', '_2h']):
        features_df = features_df.join(daily_df, rsuffix=suffix)
    features_df.dropna(subset=['Close'],inplace=True)
    if features_df.empty: return features_df
    atr,closes,labels=features_df['ATR_daily'].fillna(0),features_df['Close'].values,np.ones(len(features_df),dtype=int)
    up,dn=features_df['Close']+2.0*atr, features_df['Close']-2.0*atr
    for i in range(len(features_df)-horizon):
        w,u_idx,d_idx=closes[i+1:i+1+horizon],np.where(closes[i+1:i+1+horizon]>=up.iloc[i])[0],np.where(closes[i+1:i+1+horizon]<=dn.iloc[i])[0]
        if u_idx.size and d_idx.size: labels[i]=2 if u_idx[0]<d_idx[0] else 0
        elif u_idx.size: labels[i]=2
        elif d_idx.size: labels[i]=0
    features_df['future_class']=labels
    if drop_recent: features_df=features_df.iloc[:-horizon]
    else: features_df.loc[features_df.index[-horizon:],'future_class']=np.nan
    return features_df

def refine_features(df,cut=0.0001,corr=0.9):
    if df.empty or 'future_class' not in df.columns: return df
    y,X=df['future_class'],df.drop(columns=['future_class']).copy().ffill().bfill()
    m=XGBClassifier(objective='multi:softmax',num_class=3,use_label_encoder=False,eval_metric='mlogloss',verbosity=0,tree_method='hist',device='cuda')
    s=int(0.8*len(X)); Xt,Xv,yt,yv=X.iloc[:s],X.iloc[s:],y.iloc[:s],y.iloc[s:]
    if len(np.unique(yt))<2: return df
    m.fit(Xt,yt); imp=pd.Series(m.feature_importances_,index=X.columns).sort_values(ascending=False)
    X.drop(columns=imp[imp<cut].index.tolist(),inplace=True,errors='ignore')
    cm=X.select_dtypes(include=[np.number]).corr(); up=cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool))
    X.drop(columns=[c for c in up.columns if any(up[c].abs()>corr)],inplace=True,errors='ignore'); return X.join(y)

def tune_threshold_and_train(df):
    if df.empty or 'future_class' not in df.columns: return None,None
    Xf,yf=df.drop(columns=['future_class']).copy().ffill().bfill(),df['future_class']
    s=int(len(df)*0.8); Xt,yt=Xf.iloc[:s],yf.iloc[:s]
    td=pd.concat([Xt,yt],axis=1); cs=[td[td['future_class']==i] for i in range(3)]
    mc=max(len(c) for c in cs if c is not None and not c.empty); ts=pd.concat([resample(c,replace=True,n_samples=mc,random_state=42) for c in cs if c is not None and not c.empty],axis=0)
    Xt,yt=ts.drop(columns=['future_class']),ts['future_class']
    bt,bs=None,-np.inf
    for thr in [0.01,0.02,0.03,0.04,0.05]:
        if 'Close' not in df.columns: continue
        frt,frv=(df['Close'][:s].shift(-5)/df['Close'][:s]-1.0),(df['Close'][s:].shift(-5)/df['Close'][s:]-1.0)
        ytt,ytv=yt.copy(),yf.iloc[s:].copy(); ytt[:]=1;ytt[frt>thr]=2;ytt[frt<-thr]=0; ytv[:]=1;ytv[frv>thr]=2;ytv[frv<-thr]=0
        if len(np.unique(ytt))<3: continue
        m=XGBClassifier(objective='multi:softprob',num_class=3,use_label_encoder=False,eval_metric='mlogloss',verbosity=0,tree_method='hist',device='cuda',max_depth=5,min_child_weight=10,gamma=1.0,subsample=0.8,colsample_bytree=0.8,learning_rate=0.05,n_estimators=500)
        m.fit(Xt,ytt); yp=m.predict(Xf.iloc[s:])
        r=classification_report(ytv,yp,output_dict=True,zero_division=0); af1=(r.get('2',{}).get('f1-score',0.0)+r.get('0',{}).get('f1-score',0.0))/2.0
        if af1>bs: bs,bt=af1,thr
    if bt is None: return None,None
    frall=(df['Close'].shift(-5)/df['Close']-1.0); fy=yf.copy(); fy[:]=1;fy[frall>bt]=2;fy[frall<-bt]=0
    if len(np.unique(fy))<3: return None,bt
    fm=XGBClassifier(objective='multi:softprob',num_class=3,use_label_encoder=False,eval_metric='mlogloss',verbosity=0,tree_method='hist',device='cuda'); fm.fit(Xf,fy); return fm,bt

def generate_signal_output(t,lr,m,thr):
    p=m.predict_proba(lr.to_frame().T.values)[0]; ci,ps=int(np.argmax(p)),p[int(np.argmax(p))]
    if ps<0.60 or ci==1: return None
    d="LONG" if ci==2 else "SHORT"
    if (d=="LONG" and not lr.get("Sequoia_long",False)) or (d=="SHORT" and not lr.get("Sequoia_short",False)): return None
    pr,pl,ph=float(lr["Close"]),float(lr.get("PrevLow_daily",np.nan)),float(lr.get("PrevHigh_daily",np.nan))
    if d=="LONG": st,ri,mt,ta=(pl*0.995 if not np.isnan(pl) else pr*0.94),0.0,pr*1.10,0.0; ri=pr-st; ta=pr+max(2*ri,mt-pr)
    else: st,ri,mt,ta=(ph*1.005 if not np.isnan(ph) else pr*1.06),0.0,pr*0.90,0.0; ri=st-pr; ta=pr-max(2*ri,pr-mt)
    cl=Fore.GREEN if d=="LONG" else Fore.RED; return (f"{Fore.CYAN}{t}{Style.RESET_ALL}: {cl}{d}{Style.RESET_ALL} @ ${pr:.2f} Stop ${st:.2f} Target ${ta:.2f} P={ps:.2f}")

def _build_entry(r,d):
    pr=float(r["Close"])
    if d=="LONG": pl,st,ri,ta,tr=(float(r.get("PrevLow_daily",np.nan))),0.0,0.0,pr*1.10,0.0; st=(pl*0.995 if not np.isnan(pl) else pr*0.94); ri=pr-st; tr=pr+max(2*ri,ta-pr); return st,tr
    else: ph,st,ri,ta,tr=(float(r.get("PrevHigh_daily",np.nan))),0.0,0.0,pr*0.90,0.0; st=(ph*1.005 if not np.isnan(ph) else pr*1.06); ri=st-pr; tr=pr-max(2*ri,pr-ta); return st,tr

def _summarise_performance(trades_df, total_days):
    if trades_df.empty: return {"total":0,"win_rate":0,"avg_pnl":0,"sharpe":0,"max_dd":0,"tim":0}
    summary = {"total":len(trades_df),"win_rate":np.mean(trades_df.pnl_pct > 0) if not trades_df.empty else 0,"avg_pnl":np.mean(trades_df.pnl_pct) if not trades_df.empty else 0,"sharpe":0,"max_dd":0,"tim":0}
    return summary

def backtest_strategy(t,sd,ed,lf=None):
    d5,d30,d1h,d90,d1d=fetch_data(t,start=sd,end=ed); ft=prepare_features(d5,d30,d1h,d90,d1d); ft=refine_features(ft); m,thr=tune_threshold_and_train(ft)
    if m is None: print(Fore.YELLOW+f"Model fail {t}."+Style.RESET_ALL); return
    Xa,prds,pbs=ft.drop(columns=['future_class']).ffill().bfill(),None,None; prds,pbs=m.predict(Xa),m.predict_proba(Xa)
    dp=ft.copy(); dp['prediction']=prds; trds,ip,di,ep,sp,tp,ets=[],False,None,None,None,None,None
    for i,r in dp.iterrows():
        prc=r["Close"]
        if ip:
            hs,ht=(prc<=sp) if di=="LONG" else (prc>=sp),(prc>=tp) if di=="LONG" else (prc<=tp)
            if hs or ht: pnl=(prc-ep)/ep if di=="LONG" else (ep-prc)/ep; trds.append({"entry_timestamp":ets,"exit_timestamp":i,"direction":di,"entry_price":round(ep,2),"exit_price":round(prc,2),"pnl_pct":round(pnl,4),"stop_price":round(sp,2),"target_price":round(tp,2)}); ip=False
            if ip: continue
        cl,pb=int(r["prediction"]),pbs[dp.index.get_loc(i)][int(r["prediction"])]
        if pb<0.60: continue
        if cl==2 and r.get("Sequoia_long",False): di="LONG"
        elif cl==0 and r.get("Sequoia_short",False): di="SHORT"
        else: continue
        ep,sp,tp,ets,ip=prc,*_build_entry(r,di),i,True
    if not trds: print(Fore.YELLOW+f"No trades {t}."+Style.RESET_ALL); return
    tdf,smry=pd.DataFrame(trds),_summarise_performance(pd.DataFrame(trds),len(ft))
    p=lambda x:f"{x*100:.2f}%"; print(Fore.BLUE+f"\nBacktest {t}"+Style.RESET_ALL+f" ({sd}→{ed}):"); print(f" Trades:{Fore.CYAN}{smry['total']}{Style.RESET_ALL} | WR:{Fore.CYAN}{p(smry['win_rate'])}{Style.RESET_ALL} | AvgP/L:{Fore.CYAN}{p(smry['avg_pnl'])}{Style.RESET_ALL} | Shp:{Fore.CYAN}{smry['sharpe']:.2f}{Style.RESET_ALL} | MDD:{Fore.CYAN}{p(smry['max_dd'])}{Style.RESET_ALL} | TIM:{Fore.CYAN}{p(smry['tim'])}{Style.RESET_ALL}")

def prepare_features_intraday(d30:pd.DataFrame|None=None)->pd.DataFrame:
    d30=d30.copy(); d30=compute_indicators(d30,'intraday'); d30['AnchoredVWAP_30m']=compute_anchored_vwap(d30,200)
    if d30.empty or 'Close' not in d30.columns: return pd.DataFrame()
    dly=to_daily(d30,"intraday"); dly=compute_indicators(dly,'daily'); dly['AnchoredVWAP']=compute_anchored_vwap(dly,252)
    if dly.index.tz is not None: dly.index=dly.index.tz_localize(None)
    if d30.index.tz is not None: d30.index=d30.index.tz_localize(None)
    d30=d30.join(dly.reindex(d30.index,method='ffill'),rsuffix='_daily'); hb,ac=16,'ATR_intraday'
    if ac not in d30.columns: d30[ac]=d30['Close'].rolling(14).std().fillna(0)
    u,d,cs,lb=d30['Close']+2*d30[ac],d30['Close']-2*d30[ac],d30['Close'].values,np.ones(len(d30),int)
    for i in range(len(d30)-hb):
        w,a,b=cs[i+1:i+1+hb],np.where(cs[i+1:i+1+hb]>=u.iloc[i])[0],np.where(cs[i+1:i+1+hb]<=d.iloc[i])[0]
        if a.size and b.size: lb[i]=2 if a[0]<b[0] else 0
        elif a.size: lb[i]=2
        elif b.size: lb[i]=0
    d30['future_class']=lb; d30=d30.iloc[:-hb]; return d30

def backtest_strategy_intraday(t,sd,ed,lf=None):
    _,d30,_,_,_=fetch_data(t,start=sd,end=ed); ft=prepare_features_intraday(d30); ft=refine_features(ft); m,thr=tune_threshold_and_train(ft)
    if m is None: print(Fore.YELLOW+f"Model fail {t}."+Style.RESET_ALL); return
    Xa=ft.drop(columns=['future_class']).ffill().bfill(); prds,pbs=m.predict(Xa),m.predict_proba(Xa)
    dp=ft.copy(); dp['prediction']=prds; trds,ip,di,ep,sp,tp,ets=[],False,None,None,None,None,None
    for i,r in dp.iterrows():
        prc=r["Close"]
        if ip:
            hs,ht=(prc<=sp) if di=="LONG" else (prc>=sp),(prc>=tp) if di=="LONG" else (prc<=tp)
            if hs or ht: pnl=(prc-ep)/ep if di=="LONG" else (ep-prc)/ep; trds.append({"entry_timestamp":ets,"exit_timestamp":i,"direction":di,"entry_price":round(ep,2),"exit_price":round(prc,2),"pnl_pct":round(pnl,4),"stop_price":round(sp,2),"target_price":round(tp,2)}); ip=False
            if ip: continue
        cl,pb=int(r["prediction"]),pbs[dp.index.get_loc(i)][int(r["prediction"])]
        if pb<0.60: continue
        if cl==2 and r.get("Sequoia_long",False): di="LONG"
        elif cl==0 and r.get("Sequoia_short",False): di="SHORT"
        else: continue
        ep,sp,tp,ets,ip=prc,*_build_entry(r,di),i,True
    if not trds: print(Fore.YELLOW+f"No trades {t} intraday."+Style.RESET_ALL); return
    tdf,smry=pd.DataFrame(trds),_summarise_performance(pd.DataFrame(trds),len(ft))
    p=lambda x:f"{x*100:.2f}%"; print(Fore.BLUE+f"\nBacktest {t}"+Style.RESET_ALL+f" ({sd}→{ed}):"); print(f" Trades:{Fore.CYAN}{smry['total']}{Style.RESET_ALL} | WR:{Fore.CYAN}{p(smry['win_rate'])}{Style.RESET_ALL} | AvgP/L:{Fore.CYAN}{p(smry['avg_pnl'])}{Style.RESET_ALL} | Shp:{Fore.CYAN}{smry['sharpe']:.2f}{Style.RESET_ALL} | MDD:{Fore.CYAN}{p(smry['max_dd'])}{Style.RESET_ALL} | TIM:{Fore.CYAN}{p(smry['tim'])}{Style.RESET_ALL}")

def run_signals_on_watchlists(use_intraday:bool=True):
    all_syms=load_watchlist("long")+load_watchlist("short"); preload_alpaca_interval_cache(all_syms,"60d","30m"); preload_alpaca_interval_cache(all_syms,"380d","1d")
    _,_,_,_=load_config()
    for side in ("long","short"):
        tickers=load_watchlist(side)
        if not tickers: continue
        print(f"\n========== {side.upper()} WATCH-LIST ==========")
        want_dir="LONG" if side=="long" else "SHORT"
        for ticker in tickers:
            try: d5,d30,d1h,d90,d1d=get_or_fetch(ticker)
            except Exception as e: print(f"{ticker}: data error → {e}"); continue
            feats=(prepare_features_intraday(d30) if use_intraday else prepare_features(d5,d30,d1h,d90,d1d))
            feats=refine_features(feats)
            if feats.empty or 'future_class' not in feats.columns: print(f"{ticker}: no usable rows."); continue
            model,thr=tune_threshold_and_train(feats)
            if model is None: print(f"{ticker}: model training failed."); continue
            latest=feats.drop(columns='future_class').iloc[-1]; sig=generate_signal_output(ticker,latest,model,thr)
            if sig and (want_dir in sig): print(sig)

def show_signals_since_start_of_week()->None:
    all_syms=load_watchlist("long")+load_watchlist("short")
    for p,i in [("14d","5m"),("60d","30m"),("120d","1h"),("60d","2h"),("380d","1d")]: preload_alpaca_interval_cache(all_syms,p,i) # Changed 2h to 2h
    today,monday=datetime.datetime.today().replace(hour=0,minute=0,second=0,microsecond=0),datetime.datetime.today()-datetime.timedelta(days=datetime.datetime.today().weekday())
    start_s,end_s=(monday-datetime.timedelta(days=180)).strftime("%Y-%m-%d"),today.strftime("%Y-%m-%d")
    cache,open_by_symbol=load_predictions(),{p["symbol"]:p for p in load_predictions() if p["status"]=="Open"}
    for side in ("long","short"):
        tickers=load_watchlist(side)
        if not tickers: continue
        print(f"\n========== {side.upper()} WEEKLY SIGNALS ==========")
        want_dir="LONG" if side=="long" else "SHORT"
        for ticker in tickers:
            print(f"\n--- {ticker} ---")
            try: d5,d30,d1h,d90,d1d=get_or_fetch(ticker) # d90 will now be 2h data
            except Exception as e: print(f"Fetch error {e}"); continue
            if d1d.empty: print(f"No daily data for {ticker}."); continue
            d1d_s=d1d.loc[start_s:end_s] if not d1d.loc[start_s:end_s].empty else d1d
            feats_all=prepare_features(d5,d30,d1h,d90,d1d_s,drop_recent=False) # d90 is 2h
            if feats_all.empty: print("No data for features."); continue
            train_df=refine_features(feats_all.dropna(subset=["future_class"]))
            if train_df.empty: print("No labels for training."); continue
            model,thr=tune_threshold_and_train(train_df)
            if model is None: print("Model training failed."); continue
            feat_cols=[c for c in train_df.columns if c!="future_class"]
            wf_full=feats_all.loc[monday:today]
            if wf_full.empty: print(f"No feature data for {ticker} in current week."); continue
            wf=wf_full[feat_cols].ffill().bfill()
            if wf.empty: print(f"No feature data for {ticker} after fill in current week."); continue
            op_dir=open_by_symbol.get(ticker,{}).get("direction")
            for dt,r in wf.iterrows():
                sig=generate_signal_output(ticker,r,model,thr)
                if not sig or (want_dir not in sig): continue
                if op_dir==want_dir: break
                print(f"{dt.date()}: {sig}")
                prc,st,tg=float(r["Close"]),*_build_entry(r,want_dir)
                nr={"symbol":ticker,"entry_date":dt.strftime("%Y-%m-%d"),"entry_price":round(prc,2),"direction":want_dir,"stop_loss":round(st,2),"profit_target":round(tg,2),"status":"Open"}
                if op_dir: old=open_by_symbol[ticker]; old.update({"status":"Closed","exit_date":dt.strftime("%Y-%m-%d"),"exit_price":prc})
                open_by_symbol[ticker],op_dir=nr,want_dir; break
    save_predictions([p for p in cache if p["status"]!="Open"]+list(open_by_symbol.values()))

def signals_performance_cli():
    if ALPACA_API_CLIENT is None:
        print(Fore.RED + "Alpaca API client not initialized. Cannot show signals performance." + Style.RESET_ALL)
        input("\nPress Enter to return …"); return

    all_recs = load_predictions(); open_tr = [p for p in all_recs if p['status'] == 'Open']
    if not open_tr: print("No open positions."); input("\nPress Enter..."); return

    palette=[('title','white,bold',''),('headers','light blue,bold',''),('positive','dark green',''),('negative','dark red',''),('hit','white','dark cyan'),('footer','white,bold','')]
    header=urwid.AttrMap(urwid.Text(" Weekly Signals – Open Trades",align='center'),'title')
    footer=urwid.AttrMap(urwid.Text(" (R)efresh (D)eject (Q)uit"),'footer')
    txt,lay=urwid.Text(""),urwid.Frame(header=header,body=urwid.AttrMap(urwid.Filler(txt,'top'),'body'),footer=footer)

    def get_prices():
        syms = sorted({p['symbol'] for p in open_tr})
        prices = {}
        if not syms: return prices
        for sym in syms:
            try:
                latest_trade = ALPACA_API_CLIENT.get_latest_trade(sym)
                prices[sym] = latest_trade.price
            except Exception as e:
                entry_price_fallback = next((p_item['entry_price'] for p_item in open_tr if p_item['symbol'] == sym), None)
                print(Fore.YELLOW + f"Could not fetch latest price for {sym} (dashboard): {e}. Using entry price." + Style.RESET_ALL)
                prices[sym] = entry_price_fallback if entry_price_fallback is not None else 0
        return prices

    rec_cache=[]
    def refresh(*_):
        nonlocal rec_cache
        prices = get_prices()
        rows=[('headers',f"{'Symbol':8}{'Dir':6}{'Entry':>10}{'Now':>10}{'P/L%':>8}{'Stop':>10}{'Target':>10}{'Status':>12}{'Date':>12}\n")]
        rec_info,today_str = [], datetime.datetime.today().strftime('%Y-%m-%d')
        for p_rec in open_tr:
            sym,now_prc=p_rec['symbol'],float(prices.get(p_rec['symbol'],p_rec['entry_price']))
            pnl=((now_prc-p_rec['entry_price'])/p_rec['entry_price']*100) if p_rec['direction']=='LONG' else ((p_rec['entry_price']-now_prc)/p_rec['entry_price']*100)
            stat,is_hit="Open",False
            if p_rec['direction']=='LONG':
                if now_prc<=p_rec['stop_loss']: stat,is_hit="Stop",True
                elif now_prc>=p_rec['profit_target']: stat,is_hit="Target",True
            else: # SHORT
                if now_prc>=p_rec['stop_loss']: stat,is_hit="Stop",True
                elif now_prc<=p_rec['profit_target']: stat,is_hit="Target",True
            attr='hit' if is_hit else ('positive' if pnl>=0 else 'negative')
            rows.append((attr,f"{sym:8}{p_rec['direction']:6}{p_rec['entry_price']:>10.2f}{now_prc:>10.2f}{pnl:>8.2f}%{p_rec['stop_loss']:>10.2f}{p_rec['profit_target']:>10.2f}{stat:>12}{p_rec['entry_date']:>12}\n"))
            rec_info.append((p_rec,is_hit,stat,now_prc,pnl,today_str))
        txt.set_text(rows); rec_cache=rec_info
    refresh()
    def unhandled(k):
        nonlocal open_tr,all_recs
        if k.lower()=='q': raise urwid.ExitMainLoop()
        if k.lower()=='r': refresh()
        if k.lower()=='d':
            changed=False
            for p_rec,is_hit,stat,now_prc,pnl,today_s in rec_cache:
                if is_hit: p_rec.update({'status':stat,'exit_price':round(now_prc,2),'exit_date':today_s,'pnl_pct':round(pnl,2)}); changed=True
            if changed:
                open_tr=[p_ for p_ in open_tr if p_['status']=='Open']
                all_recs=[p_ for p_ in all_recs if p_['status']!='Open']+open_tr
                save_predictions(all_recs)
            refresh()
    urwid.MainLoop(lay,palette,unhandled_input=unhandled).run()

def closed_stats_cli():
    recs=[p for p in load_predictions() if p['status']!='Open']
    if not recs: print("No closed trades."); input("\nPress Enter..."); return
    for r_item in recs:
        if 'pnl_pct' not in r_item or r_item['pnl_pct'] is None:
            ep,xp=r_item['entry_price'],r_item.get('exit_price',r_item['entry_price'])
            r_item['pnl_pct']=round(((xp-ep)/ep*100) if r_item['direction']=='LONG' else ((ep-xp)/ep*100),2)

    wins,losses,total=[r for r in recs if r['status']=='Target'],[r for r in recs if r['status'] in ('Stop','Closed')],len(recs)
    win_rt=(len(wins)/total*100 if total else 0)
    avg_win,avg_los=(np.mean([w['pnl_pct'] for w in wins]) if wins else 0.0),(np.mean([l['pnl_pct'] for l in losses]) if losses else 0.0)
    equity,tot_ret=1.0,0.0
    for r_item in recs: equity*= (1+r_item['pnl_pct']/100)
    tot_ret=(equity-1)*100

    g,r_color,b=lambda s:Fore.GREEN+s+Style.RESET_ALL,lambda s:Fore.RED+s+Style.RESET_ALL,lambda s:Fore.CYAN+s+Style.RESET_ALL

    print(b("\nClosed Stats")); print(b("----------"))
    print(f"Trades:{b(str(total))} | Wins:{g(str(len(wins)))}({g(f'{avg_win:+.2f}%')}) | Losses:{r_color(str(len(losses)))}({r_color(f'{avg_los:+.2f}%')})")
    print(f"Win Rate:{(g if win_rt>=50 else r_color)(f'{win_rt:.2f}%')} | Total Return:{(g if tot_ret>=0 else r_color)(f'{tot_ret:+.2f}%')}")

    print(b("\nRecent 10:"))
    for rcd in recs[-10:]:
        pnl_color = g if rcd['pnl_pct'] >= 0 else r_color
        print(f"{rcd['exit_date']} {rcd['symbol']:5} {rcd['direction']:5} {rcd['status']:6} PnL {pnl_color(f'{rcd["pnl_pct"]:+6.2f}%')}")

    if input(Fore.YELLOW+"\n(C)lear or Enter: "+Style.RESET_ALL).lower()=='c':
        save_predictions([p for p in load_predictions() if p['status']=='Open'])
        print(Fore.YELLOW+"History cleared."+Style.RESET_ALL)
    input("\nPress Enter...")


def run_signals_for_watchlist(side:str,use_intraday:bool=True):
    _,_,_,_=load_config(); tickers=load_watchlist(side)
    if not tickers: print(f"{side.capitalize()} watch-list empty."); return
    for ticker in tickers:
        print(f"\n=== {ticker} ({side}) ===")
        d5,d30,d1h,d90,d1d=get_or_fetch(ticker)
        feats=(prepare_features_intraday(d30) if use_intraday else prepare_features(d5,d30,d1h,d90,d1d))
        feats=refine_features(feats)
        if feats.empty or 'future_class' not in feats.columns: print("No valid data."); continue
        model,thr=tune_threshold_and_train(feats)
        if model is None: print("Model training failed."); continue
        latest=feats.drop(columns='future_class').iloc[-1]; sig=generate_signal_output(ticker,latest,model,thr)
        if sig and side.upper() in sig: print(sig)

def interactive_menu():
    if ALPACA_API_CLIENT is None: _,key,sec,url=load_config(); initialize_alpaca_client(key,sec,url)
    while True:
        print("\nMain Menu:\n1.Manage WL\n2.Run Signals\n3.Weekly Signals\n4.Backtest WLs\n5.Open Pos Perf\n6.Closed Stats\n0.Exit")
        ch=input("Select:").strip()
        if ch=='0': print("Exiting."); break
        elif ch=='1': manage_watchlist()
        elif ch=='2': run_signals_on_watchlists(True)
        elif ch=='3': show_signals_since_start_of_week()
        elif ch=='4': print("Backtest WLs (NIY)")
        elif ch=='5': _update_positions_status(); signals_performance_cli()
        elif ch=='6': closed_stats_cli()
        else: print("Invalid.")

def main():
    p=argparse.ArgumentParser(description="Signal generator/backtester")
    p.add_argument('tickers',nargs='*',help="Ticker symbols")
    p.add_argument('--log',action='store_true',help="Log trades")
    p.add_argument('--backtest',action='store_true',help="Backtest mode")
    p.add_argument('--start',default=None,help="Backtest start (YYYY-MM-DD)")
    p.add_argument('--end',default=None,help="Backtest end (YYYY-MM-DD)")
    p.add_argument('--real',action='store_true',help="30-min intraday backtest")
    p.add_argument('--live-real',action='store_true',help="30-min intraday live")
    a=p.parse_args()
    if not a.tickers and not any([a.log,a.backtest,a.start,a.end,a.real,a.live_real]): interactive_menu(); return
    _,key,sec,url=load_config(); initialize_alpaca_client(key,sec,url)
    ts,lt,rb,sa,ea,ui=a.tickers,a.log,a.backtest,a.start,a.end,a.real
    lf=open("trades_log.csv","a") if lt else None
    if rb and sa and ea:
        preload_alpaca_interval_cache(ts,"60d","30m"); preload_alpaca_interval_cache(ts,"380d","1d") # Preload for 30m and 1d
        # Also preload for 2h if it's going to be used by '2h' key via intervals in fetch_data
        preload_alpaca_interval_cache(ts,"120d","2h")


        for t in ts:
            fs=f"30m_{sa}_{ea}.csv" if ui else f"{sa}_{ea}.csv"; fn=f"{t}_{fs}"
            with open(fn,"a") as lfl: print(f"\n=== {'Intraday ' if ui else ''}Backtesting {t} {sa}→{ea} ==="); (backtest_strategy_intraday if ui else backtest_strategy)(t,sa,ea,log_file=lfl)
        if lf: lf.close() # Corrected indentation
        return
    uil=a.live_real
    for t in ts:
        print(f"\n=== Processing {t} (live{' intraday' if uil else ''}) ===")
        try:
            fetched_data = get_or_fetch(t)
            if isinstance(fetched_data, tuple) and len(fetched_data) == 5:
                d5,d30,d1h,d90,d1d = fetched_data
            else:
                print(f"Data fetch error for {t}: get_or_fetch did not return 5 DataFrames."); continue
        except Exception as e: print(f"Data fetch error {t}: {e}"); continue
        fts=None
        if uil:
            if not d30.empty: fts=prepare_features_intraday(d30)
            else: print(f"No 30m data {t} live intraday."); continue
        else:
            if not d1d.empty: fts=prepare_features(d5,d30,d1h,d90,d1d) # d90 is 2h data here
            else: print(f"No daily data {t} live daily."); continue
        if fts.empty: print(f"Insufficient data for features: {t}"); continue
        fts=refine_features(fts)
        if fts.empty or 'future_class' not in fts.columns: print(f"No features after refinement: {t}"); continue
        m,thr=tune_threshold_and_train(fts)
        if m is None: print(f"Model training failed: {t}."); continue
        lr=fts.drop(columns='future_class').iloc[-1]; sig=generate_signal_output(t,lr,m,thr)
        if sig: print(sig)
        else: print(f"No signal for {t}.")
        if lf and sig: lf.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{t},{sig}\n"); lf.flush()
    if lf: lf.close()

if __name__=="__main__":
    main()

[end of Sequoia/axist-tos.py]
