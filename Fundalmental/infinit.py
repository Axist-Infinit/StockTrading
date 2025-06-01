#!/usr/bin/env python3

import os
import sys
import math
import json
import time
import random
import datetime
import traceback
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Interactive
import inquirer
from sec_edgar_downloader import Downloader

# Stats / ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

# Optional LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# CLI-based UI
import urwid
import yfinance as yf

# For asynchronous FRED data retrieval
import asyncio
import aiohttp

print("*** Not real financial advice. Use responsibly. ***")

# --------------------------------------------------------------------------------------
#                                 CONFIG & FILENAMES
# --------------------------------------------------------------------------------------
CONFIG_FILE = "config.json"
ANALYSIS_CACHE_FILE = "analysis_cache.json"
PREDICTION_HISTORY_FILE = "predictions_history.json"

SECTOR_INDUSTRY_MAP = {
    "Utilities": [
        "Utilities Diversified",
        "Utilities Independent Power Producers",
        "Utilities Regulated Electric",
        "Utilities Regulated Gas",
        "Utilities Regulated Water",
        "Utilities Renewable"
    ],
    "Energy": [
        "Oil & Gas Drilling",
        "Oil & Gas E&P",
        "Oil & Gas Equipment & Services",
        "Oil & Gas Integrated",
        "Oil & Gas Midstream",
        "Oil & Gas Refining & Marketing",
        "Coking Coal",
        "Thermal Coal",
        "Uranium",
        "Solar"
    ],
    "Communication Services": [
        "Oil & Gas Drilling",
        "Oil & Gas E&P",
        "Oil & Gas Equipment & Services",
        "Oil & Gas Integrated",
        "Oil & Gas Midstream",
        "Oil & Gas Refining & Marketing",
        "Coking Coal",
        "Thermal Coal",
        "Uranium",
        "Solar"
    ],
    "Consumer Discretionary": [
        "Apparel Retail",
        "Auto & Truck Dealerships",
        "Auto Manufacturers",
        "Auto Parts",
        "Consumer Electronics",
        "Department Stores",
        "Discount Stores",
        "Electronic Gaming & Multimedia",
        "Entertainment",
        "Footwear & Accessories",
        "Home Improvement Retail",
        "Hotels, Resorts & Cruise Lines",
        "Internet Retail",
        "Leisure",
        "Lodging",
        "Luxury Goods",
        "Recreational Vehicles",
        "Resorts & Casinos",
        "Restaurants",
        "Specialty Retail",
        "Travel Services",
        "Gambling",
        "Personal Services"
    ],
    "Consumer Staples": [
        "Beverages - Brewers",
        "Beverages - Non-Alcoholic",
        "Beverages - Wineries & Distilleries",
        "Confectioners",
        "Food Distribution",
        "Grocery Stores",
        "Household & Personal Products",
        "Packaged Foods",
        "Tobacco"
    ],
    "Healthcare": [
        "Biotechnology",
        "Diagnostics & Research",
        "Drug Manufacturers - General",
        "Drug Manufacturers - Specialty & Generic",
        "Health Information Services",
        "Healthcare Plans",
        "Medical Care Facilities",
        "Medical Devices",
        "Medical Distribution",
        "Medical Instruments & Supplies",
        "Pharmaceutical Retailers"
    ],
    "Materials": [
        "Chemicals",
        "Copper",
        "Gold",
        "Other Industrial Metals & Mining",
        "Other Precious Metals & Mining",
        "Paper & Paper Products",
        "Silver",
        "Specialty Chemicals",
        "Steel"
    ],
    "Industrials": [
        "Aerospace & Defense",
        "Airlines",
        "Building Materials",
        "Building Products & Equipment",
        "Business Equipment & Supplies",
        "Conglomerates",
        "Electrical Equipment & Parts",
        "Engineering & Construction",
        "Farm & Heavy Construction Machinery",
        "Industrial Distribution",
        "Industrial Machinery",
        "Marine Shipping",
        "Metal Fabrication",
        "Pollution & Treatment Controls",
        "Railroads",
        "Rental & Leasing Services",
        "Specialty Industrial Machinery",
        "Staffing & Employment Services",
        "Textile Manufacturing",
        "Tools + Accessories",
        "Trucking",
        "Waste Management"
    ],
    "Financials": [
        "Asset Management",
        "Banks - Diversified",
        "Banks - Regional",
        "Capital Markets",
        "Closed-End Fund - Debt",
        "Closed-End Fund - Equity",
        "Credit Services",
        "Financial Conglomerates",
        "Financial Data & Stock Exchanges",
        "Insurance - Diversified",
        "Insurance - Life",
        "Insurance - Property & Casualty",
        "Insurance - Reinsurance",
        "Insurance - Specialty",
        "Insurance Brokers",
        "Mortgage Finance"
    ],
    "Information Technology": [
        "Communication Equipment",
        "Computer Hardware",
        "Consulting Services",
        "Electronics & Computer Distribution",
        "Information Technology Services",
        "Scientific & Technical Instruments",
        "Security & Protection Services",
        "Semiconductors",
        "Semiconductors Materials",
        "Software - Application",
        "Software - Infrastructure"
    ],
    "Real Estate": [
        "REIT - Diversified",
        "REIT - Healthcare Facilities",
        "REIT - Hotel & Motel",
        "REIT - Industrial",
        "REIT - Mortgage",
        "REIT - Office",
        "REIT - Residential",
        "REIT - Retail",
        "REIT - Specialty",
        "Real Estate - Development",
        "Real Estate - Diversified",
        "Real Estate Services"
    ]
}

CONFIG = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE,'r') as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {CONFIG_FILE}: {e}")
else:
    print(f"No {CONFIG_FILE} found. Some functionality may be limited...")

FRED_API_KEY = CONFIG.get("FRED_API_KEY", None)
ALPHA_VANTAGE_API_KEY = CONFIG.get("ALPHA_VANTAGE_API_KEY","YOUR_ALPHA_VANTAGE_KEY")
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'


# --------------------------------------------------------------------------------------
#                                LOAD/SAVE Cache
# --------------------------------------------------------------------------------------
def load_analysis_cache():
    if os.path.exists(ANALYSIS_CACHE_FILE):
        with open(ANALYSIS_CACHE_FILE,'r') as f:
            return json.load(f)
    return {}

def save_analysis_cache(cache):
    with open(ANALYSIS_CACHE_FILE,'w') as f:
        json.dump(cache,f, indent=4)


# --------------------------------------------------------------------------------------
#                           SEC Filings Download
# --------------------------------------------------------------------------------------
def download_sec_filings():
    forms_list= [
        '10-K','8-K','13F-HR','4','3','5','S-1',
        '424B2','424B3','424B4','424B5','424B7','424B8','POS AM','DEF 14A'
    ]
    qs= [
        inquirer.Text('ticker', message="Enter Ticker:"),
        inquirer.Checkbox('forms', message="Select forms:", choices=forms_list, default=['10-K','8-K']),
        inquirer.List('num_10Q', message="How many 10-Q?", choices=[str(i) for i in range(11)], default='2')
    ]
    ans= inquirer.prompt(qs)
    if not ans:
        print("Canceled.")
        return
    tkr= ans['ticker'].upper()
    sforms= ans['forms']
    nq= int(ans['num_10Q'])
    if not os.path.exists(tkr):
        os.makedirs(tkr)
    dl= Downloader(company_name="YourCompanyName", email_address="test@domain.com", download_folder=tkr)
    for fm in sforms:
        try:
            dl.get(fm,tkr, limit=100)
        except Exception as e:
            print(f"Error on {fm}: {e}")
    if nq>0:
        try:
            dl.get("10-Q",tkr,limit=nq)
        except Exception as e:
            print(f"Error on 10-Q: {e}")
    print("Downloaded SEC Filings.")


# --------------------------------------------------------------------------------------
#                FRED & MACRO DATA
# --------------------------------------------------------------------------------------
async def fetch_fred_series(session, series_id):
    if not FRED_API_KEY:
        return None
    url= "https://api.stlouisfed.org/fred/series/observations"
    params= {
        "api_key": FRED_API_KEY,
        "file_type":"json",
        "series_id": series_id
    }
    try:
        async with session.get(url,params=params, timeout=10) as resp:
            data= await resp.json()
            obs= data.get("observations",[])
            if not obs:
                return None
            vals=[]
            for o in obs:
                dt_str= o['date']
                val_str= o['value']
                try:
                    val= float(val_str)
                    dt= pd.to_datetime(dt_str)
                    vals.append((dt,val))
                except:
                    pass
            if not vals:
                return None
            vals.sort(key=lambda x:x[0])
            return vals[-1][1]
    except:
        return None

async def fetch_fred_data_concurrent():
    if not FRED_API_KEY:
        return {}
    results={}
    series_map= {
        "10Y":"DGS10",
        "2Y":"DGS2",
        "FEDFUNDS":"FEDFUNDS"
    }
    async with aiohttp.ClientSession() as session:
        tasks=[]
        for tag, sid in series_map.items():
            tasks.append((tag, fetch_fred_series(session,sid)))
        coros= [t[1] for t in tasks]
        fetched= await asyncio.gather(*coros, return_exceptions=True)
        for (tag,_), val in zip(tasks,fetched):
            if isinstance(val,Exception):
                results[tag]= None
            else:
                results[tag]= val
    return results

def get_macro_data():
    return asyncio.run(fetch_fred_data_concurrent())

def compute_macro_discount_rate(macro_data):
    if not macro_data:
        return 0.08
    fed= macro_data.get("FEDFUNDS",None)
    ten= macro_data.get("10Y",None)
    two= macro_data.get("2Y",None)
    base=2.0
    if fed:
        base= fed
    elif two:
        base= two
    spread=0.0
    if two and ten:
        spread= ten - two
    if spread<0:
        rp=5.0
    else:
        rp=3.0
    disc= (base+ rp)/100.0
    return disc

def macro_adjust_final_score(final_score, macro_data):
    fed_rate = macro_data.get('FEDFUNDS', 2.0)
    if fed_rate > 4.0:
        # If rates are high, discount final score a bit
        return final_score - 2
    return final_score

# --------------------------------------------------------------------------------------
#    SURVIVORSHIP-BIAS-FREE FUNDAMENTALS (Placeholder)
# --------------------------------------------------------------------------------------
def fetch_survivorship_bias_free_fundamentals(ticker, as_of_date=None):
    """
    Advanced version: truly SBF approach.
    Query a local DB or pro data feed for fundamentals 'as_of_date' so we don't
    accidentally see updated statements or restated data from the future.

    Example pseudo-logic below. Adjust to match your real DB schema, date columns, etc.
    """

    if as_of_date is None:
        # default to 'today' or some recent date
        as_of_date = datetime.datetime.now().date()

    # 1) Connect to your local SBF database (SQL, CSV, or external feed)
    #    We'll assume you have a table: point_in_time_fundamentals
    #    with columns: ticker, date, yoy_rev, yoy_ni, free_cf,
    #                 enterprise_value, operating_income, ebitda, forward_pe, etc.

    try:
        # Pseudocode: you'd do something like:
        # sbf_data = db.execute("""
        #    SELECT *
        #    FROM point_in_time_fundamentals
        #    WHERE ticker=? AND date<=?
        #    ORDER BY date DESC
        #    LIMIT 1
        # """, (ticker, as_of_date)).fetchone()

        # For demonstration, let's show a dictionary that might have been returned:
        sbf_data = {
            'yoy_rev': 0.05,  # +5% yoy
            'yoy_ni': 0.08,   # +8% yoy
            'free_cf': 1.2e9, # 1.2B
            'enterprise_value': 45e9,
            'operating_income': 2.5e9,
            'ebitda': 3.1e9,
            'forward_pe': 18.0
        }
        # In reality, you'd parse sbf_data from your DB row, or handle if None returned.

        # Now just return it in the same structure your advanced code expects:
        return {
            'yoy_rev': sbf_data['yoy_rev'],
            'yoy_ni': sbf_data['yoy_ni'],
            'free_cf': sbf_data['free_cf'],
            'enterprise_value': sbf_data['enterprise_value'],
            'operating_income': sbf_data['operating_income'],
            'ebitda': sbf_data['ebitda'],
            'forward_pe': sbf_data.get('forward_pe', None)
        }

    except Exception as e:
        print(f"[SBF ERROR] Could not fetch advanced fundamentals for {ticker}: {e}")
        return None


# --------------------------------------------------------------------------------------
#                ADVANCED FUNDAMENTALS + ML (with Hyperparam Tuning)
# --------------------------------------------------------------------------------------
def gather_historical_fundamentals_SBF(tickers, start='2018-01-01', end=None):
    """
    Advanced version that truly does a date-by-date approach for each ticker.
    For each month (or quarter) between 'start' and 'end', we fetch fundamentals
    from our SBF feed as_of_date, then label the performance over the next N days.

    Returns a real time-series classification DataFrame containing columns like:
    [ticker, date, yoy_rev, yoy_ni, free_cf, enterprise_value, operating_income,
     ebitda, forward_pe, ev_ebit, ev_ebitda, label]
    """
    if not end:
        end = datetime.datetime.now().strftime('%Y-%m-%d')

    # Convert string to datetime
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    data_rows = []

    # Step monthly (you can step quarterly if desired)
    current_date = start_dt
    while current_date <= end_dt:
        for tkr in tickers:
            # 1) Fetch point-in-time fundamentals as_of_date
            funda = fetch_survivorship_bias_free_fundamentals(tkr, as_of_date=current_date.date())
            if not funda:
                continue

            # 2) Label example: compare 30-day performance vs SPY
            #    If ticker outperforms SPY by >5% => label=2, >-5% => label=1, else 0
            try:
                day_30 = current_date + pd.Timedelta(days=30)
                tkr_hist = yf.Ticker(tkr).history(start=current_date, end=day_30)
                if len(tkr_hist) < 2:
                    continue
                ret_t = (tkr_hist['Close'].iloc[-1] / tkr_hist['Close'].iloc[0]) - 1

                spy_hist = yf.Ticker("SPY").history(start=current_date, end=day_30)
                if len(spy_hist) < 2:
                    continue
                ret_spy = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1

                perf_diff = ret_t - ret_spy
                if perf_diff > 0.05:
                    label = 2
                elif perf_diff > -0.05:
                    label = 1
                else:
                    label = 0

                # ---------------------------
                # NEW: calculate ev_ebit, ev_ebitda
                ev_ebit = None
                if (funda["enterprise_value"] and funda["operating_income"] 
                        and funda["operating_income"] > 0):
                    ev_ebit = funda["enterprise_value"] / funda["operating_income"]

                ev_ebitda = None
                if (funda["enterprise_value"] and funda["ebitda"] 
                        and funda["ebitda"] > 0):
                    ev_ebitda = funda["enterprise_value"] / funda["ebitda"]
                # ---------------------------

                row = {
                    'ticker': tkr,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'yoy_rev': funda['yoy_rev'],
                    'yoy_ni': funda['yoy_ni'],
                    'free_cf': funda['free_cf'],
                    'enterprise_value': funda['enterprise_value'],
                    'operating_income': funda['operating_income'],
                    'ebitda': funda['ebitda'],
                    'forward_pe': funda['forward_pe'],

                    # Add the two new columns below:
                    'ev_ebit': ev_ebit,
                    'ev_ebitda': ev_ebitda,

                    'label': label
                }
                data_rows.append(row)

            except Exception as e:
                print(f"[SBF WARN] {tkr} on {current_date} => {e}")

        # Move to the next month (or next quarter if you prefer)
        current_date = current_date + pd.DateOffset(months=1)

    # Build final DataFrame
    df = pd.DataFrame(data_rows)
    return df



def hyperparam_tune_fundamental_rf(X, y):
    """
    Example hyperparameter tuning with RandomizedSearchCV for fundamentals model.
    """
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    rsearch = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1
    )
    rsearch.fit(X,y)
    print(f"Fundamental Model - Best Params => {rsearch.best_params_}")
    best_model = rsearch.best_estimator_
    return best_model


def train_real_fundamental_ml_model(ticker):
    # try load cached model ----------------------------------------------------
    cached = load_model(ticker, "fund", train_real_fundamental_ml_model)
    if cached is not None:
        print(f"[Fundamental ML for {ticker}] Loaded cached model.") # Optional: add a print statement
        return cached
    """
    On-demand, gather advanced SBF fundamentals *just* for the specified `ticker`,
    build a small dataset, do hyperparam tuning, then cache the info.

    This runs only when you actually run analysis on that ticker, 
    so it doesn't bulk train on many tickers in the background.
    """

    # 1) Gather advanced SBF fundamentals for *this* ticker
    #    For demonstration, let's pretend we fetch ~1 year or 2 years of fundamentals 
    #    from our point-in-time database or feed, building a small dataset with labels.
    single_df = gather_historical_fundamentals_SBF([ticker], start="2018-01-01")
    # If your advanced function is named differently, just call that here. 
    # The key is it returns a DataFrame with yoy_rev, yoy_ni, free_cf, ev_ebit, ev_ebitda, label

    if single_df.empty:
        print(f"No advanced fundamentals found for {ticker}. Cannot train model.")
        return None

    # 2) Clean up data, drop missing
    feats = ["yoy_rev","yoy_ni","free_cf","ev_ebit","ev_ebitda"]
    single_df.dropna(subset=feats + ["label"], inplace=True)
    if single_df.empty:
        print(f"No valid fundamental rows after dropna for {ticker}.")
        return None

    X = single_df[feats].values
    y = single_df["label"].values.astype(int)

    if len(single_df) < 20:
        print(f"Warning: only {len(single_df)} data rows for {ticker}, might be too small.")
        # You could bail out or proceed.

    # 3) Hyperparameter tune on this single-ticker dataset
    best_model = hyperparam_tune_fundamental_rf(X, y)

    # 4) Evaluate & fit final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    acc = best_model.score(X_test, y_test)
    print(f"[Fundamental ML for {ticker}] test accuracy= {acc:.2f}, #samples= {len(single_df)}")

    # 5) Cache the results with the ticker in analysis_cache
    cache = load_analysis_cache()
    if ticker not in cache:
        cache[ticker] = {}
    cache[ticker]["fundamental_ml_model"] = {
        "model_params": best_model.get_params(),
        "train_accuracy": acc,
        # You might also store the entire fitted model with pickle or joblib
        # or store the DataFrame as well if you want.
    }
    save_analysis_cache(cache)
    print(f"[Fundamental ML for {ticker}] Saving model.")
    save_model(best_model, ticker, "fund", train_real_fundamental_ml_model)

    return best_model



# --------------------------------------------------------------------------------------
#                MULTI-TIMEFRAME TECHNICAL INDICATORS + ML
# --------------------------------------------------------------------------------------
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta<0, 0)).rolling(window=period).mean()
    rs = (gain / loss).replace({0:1e-10})
    rsi = 100 - (100/(1 + rs))
    return rsi

def calculate_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    exp1 = df['Close'].ewm(span=fastperiod, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slowperiod, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bollinger_bands(df, period=20, num_std=2):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + (std*num_std)
    lower = sma - (std*num_std)
    return sma, upper, lower

def calculate_obv(df):
    obv = [0]
    close = df['Close'].values
    volume = df['Volume'].values
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9  = df['Low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26  = df['Low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2

    span_a = ((tenkan_sen + kijun_sen)/2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52  = df['Low'].rolling(window=52).min()
    span_b = ((high_52 + low_52)/2).shift(26)

    chikou_span = df['Close'].shift(-26)
    return tenkan_sen, kijun_sen, span_a, span_b, chikou_span

def fetch_multi_timeframe_data(ticker):
    """
    Fetch data for multiple intervals/timeframes only once analyzing a stock:
    5m, 30m, 1h (aka 60m), 4h (aka 240m if your yfinance version supports it),
    1d, 1wk, 1mo
    This function returns a dict: { timeframe -> DataFrame }
    If yfinance cannot fetch certain intervals (like 4h), handle gracefully.
    """
    timeframes = {
        "5m":  {"interval":"5m",   "period":"5d"},
        "30m": {"interval":"30m",  "period":"1mo"},
        "1h":  {"interval":"60m",  "period":"2mo"},  # 1h => "60m"
        "1d":  {"interval":"1d",   "period":"2y"},
        "1w":  {"interval":"1wk",  "period":"5y"},
        "1mo": {"interval":"1mo",  "period":"10y"}
    }
    results = {}
    for tf, opts in timeframes.items():
        try:
            df = yf.download(ticker, interval=opts["interval"], period=opts["period"], progress=False, auto_adjust=False)
            if not df.empty:
                results[tf] = df
        except Exception as e:
            print(f"[{tf} data] Unable to fetch: {e}")
    return results

def compute_indicators_for_one_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RSI, MACD, Bollinger, OBV, Ichimoku, etc. for a single timeframe DF.
    """
    data = df.copy()

    # 1) RSI
    delta = data['Close'].diff()
    gain = delta.where(delta>0, 0.0).rolling(14).mean()
    loss = -delta.where(delta<0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100/(1 + rs))

    # 2) MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_line']   = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist']   = data['MACD_line'] - data['MACD_signal']

    # 3) Bollinger Bands (20, 2)
    sma20 = data['Close'].rolling(window=20).mean()
    std20 = data['Close'].rolling(window=20).std()
    data['Boll_SMA']   = sma20
    data['Boll_Upper'] = sma20 + 2 * std20
    data['Boll_Lower'] = sma20 - 2 * std20

    # 4) OBV
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    # 5) Ichimoku
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(data)
    data['Ichimoku_tenkan'] = tenkan
    data['Ichimoku_kijun']  = kijun
    data['Ichimoku_spanA']  = span_a
    data['Ichimoku_spanB']  = span_b
    data['Ichimoku_chikou'] = chikou

    # drop initial NaNs
    data.dropna(inplace=True)
    return data

###################### MERGE TIMEFRAMES ######################
def compute_all_timeframe_indicators(data_map: dict) -> pd.DataFrame:
    # Same as previous, merges each timeframe's DF after computing indicators
    if not data_map:
        return pd.DataFrame()
    ind_map = {}
    for tf, df in data_map.items():
        if df.empty:
            continue
        ind_df = compute_indicators_for_one_df(df)
        ind_map[tf] = ind_df
    if not ind_map:
        return pd.DataFrame()
    # pick largest as base
    base_tf = max(ind_map.keys(), key=lambda k: len(ind_map[k]))
    base = ind_map[base_tf].copy()

    merged = base.copy()
    for tf, idf in ind_map.items():
        if tf == base_tf:
            continue
        aligned = idf.reindex(merged.index, method='ffill')
        rename_map= {}
        for c in aligned.columns:
            rename_map[c] = f"{c}_{tf}"
        aligned.rename(columns=rename_map, inplace=True)
        merged = merged.join(aligned, how='left')
    merged.dropna(inplace=True)
    return merged


###################### XGBoost HYPERPARAM TUNING ######################
def hyperparam_tune_technical_xgb(X_train, y_train):
    param_dist = {
        'n_estimators': [100,200,500,800],
        'max_depth': [3,5,7,10],
        'learning_rate': [0.01,0.05,0.1,0.2],
        'subsample': [0.8,1.0],
        'colsample_bytree': [0.8,1.0],
        'min_child_weight': [1,3,5],
        'gamma': [0,0.1,0.3],
    }
    model= xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    rs= RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    rs.fit(X_train, y_train)
    print(f"[XGBoost] best params => {rs.best_params_}")
    return rs.best_params_


def build_technical_label(df):
    """
    Basic label: if next bar's close is +0.2% above current => 1, else 0
    (Just for demonstration. Real approach might differ per timeframe.)
    """
    df['Close_next'] = df['Close'].shift(-1)
    df['pct_move_next'] = (df['Close_next'] - df['Close'])/ df['Close']*100
    df['label'] = df['pct_move_next'].apply(lambda x:1 if x>0.2 else 0)
    df.dropna(inplace=True)

def hyperparam_tune_technical_rf(X, y):
    """Example hyperparameter tuning for technical model."""
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    rsearch = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1
    )
    rsearch.fit(X,y)
    print(f"Technical Model - Best Params => {rsearch.best_params_}")
    best_model = rsearch.best_estimator_
    return best_model

def train_technical_ml_model(ticker, feature_selection_method=None):
    # try load cached model ----------------------------------------------------
    cached = load_model(ticker, "tech", train_technical_ml_model)
    if cached is not None and not feature_selection_method: # If feature_selection_method is active, retrain
        print(f"[XGB] Loaded cached technical model for {ticker}") # Optional: add a print statement
        return cached
    """
    1) For daily & weekly, fetch 3 years; for 5m,30m,1h => 1 year
    2) Compute full set of RSI, MACD, Boll, OBV, Ichimoku on each timeframe
    3) Merge into one DataFrame
    4) Label => next close > current => classification=1 else 0
    5) Feature selection optional
    6) XGBoost hyperparam tune + final train
    7) Return model
    """
    timeframe_map = {
        '5m':  ("1y","5m"),
        '30m': ("1y","30m"),
        '1h':  ("1y","60m"),
        '1d':  ("3y","1d"),
        '1w':  ("3y","1wk"),
    }
    data_map = {}
    for tf, (prd, ivl) in timeframe_map.items():
        df_tf = yf.Ticker(ticker).history(period=prd, interval=ivl)
        if df_tf.empty:
            print(f"[{tf}] empty for {ticker}, skipping.")
            continue
        data_map[tf] = df_tf
    
    merged_df = compute_all_timeframe_indicators(data_map)
    if merged_df.empty or len(merged_df) < 50:
        print(f"No valid multi-timeframe data for {ticker}.")
        return None

    # define label => next day/next bar
    merged_df['NextClose'] = merged_df['Close'].shift(-1)
    merged_df['label'] = (merged_df['NextClose']> merged_df['Close']).astype(int)
    merged_df.dropna(subset=['label'], inplace=True)

    # We remove any raw columns if you don't want them as features
    # We keep all the indicator columns from each timeframe
    drop_cols = ['Open','High','Low','Close','Volume','NextClose']
    feats = [c for c in merged_df.columns if c not in drop_cols+['label']]
    X_full = merged_df[feats].values
    y_full = merged_df['label'].values

    # optional feature selection
    if feature_selection_method=='PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)
        X_full = pca.fit_transform(X_full)
    elif feature_selection_method=='SHAP':
        # placeholder
        pass

    # train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=False
    )

    # hyperparam tune XGBoost
    best_params = hyperparam_tune_technical_xgb(X_train, y_train)

    # final XGBoost
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        eval_metric='logloss',
        verbose=False
    )
    from sklearn.metrics import accuracy_score
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"[XGB] final val acc= {acc:.3f}, #rows= {len(merged_df)}")

    print(f"[XGB] Saving technical model for {ticker}") # Optional: add a print statement
    save_model(model, ticker, "tech", train_technical_ml_model)
    return model


# --------------------------------------------------------------------------------------
#             FUNCTIONS FOR ACTUAL ANALYSIS
# --------------------------------------------------------------------------------------
def fetch_fundamentals(ticker):
    """
    Basic fundamentals from yfinance (not truly SBF).
    yoy_rev, yoy_ni, free_cf, enterprise_value, ebitda, operating_income, forward_pe, etc.
    """
    stock= yf.Ticker(ticker)
    data={}
    try:
        info= stock.info
        data['forward_pe']= info.get('forwardPE', None)
        data['trailing_pe']= info.get('trailingPE', None)
        data['market_cap']= info.get('marketCap', None)
        data['enterprise_value']= info.get('enterpriseValue', None)
        data['ebitda']= info.get('ebitda', None)
        data['operating_income']= None  # parse from statements
    except:
        pass

    try:
        fin= stock.financials
        cf = stock.cashflow

        def safe_float(df, row):
            try:
                return float(df.loc[row][0])
            except:
                return None

        rev= safe_float(fin,'Total Revenue')
        ni = safe_float(fin,'Net Income')
        opinc= safe_float(fin,'Operating Income')
        data['operating_income']= opinc
        op_cf = safe_float(cf, 'Total Cash From Operating Activities')
        capex= safe_float(cf, 'Capital Expenditures')
        if op_cf and capex:
            data['free_cf']= op_cf+ capex
        else:
            data['free_cf']= None

        yoy_rev=0
        yoy_ni=0
        if fin.shape[1]>1:
            try:
                rev_vals= fin.loc['Total Revenue'][:2].values.astype(float)
                if len(rev_vals)==2 and rev_vals[1]!=0:
                    yoy_rev= (rev_vals[0]/ rev_vals[1])-1
            except:
                pass
            try:
                ni_vals= fin.loc['Net Income'][:2].values.astype(float)
                if len(ni_vals)==2 and ni_vals[1]!=0:
                    yoy_ni= (ni_vals[0]/ ni_vals[1])-1
            except:
                pass
        data['yoy_rev']= yoy_rev
        data['yoy_ni']= yoy_ni
    except:
        pass

    if data.get('enterprise_value') and data.get('ebitda') and data['ebitda']>0:
        data['ev_ebitda']= data['enterprise_value']/ data['ebitda']
    else:
        data['ev_ebitda']= None
    return data


def ml_fundamental_classification(fdata):
    if (REAL_FUNDAMENTAL_ML_MODEL is None) or (not fdata):
        return None
    yoy_rev= fdata.get('yoy_rev',0)
    yoy_ni= fdata.get('yoy_ni',0)
    fcf= fdata.get('free_cf',0)
    ev_ebit=0
    if fdata.get('enterprise_value') is not None and fdata.get('operating_income') is not None:
        opi = fdata['operating_income']
        if opi > 0:
            ev_ebit = fdata['enterprise_value'] / opi
    ev_ebitda= fdata.get('ev_ebitda',0) if fdata.get('ev_ebitda') else 0

    arr= np.array([[yoy_rev,yoy_ni,fcf,ev_ebit,ev_ebitda]], dtype=float)
    pred= REAL_FUNDAMENTAL_ML_MODEL.predict(arr)
    if len(pred)==0:
        return None
    c= pred[0]
    if c==2:
        return "Bullish"
    elif c==1:
        return "Hold"
    else:
        return "Bearish"

def score_fundamentals_with_ml(fund_data):
    score=0
    yoy_rev= fund_data.get('yoy_rev',0)
    if yoy_rev>0:
        score+=2
    yoy_ni= fund_data.get('yoy_ni',0)
    if yoy_ni>0:
        score+=2
    fcf= fund_data.get('free_cf',0) if fund_data.get('free_cf') else 0
    if fcf>1e8:
        score+=2
    ev_ebitda = fund_data.get('ev_ebitda', None)
    if ev_ebitda is not None and ev_ebitda < 12:
        score += 2
    if score>10:
        score=10

    ml_class= ml_fundamental_classification(fund_data)
    if ml_class=="Bullish":
        score+=2
    elif ml_class=="Hold":
        score+=1
    if score>12:
        score=12

    if score>=10:
        op= "Bullish"
    elif score>=6:
        op= "Hold"
    else:
        op= "Bearish"
    return score, op

def compute_buy_trigger_based_on_forward_pe(current_price, forward_pe, target_pe=15.0):
    if forward_pe is None or forward_pe <= 0:
        return None
    forward_earnings = current_price / forward_pe
    buy_price = target_pe * forward_earnings
    return buy_price


def do_advanced_fundamental_analysis(ticker):
    """
    - Calls get_macro_data, compute_macro_discount_rate
    - fetches actual advanced SBF fundamentals
    - scores them with ML
    - does a real DCF if free_cf>0
    """
    macro= get_macro_data()
    disc= compute_macro_discount_rate(macro)
    data= fetch_survivorship_bias_free_fundamentals(ticker, as_of_date=datetime.date.today())
    if not data:
        return {
            'funda_score': 0,
            'funda_opinion': "Bearish",
            'dcf_value': None,
            'macro_data': macro,
            'fund_data': None
        }
    # Score fundamentals with ML
    funda_score, funda_op= score_fundamentals_with_ml(data)

    dcf_val= None
    fcf= data.get('free_cf',0)
    if fcf>0:
        # same naive 5-year DCF approach
        yrs=5
        hi_g=0.05
        per_g=0.02
        cfs=[]
        for i in range(1,yrs+1):
            cfs.append(fcf*((1+hi_g)** i))
        disc_cfs= [ cfs[i]/((1+disc)**(i+1)) for i in range(yrs)]
        tv= (cfs[-1]*(1+per_g))/ (disc- per_g)
        disc_tv= tv / ((1+disc)**yrs)
        dcf_val= sum(disc_cfs)+ disc_tv

    return {
        'funda_score': funda_score,
        'funda_opinion': funda_op,
        'dcf_value': dcf_val,
        'macro_data': macro,
        'fund_data': data
    }



# --------------------------------------------------------------------------------------
#                TECHNICAL ANALYSIS USING MULTI-TIMEFRAME + ML
# --------------------------------------------------------------------------------------
def technical_analysis_multi_tf(ticker):
    """
    1) Check or train an XGBoost model for the ticker
    2) For quick inference, fetch the same timeframes (1y or 3y) but typically just the last portion
    3) Build the same indicators, predict the last row => return (score, label)
    """
    cache = load_analysis_cache()
    if ticker not in cache or "technical_xgb_model" not in cache[ticker]:
        print(f"No XGBoost model in cache for {ticker}. Training new ...")
        model = train_technical_ml_model(ticker)
        if model is None:
            return 0.0, "NoModel"
        cache[ticker]["technical_xgb_model"] = "SAVED_xgb_placeholder"
        save_analysis_cache(cache)
    else:
        print(f"Found XGBoost model in cache for {ticker} (Placeholder).")
        # In real scenario, load model from disk
        model = train_technical_ml_model(ticker)  # Re-train for demonstration

    # For inference, fetch the same timeframes (1y or 3y)
    # Usually we'd do a short fetch, but let's keep consistent
    timeframe_map = {
        '5m': ("1y","5m"),
        '30m':("1y","30m"),
        '1h': ("1y","60m"),
        '1d': ("3y","1d"),
        '1w': ("3y","1wk"),
    }
    data_map={}
    for tf,(prd,ivl) in timeframe_map.items():
        df= yf.Ticker(ticker).history(period=prd, interval=ivl)
        if not df.empty:
            data_map[tf]= df

    inf_df= compute_all_timeframe_indicators(data_map)
    if inf_df.empty:
        return 0.0, "NoData"

    # No need to define label; we just want the last row’s indicators
    drop_cols= ['Open','High','Low','Close','Volume']
    feats= [c for c in inf_df.columns if c not in drop_cols]
    X= inf_df[feats].values
    X_last= X[-1].reshape(1, -1)

    prob = model.predict_proba(X_last)[0][1]
    score= prob*10.0
    if prob>0.66:
        label="Bullish"
    elif prob>0.33:
        label="Neutral"
    else:
        label="Bearish"
    return score,label



# --------------------------------------------------------------------------------------
#             PRICE FORECAST (ARIMA or fallback)
# --------------------------------------------------------------------------------------
def try_arima_models(hist_close):
    orders = [(1,1,1),(2,1,2),(5,1,0),(3,1,1),(1,1,2)]
    best_aic= float('inf')
    best_model=None
    for od in orders:
        try:
            model= ARIMA(hist_close, order=od)
            mf= model.fit()
            if mf.aic<best_aic:
                best_aic= mf.aic
                best_model= mf
        except:
            continue
    return best_model

def fallback_linear_regression(hist_close, future_days):
    hist_close_ts= hist_close.to_timestamp().reset_index()
    hist_close_ts['Date_Ordinal']= hist_close_ts['Date'].map(datetime.date.toordinal)
    X= hist_close_ts[['Date_Ordinal']]
    y= hist_close_ts['Close']
    lin= LinearRegression()
    lin.fit(X,y)
    last_date= hist_close_ts['Date'].iloc[-1]
    future_date= last_date+ timedelta(days=future_days)
    future_ord= np.array([[future_date.toordinal()]])
    fc= lin.predict(future_ord)[0]
    return float(fc)

def price_forecast(hist_data):
    if hist_data.empty:
        return {}
    df= hist_data.copy()
    df= df.tz_localize(None)
    c= df['Close']
    c.index= pd.DatetimeIndex(c.index).to_period('D')
    out={}
    for d in [14,30]:
        try:
            best_model= try_arima_models(c)
            if best_model:
                pred= best_model.forecast(steps=d)
                out[d] = float(pred.iloc[-1])
            else:
                out[d]= fallback_linear_regression(c,d)
        except:
            out[d]= fallback_linear_regression(c,d)
    return out


# --------------------------------------------------------------------------------------
#             COMBINED ANALYSIS
# --------------------------------------------------------------------------------------
def combined_analysis(ticker):
    try:
        # 1) Fundamentals
        adv = do_advanced_fundamental_analysis(ticker)
        f_sc = adv['funda_score']
        f_op = adv['funda_opinion']
        dcf_v = adv['dcf_value']
        macro = adv['macro_data']
        fund_data = adv['fund_data']

        # 2) Price forecast
        st   = yf.Ticker(ticker)
        hist = st.history(period="5y")
        if hist.empty:
            fc = {}
            cp = 0
        else:
            fc = price_forecast(hist)
            cp = float(hist['Close'].iloc[-1])

        # 3) Technical
        t_sc, t_label = technical_analysis_multi_tf(ticker)
        final_s = f_sc + t_sc

        # 4) Macro
        final_s = macro_adjust_final_score(final_s, macro)
        if final_s > 20:
            final_s = 20

        # 5) ATR-based volatility penalty or check
        if not hist.empty and cp>0:
            atr_df = compute_adaptive_atr(hist, period=14, adaptive=True)
            last_row = atr_df.iloc[-1]
            atr_val  = last_row.get('Adaptive_ATR', last_row.get('ATR', None))
            if atr_val:
                atr_pct = (atr_val / cp)*100
                # Example: if ATR% > 10 => penalty 2, if >5 => penalty 1
                if atr_pct>10:
                    final_s -= 2
                elif atr_pct>5:
                    final_s -= 1
                if final_s<0:
                    final_s=0

        # 6) Score => Recommendation
        if final_s >= 15:
            over_rec = "Strong Buy"
        elif final_s >= 10:
            over_rec = "Buy"
        elif final_s >= 5:
            over_rec = "Hold"
        else:
            over_rec = "Sell"

        # 7) Optional: Buy trigger price from forward P/E
        buy_trigger_price = None
        if fund_data:
            buy_trigger_price = compute_buy_trigger_based_on_forward_pe(cp, fund_data.get('forward_pe'), target_pe=15.0)

        # 8) If we ended up with a bullish rec, use the adaptive ATR for recommended stop/profit
        recommended_entry = None
        recommended_stop  = None
        recommended_profit= None

        if over_rec in ["Buy","Strong Buy"] and not hist.empty:
            atr_data = compute_adaptive_atr(hist, period=14, adaptive=True)
            # fallback to last ATR if adaptive not present
            last_atr_val = atr_data['Adaptive_ATR'].iloc[-1] if 'Adaptive_ATR' in atr_data else atr_data['ATR'].iloc[-1]
            # For demonstration: stop = entry - 1.5× ATR, take profit = entry + 3× ATR
            recommended_entry  = cp
            recommended_stop   = cp - (1.5 * last_atr_val)
            recommended_profit = cp + (3.0 * last_atr_val)

            # (Optional) check R:R ratio:
            risk   = recommended_entry - recommended_stop
            reward = recommended_profit - recommended_entry
            if reward<=0 or risk<=0 or (reward/risk<2.0):
                # If <2:1 risk:reward, we might degrade the rec to "Hold"
                over_rec = "Hold"

        combined = {
            'ticker': ticker,
            'fundamental_score': f_sc,
            'fundamental_opinion': f_op,
            'technical_score': t_sc,
            'technical_label': t_label,
            'final_score': final_s,
            'overall_recommendation': over_rec,
            'dcf_value': dcf_v,
            'macro_discount_rate': compute_macro_discount_rate(macro),
            'current_price': cp,
            'forecasts': {str(k): float(v) for k, v in fc.items()},
            'expected_returns': {},
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'buy_trigger_price': buy_trigger_price,
            'recommended_entry_price': recommended_entry,
            'recommended_stop_loss': recommended_stop,
            'recommended_take_profit': recommended_profit,
        }
        for k, v in fc.items():
            k_int = int(k)
            if cp > 0:
                ret = ((v - cp) / cp)*100
                combined['expected_returns'][k_int] = ret
            else:
                combined['expected_returns'][k_int] = 0
        return combined
    except Exception as e:
        print("Error in comprehensive analysis:", e)
        return None

def display_analysis_results(res):
    print("\n=== Full Analysis Results ===")
    print(f"Ticker: {res['ticker']}")
    print(f"Fundamental Score: {res['fundamental_score']:.1f} => {res['fundamental_opinion']}")
    
    dcf_v= res.get('dcf_value', None)
    if dcf_v:
        print(f"DCF Value: ${dcf_v:,.0f}")
    else:
        print("DCF Value: N/A")

    print(f"Macro-based Discount Rate: {res['macro_discount_rate']*100:.2f}%")
    print(f"Technical Score: {res['technical_score']:.1f} => {res['technical_label']}")
    print(f"Final Score: {res['final_score']:.1f}/20 => {res['overall_recommendation']}")

    print(f"Current Price: ${res['current_price']:.2f}")

    # Show Price Forecast
    print("\nForecasts:")
    for d in sorted(res['forecasts'].keys(), key=lambda x:int(x)):
        fv = res['forecasts'][d]
        ret= res['expected_returns'][int(d)]
        print(f"  {d} days => ${fv:.2f}, ExpRet={ret:.2f}%")

    # --- NEW: If recommended as buy, show the suggested levels ---
    if res['overall_recommendation'] in ["Buy", "Strong Buy"]:
        entry  = res.get('recommended_entry_price')
        stop   = res.get('recommended_stop_loss')
        profit = res.get('recommended_take_profit')
        if entry and stop and profit:
            print("\nRecommended Trade Setup:")
            print(f"  Entry Price:       ${entry:.2f}")
            print(f"  Stop-Loss Price:   ${stop:.2f}")
            print(f"  Take-Profit Price: ${profit:.2f}")

    print("=== End ===")


# --------------------------------------------------------------------------------------
#             BUY/SELL LEVELS
# --------------------------------------------------------------------------------------
def buy_sell_levels(hist_data):
    """
    Enhanced to incorporate ATR logic into support/resistance or buy/sell zones.
    """
    lookback=60
    df= hist_data.tail(lookback)
    s= df['Close'].min()
    r= df['Close'].max()

    # We'll compute the adaptive ATR from the last 60 bars
    atr_df = compute_adaptive_atr(df, period=14, adaptive=True)
    last_row = atr_df.iloc[-1]
    atr_val = last_row.get('Adaptive_ATR', last_row.get('ATR', None))
    current_price = df['Close'].iloc[-1]


    zone_buy_low = current_price - current_atr
    zone_buy_high = current_price - 0.5 * current_atr
    zone_sell_low = current_price + 0.5 * current_atr
    zone_sell_high = current_price + current_atr

    return {
        'support': s,
        'resistance': r,
        'buy_zone': (zone_buy_low, zone_buy_high),
        'sell_zone': (zone_sell_low, zone_sell_high),
        'atr': atr_val
    }


# --------------------------------------------------------------------------------------
#             COMPARE/RANK RESULTS
# --------------------------------------------------------------------------------------
def compare_cached_results():
    c= load_analysis_cache()
    if not c:
        print("No data in cache.")
        return
    choices= ["All Cached Results","By Sector","By Industry","Specify List of Tickers"]
    q= [ inquirer.List('mode', message="Compare by?", choices=choices) ]
    ans= inquirer.prompt(q)
    mode= ans['mode']
    items=[]
    for tkr, data in c.items():
        sc= data.get('final_score',0)
        sec= data.get('sector_chosen', None)
        ind= data.get('industry_chosen', None)
        items.append((sc,tkr,sec,ind))
    if mode=="All Cached Results":
        flt= items
    elif mode=="By Sector":
        sl= list(SECTOR_INDUSTRY_MAP.keys())
        sq= [ inquirer.List('pick_sec', message="Which sector?", choices=sl) ]
        a2= inquirer.prompt(sq)
        pick= a2['pick_sec']
        flt= [x for x in items if x[2]== pick]
    elif mode=="By Industry":
        all_inds=[]
        for s,ll in SECTOR_INDUSTRY_MAP.items():
            all_inds.extend(ll)
        all_inds= list(set(all_inds))
        all_inds.sort()
        iq= [ inquirer.List('pick_industry', message="Which Industry?", choices=all_inds) ]
        a3= inquirer.prompt(iq)
        pick= a3['pick_industry']
        flt= [x for x in items if x[3]== pick]
    else:
        typed= input("Comma-sep tickers: ").upper()
        arr= [z.strip() for z in typed.split(",") if z.strip()]
        flt= [x for x in items if x[1] in arr]
    if not flt:
        print("No matching results.")
        return
    ranked= sorted(flt, key=lambda x:x[0], reverse=True)
    print("\n=== Ranked Cached Results ===")
    for idx,(scr,tkr,sec,ind) in enumerate(ranked, start=1):
        print(f"{idx}. {tkr} => {scr:.2f}, Sector={sec}, Industry={ind}")
    print("================================")


# --------------------------------------------------------------------------------------
#             PREDICT PROBABILITY (ATR)
# --------------------------------------------------------------------------------------
def calculate_atr(hist_data, period=14):
    df= hist_data.copy()
    df['prev_close']= df['Close'].shift(1)
    df['H-L']= df['High']- df['Low']
    df['H-Pc']= abs(df['High']- df['prev_close'])
    df['L-Pc']= abs(df['Low']- df['prev_close'])
    df['TR']= df[['H-L','H-Pc','L-Pc']].max(axis=1)
    df['ATR']= df['TR'].rolling(period).mean()
    return df['ATR'].iloc[-1]

def predict_probability(hist_data, ticker):
    """
    Example function for direct ATR-based probability + recommended stops, 
    using the new adaptive ATR approach.
    """
    if hist_data.empty:
        return None
    
    # Enhanced approach
    df_atr = compute_adaptive_atr(hist_data, period=14, adaptive=True)
    if df_atr.empty:
        return None
    
    atr_val = df_atr['Adaptive_ATR'].iloc[-1] if 'Adaptive_ATR' in df_atr else df_atr['ATR'].iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return None
    
    cp = hist_data['Close'].iloc[-1]
    stop_loss = cp - atr_val
    take_profit = cp + 2*atr_val

    df = hist_data.copy()
    df['Next_High'] = df['High'].shift(-1)
    df['Next_Low'] = df['Low'].shift(-1)
    valid = df.dropna(subset=['Next_High', 'Next_Low'])
    if valid.empty:
        return None

    def prob_move(factor):
        up = (valid['Next_High'] > valid['Close'] + factor*atr_val).mean() * 100
        down= (valid['Next_Low']  < valid['Close'] - factor*atr_val).mean() * 100
        return up, down

    p_up1, p_down1 = prob_move(1)
    p_up3, p_down3 = prob_move(3)
    p_up5, p_down5 = prob_move(5)

    def likely_target(u, d):
        if u>d:
            return "Bullish target more likely first"
        elif d>u:
            return "Bearish target more likely first"
        else:
            return "Both equally likely"

    likely_1 = likely_target(p_up1, p_down1)
    likely_3 = likely_target(p_up3, p_down3)
    likely_5 = likely_target(p_up5, p_down5)

    total_up = p_up1+p_up3+p_up5
    total_dn = p_down1+p_down3+p_down5
    if total_up>total_dn:
        overall = "Bullish"
    elif total_dn>total_up:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {
        'current_price': cp,
        'atr_value': atr_val,
        'prob_up_1': p_up1, 'prob_down_1': p_down1,
        'prob_up_3': p_up3, 'prob_down_3': p_down3,
        'prob_up_5': p_up5, 'prob_down_5': p_down5,
        'likely_1ATR': likely_1,
        'likely_3ATR': likely_3,
        'likely_5ATR': likely_5,
        'overall_prediction': overall,
        'stop_loss_price': stop_loss,
        'take_profit_price': take_profit,
    }

def display_atr_prediction(atr):
    print("\n=== ATR Probability Analysis ===")
    cp = atr['current_price']
    print(f"Current Price: ${cp:.2f}")

    print(f"1 ATR Move => Up: {atr['prob_up_1']:.2f}%, Down: {atr['prob_down_1']:.2f}%")
    print(f"   Targets => Bullish: ${atr['bullish_1ATR']:.2f}, Bearish: ${atr['bearish_1ATR']:.2f}")
    print(f"   Likely first: {atr['likely_1ATR']}")

    print(f"3 ATR Move => Up: {atr['prob_up_3']:.2f}%, Down: {atr['prob_down_3']:.2f}%")
    print(f"   Targets => Bullish: ${atr['bullish_3ATR']:.2f}, Bearish: ${atr['bearish_3ATR']:.2f}")
    print(f"   Likely first: {atr['likely_3ATR']}")

    print(f"5 ATR Move => Up: {atr['prob_up_5']:.2f}%, Down: {atr['prob_down_5']:.2f}%")
    print(f"   Targets => Bullish: ${atr['bullish_5ATR']:.2f}, Bearish: ${atr['bearish_5ATR']:.2f}")
    print(f"   Likely first: {atr['likely_5ATR']}")

    print(f"Overall ATR-based Prediction: {atr['overall_prediction']}\n")


# --------------------------------------------------------------------------------------
#             HISTORY / MENU
# --------------------------------------------------------------------------------------
def save_to_history(res):
    with open("history.log",'a') as f:
        f.write(f"\n=== {res['ticker']} on {datetime.datetime.now()} ===\n")
        f.write(f"Final Score: {res['final_score']:.2f}, OverallRec={res['overall_recommendation']}\n")
        if res['buy_trigger_price']:
            f.write(f"BuyTriggerPrice= {res['buy_trigger_price']:.2f}\n")
        f.write("-"*50+"\n")

def analysis_and_prediction_history_menu():
    while True:
        print("\n--- Analysis & Prediction History ---")
        print("1) View Analysis History (history.log)")
        print("2) View Prediction History (placeholder)")
        print("3) Return to Main Menu")
        c= input("Option: ").strip()
        if c=='1':
            if os.path.exists("history.log"):
                with open("history.log",'r') as f:
                    print(f.read())
            else:
                print("No history.log found.")
        elif c=='2':
            print("Placeholder for extended prediction history functionality.")
        elif c=='3':
            break
        else:
            print("Invalid.")


def prompt_sector_industry():
    s_list= list(SECTOR_INDUSTRY_MAP.keys())
    q1= [ inquirer.List('sector', message="Select Sector:", choices=s_list) ]
    a1= inquirer.prompt(q1)
    chosen_s= a1['sector']
    i_list= SECTOR_INDUSTRY_MAP[chosen_s]
    q2= [ inquirer.List('industry', message="Select Industry:", choices=i_list) ]
    a2= inquirer.prompt(q2)
    chosen_i= a2['industry']
    return chosen_s, chosen_i


# --------------------------------------------------------------------------------------
#             MAIN MENU
# --------------------------------------------------------------------------------------
def main_menu():
    while True:
        print("\n====== Combined Stock Analysis Tool (Multi-Timeframe + SBF + HPT) ======")
        print("1. Download SEC Filings")
        print("2. Comprehensive Analysis (Fundamental + Multi-Timeframe Technical)")
        print("3. Buy/Sell Levels")
        print("4. Predict Probability (ATR Targets)")
        print("5. Analysis & Prediction History (Combined)")
        print("6. Compare/Rank Cached Results")
        print("7. Modify Sector/Industry for Analyzed Ticker")
        print("8. Exit")

        choice = input("Select an option (1-8): ").strip()
        if choice == '1':
            download_sec_filings()

        elif choice == '2':
            tkr = input("Enter Ticker: ").upper()
            cache = load_analysis_cache()
            already_has_sector_ind = (
                tkr in cache and 
                'sector_chosen' in cache[tkr] and 
                'industry_chosen' in cache[tkr]
            )

            # Train fundamental & technical if desired (or re-train each time)
            print("\nInitializing Fundamental ML Model ...")
            global REAL_FUNDAMENTAL_ML_MODEL
            REAL_FUNDAMENTAL_ML_MODEL = train_real_fundamental_ml_model(tkr)
            print("Initializing Technical ML Model ...")
            global REAL_TECHNICAL_ML_MODEL
            REAL_TECHNICAL_ML_MODEL = train_technical_ml_model(tkr)

            result = combined_analysis(tkr)
            if not result:
                print("Analysis failed.")
                continue

            # Incorporate ATR Probability here:
            hist_for_atr = yf.Ticker(tkr).history(period="1y")
            atr_pred = predict_probability(hist_for_atr, tkr) if not hist_for_atr.empty else None
            result['atr_prediction'] = atr_pred

            display_analysis_results(result)
            if atr_pred:
                display_atr_prediction(atr_pred)

            # Prompt sector/industry ONLY if not already assigned:
            if not already_has_sector_ind:
                sec, ind = prompt_sector_industry()
                result['sector_chosen'] = sec
                result['industry_chosen'] = ind

            cache[tkr] = result
            save_analysis_cache(cache)
            save_to_history(result)

        elif choice == '3':
            t= input("Ticker: ").upper()
            try:
                st= yf.Ticker(t)
                hist= st.history(period="5y")
                if hist.empty:
                    print("No data from yfinance.")
                    continue
                lv= buy_sell_levels(hist)
                print(f"Support= {lv['support']:.2f}, Resist= {lv['resistance']:.2f}")
                print(f"BuyZone= {lv['buy_zone']}, SellZone= {lv['sell_zone']}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            t = input("Ticker: ").upper()
            try:
                st = yf.Ticker(t)
                hist = st.history(period="1y")
                pp = predict_probability(hist, t)
                if pp:
                    display_atr_prediction(pp)
                else:
                    print("No probability data or insufficient historical data.")
            except Exception as e:
                print(f"Error: {e}")

        elif choice=='5':
            analysis_and_prediction_history_menu()

        elif choice=='6':
            compare_cached_results()

        elif choice=='7':
            c= load_analysis_cache()
            if not c:
                print("No analysis cache found.")
            else:
                tlist= list(c.keys())
                if not tlist:
                    print("No tickers in cache.")
                else:
                    q= [ inquirer.List('pick_ticker', message="Which Ticker to modify sector/industry?", choices=tlist) ]
                    ans= inquirer.prompt(q)
                    chosen= ans['pick_ticker']
                    sec,ind= prompt_sector_industry()
                    c[chosen]['sector_chosen']= sec
                    c[chosen]['industry_chosen']= ind
                    save_analysis_cache(c)
                    print(f"Updated {chosen} => Sector={sec}, Industry={ind}")

        elif choice=='8':
            print("Exiting. Goodbye.")
            sys.exit(0)

        else:
            print("Invalid choice.")


# --------------------------------------------------------------------------------------
#             INIT MODELS AT SCRIPT LOAD
# --------------------------------------------------------------------------------------
REAL_FUNDAMENTAL_ML_MODEL = None
REAL_TECHNICAL_ML_MODEL   = None

def init_models():
    global REAL_FUNDAMENTAL_ML_MODEL, REAL_TECHNICAL_ML_MODEL
    print("\nInitializing Fundamental ML Model (SBF, Tuning)...")
    REAL_FUNDAMENTAL_ML_MODEL = train_real_fundamental_ml_model(ticker)
    print("Initializing Technical ML Model (Tuning)...")
    REAL_TECHNICAL_ML_MODEL = train_technical_ml_model(ticker)
    print("... ML Initialization complete.\n")


if __name__=="__main__":
    #init_models()
    main_menu()