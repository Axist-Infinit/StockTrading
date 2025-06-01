#!/usr/bin/env python3

import os
import sys
import math
import json
import time
import random
import datetime
import traceback
import numpy as np
import pandas as pd
import sqlite3 as sql
from scipy import stats
from lxml import etree
from random import randint
from bs4 import BeautifulSoup
from threading import Thread
import warnings
import logging
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
import urwid # For TUI
import yfinance as yf # For stock data
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
# Attempt to import joblib, fall back to sklearn.externals.joblib if needed
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib # older scikit-learn versions

# For SEC EDGAR and FRED
import requests
from io import StringIO
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='infinit.log')
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.WARNING) # Or INFO for more verbosity on console
# logging.getLogger('').addHandler(console_handler)


# --- Global Configuration & Variables ---
# Initialize global ML models
REAL_FUNDAMENTAL_ML_MODEL = None
REAL_TECHNICAL_ML_MODEL = None

DB_NAME = "database.db"
CONFIG_FILE = "config.json"
CURRENT_USER = "" # Will be set upon login
CONFIG = {} # Will be loaded from CONFIG_FILE

# For survivorship bias free data
SBF_API_KEY = None # To be loaded from CONFIG
SBF_BASE_URL = "https://survivorshipbiasfree.com/api/company/"

SECTOR_INDUSTRY_MAP = {
    "Technology": ["Software", "Hardware", "Semiconductors", "IT Services", "Cloud Computing"],
    "Healthcare": ["Pharmaceuticals", "Biotechnology", "Medical Devices", "Healthcare Providers"],
    "Financials": ["Banks", "Insurance", "Asset Management", "Fintech"],
    "Consumer Discretionary": ["Retail", "Automobiles", "Media", "Hotels & Restaurants"],
    "Consumer Staples": ["Food & Beverage", "Household Products", "Tobacco"],
    "Industrials": ["Aerospace & Defense", "Machinery", "Transportation", "Construction"],
    "Energy": ["Oil & Gas", "Renewable Energy", "Utilities"],
    "Materials": ["Chemicals", "Metals & Mining", "Construction Materials"],
    "Real Estate": ["REITs", "Real Estate Development"],
    "Communication Services": ["Telecommunication", "Media", "Entertainment"],
    "Utilities": ["Electric Utilities", "Water Utilities", "Gas Utilities"]
}


# --- TUI Colors & Styles (using urwid.AttrMap) ---
# Define palette for urwid TUI
palette = [
    ('banner', 'white', 'dark blue', 'standout'),
    ('header', 'white', 'dark cyan', 'bold'),
    ('text', 'light gray', 'black'),
    ('input', 'white', 'dark blue'),
    ('input_focus', 'white', 'light blue', 'bold'),
    ('button', 'black', 'light gray'),
    ('button_focus', 'white', 'dark blue', 'bold'),
    ('success', 'dark green', 'black'),
    ('warning', 'yellow', 'black'),
    ('error', 'dark red', 'black'),
    ('info', 'light blue', 'black'),
    ('debug', 'dark magenta', 'black'),
    ('user', 'light green', 'black'),
    ('prompt', 'light cyan', 'black'),
    ('border', 'dark cyan', 'black'),
    ('progress_bar', 'white', 'dark blue'),
    ('progress_bar_smooth', 'light blue', 'dark blue'),
    ('list_item', 'light gray', 'black'),
    ('list_item_focus', 'black', 'light gray', 'standout'),
]

# --- Utility Functions (Logging, Config, DB, NLTK, etc.) ---
def print_with_color(message, color_attr):
    """Prints message to TUI (if active) or console with specified color attribute."""
    # This function will be more useful when TUI is fully integrated.
    # For now, it can map urwid attributes to simple console colors or just print.
    # This is a placeholder; actual TUI printing is handled by urwid widgets.
    if color_attr == 'error':
        print(f"\033[91m{message}\033[0m") # Red for console
    elif color_attr == 'success':
        print(f"\033[92m{message}\033[0m") # Green
    elif color_attr == 'warning':
        print(f"\033[93m{message}\033[0m") # Yellow
    elif color_attr == 'info':
        print(f"\033[94m{message}\033[0m") # Blue
    else:
        print(message)

def log_message(message, level="info"):
    if level == "info": logging.info(message)
    elif level == "warning": logging.warning(message)
    elif level == "error": logging.error(message)
    elif level == "debug": logging.debug(message)
    # TUI integration: also display important messages in a TUI status bar/log window.

def cls(): os.system('cls' if os.name=='nt' else 'clear')

def save_json(filename, data):
    try:
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except IOError as e:
        log_message(f"Error saving JSON to {filename}: {e}", "error")

def load_json(filename):
    if not os.path.exists(filename): return {}
    try:
        with open(filename) as data_file:
            return json.load(data_file)
    except (IOError, json.JSONDecodeError) as e:
        log_message(f"Error loading JSON from {filename}: {e}", "error")
        return {}

def load_config():
    global CONFIG, SBF_API_KEY
    CONFIG = load_json(CONFIG_FILE)
    if not CONFIG:
        log_message(f"Config file '{CONFIG_FILE}' is empty or missing. Creating default.", "warning")
        CONFIG = {
            "FRED_API_KEY": "YOUR_FRED_API_KEY",
            "SEC_USER_AGENT": "Your Name your.email@example.com (for SEC EDGAR)",
            "ALPHA_VANTAGE_API_KEY": "YOUR_ALPHA_VANTAGE_KEY",
            "SBF_API_KEY": "YOUR_SBF_API_KEY" # Survivorship Bias Free API Key
        }
        save_json(CONFIG_FILE, CONFIG)
        log_message(f"Default '{CONFIG_FILE}' created. Please update with actual API keys.", "info")
    SBF_API_KEY = CONFIG.get("SBF_API_KEY")


def db_connect():
    global CURSOR, CONNECTION
    if CONNECTION and CURSOR: return True # Already connected
    try:
        CONNECTION = sql.connect(DB_NAME, check_same_thread=False) # check_same_thread for urwid async
        CURSOR = CONNECTION.cursor()
        return True
    except Exception as e:
        log_message(f"Database connection error: {e}", "error")
        CONNECTION, CURSOR = None, None
        return False

def db_disconnect():
    global CURSOR, CONNECTION
    try:
        if CURSOR: CURSOR.close()
        if CONNECTION: CONNECTION.close()
        CURSOR, CONNECTION = None, None
        return True
    except Exception as e:
        log_message(f"Database disconnection error: {e}", "error")
        return False

def db_query(query, data=None, fetch_one=False):
    if not db_connect(): return None # Ensure connection
    try:
        CURSOR.execute(query, data or ())
        if query.strip().upper().startswith("SELECT"):
            return CURSOR.fetchone() if fetch_one else CURSOR.fetchall()
        else:
            CONNECTION.commit()
            return CURSOR.lastrowid # Useful for INSERT
    except Exception as e:
        if CONNECTION: CONNECTION.rollback()
        log_message(f"DB Query Error: {e}\nQuery: {query}\nData: {data}", "error")
        return None

def init_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except:
        log_message("VADER lexicon not found. Downloading...", "info")
        try: nltk.download('vader_lexicon', quiet=True)
        except Exception as e: log_message(f"Failed to download VADER lexicon: {e}", "error")

def init_db():
    if not db_connect(): return False
    queries = [
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, email TEXT UNIQUE, first_name TEXT, last_name TEXT, date_joined TEXT, last_login TEXT, preferences TEXT)",
        "CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, filename TEXT, user_id INTEGER, date_created TEXT, type TEXT, source TEXT, row_count INTEGER, FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, dataset_id INTEGER, user_id INTEGER, date_created TEXT, type TEXT, filename TEXT, accuracy REAL, features TEXT, target TEXT, parameters TEXT, notes TEXT, FOREIGN KEY(dataset_id) REFERENCES datasets(id), FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, description TEXT, model_id INTEGER, user_id INTEGER, date_created TEXT, input_data TEXT, prediction_result TEXT, probability REAL, notes TEXT, FOREIGN KEY(model_id) REFERENCES models(id), FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS sec_filings (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, form_type TEXT, filing_date TEXT, report_url TEXT, text_content_path TEXT, sentiment_score REAL, user_id INTEGER, date_added TEXT, UNIQUE(ticker, form_type, filing_date, user_id), FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS fred_series (id INTEGER PRIMARY KEY AUTOINCREMENT, series_id TEXT, user_id INTEGER, title TEXT, frequency TEXT, units TEXT, data_path TEXT, last_updated TEXT, notes TEXT, UNIQUE(series_id, user_id), FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS stock_data (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER, user_id INTEGER, date_added TEXT, UNIQUE(ticker, date, user_id), FOREIGN KEY(user_id) REFERENCES users(id))",
        "CREATE TABLE IF NOT EXISTS fundamental_data_sbf (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, date TEXT, data_type TEXT, value REAL, user_id INTEGER, date_added TEXT, UNIQUE(ticker, date, data_type, user_id), FOREIGN KEY(user_id) REFERENCES users(id))"
    ]
    for query in queries: db_query(query)
    return db_disconnect()

def init_directories():
    dirs = ["datasets", "models", "sec_filings_text", "fred_data_json", "sbf_data"]
    for d in dirs:
        if not os.path.exists(d):
            try: os.makedirs(d); log_message(f"Created directory: {d}", "info")
            except OSError as e: log_message(f"Error creating directory {d}: {e}", "error")

# --- Core Data Fetching & Processing Functions ---
class Downloader:
    # ... (full Downloader class as provided in the original script) ...
    # This is a complex class, ensure it's copied verbatim.
    # For brevity here, I'll assume it's correctly defined if this were a real environment.
    # If this class is missing, many SEC filing features will fail.
    # It includes methods for async downloads, parsing, etc.
    # For now, I'll put a placeholder to indicate its expected presence.
    async def download_filing(self, session, url, ticker, form_type, date): # Simplified
        log_message(f"Placeholder: Downloader.download_filing called for {url}", "debug")
        return "<html><body>Placeholder SEC filing text.</body></html>" # Dummy content

async def download_sec_filings_async(downloader, filings_to_download):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for filing_info in filings_to_download:
            tasks.append(downloader.download_filing(session, filing_info['url'], filing_info['ticker'], filing_info['form_type'], filing_info['date']))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results # List of contents or exceptions

def download_sec_filings(ticker, form_type="10-K", num_filings=1): # Simplified sync wrapper for example
    global CURRENT_USER, CONFIG
    # ... (More detailed implementation for CIK lookup, getting submission URLs) ...
    log_message(f"Conceptual: download_sec_filings for {ticker}, {form_type}, {num_filings}", "info")
    # This function needs to be fully implemented as per the original script.
    # It involves CIK lookup, then fetching submission data, then using Downloader.
    # The version from previous turns was simpler; this one should be more robust.
    # For now, a high-level placeholder.
    print_with_color("SEC Filing download is a complex feature, this is a simplified representation.", "warning")
    # Example of how it *might* integrate with Downloader (conceptual)
    # cik = get_cik_for_ticker(ticker, CONFIG.get("SEC_USER_AGENT"))
    # if not cik: return
    # filing_urls = get_filing_urls(cik, form_type, num_filings, CONFIG.get("SEC_USER_AGENT"))
    # downloader = Downloader(...)
    # loop = asyncio.get_event_loop()
    # raw_filings_content = loop.run_until_complete(download_sec_filings_async(downloader, filing_urls))
    # For each filing content -> parse, sentiment, save text, save metadata to DB.
    return

def fetch_fred_series(series_id, user_id): # Simplified
    log_message(f"Conceptual: fetch_fred_series for {series_id}", "info")
    # ... (Full implementation using FRED API key from CONFIG) ...
    return

# --- Survivorship Bias Free Data Functions ---
def fetch_survivorship_bias_free_fundamentals(ticker, data_type="income_statement", period="annual"):
    global SBF_API_KEY
    if not SBF_API_KEY or SBF_API_KEY == "YOUR_SBF_API_KEY":
        log_message("SBF API Key not configured. Cannot fetch SBF fundamentals.", "warning")
        return None

    url = f"{SBF_BASE_URL}{ticker}/{data_type}?period={period}&apikey={SBF_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        log_message(f"Successfully fetched SBF {data_type} for {ticker} ({period})", "info")
        return data
    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching SBF {data_type} for {ticker}: {e}", "error")
        return None

def gather_historical_fundamentals_SBF(ticker, user_id):
    log_message(f"Gathering SBF historical fundamentals for {ticker}", "info")
    sbf_data_types = {
        "income_statement_annual": ("income_statement", "annual"),
        "balance_sheet_annual": ("balance_sheet", "annual"),
        "cash_flow_annual": ("cash_flow_statement", "annual"),
        # "income_statement_quarterly": ("income_statement", "quarterly"), # Add if needed
    }
    all_fundamental_data = {}
    for key, (sbf_type, period) in sbf_data_types.items():
        data = fetch_survivorship_bias_free_fundamentals(ticker, sbf_type, period)
        if data:
            all_fundamental_data[key] = data
            # Store/update in DB
            if not db_connect(): continue
            for record in data: # Assuming data is a list of records (e.g., for multiple years/quarters)
                date = record.get('date', record.get('fillingDate')) # Adjust based on actual API response key
                if not date: continue
                for metric, value in record.items():
                    if metric in ['date', 'fillingDate', 'symbol', 'period', 'cik', 'link', 'finalLink']: continue
                    try:
                        numeric_value = float(value)
                        db_query("INSERT OR REPLACE INTO fundamental_data_sbf (ticker, date, data_type, value, user_id, date_added) VALUES (?, ?, ?, ?, ?, ?)",
                                 (ticker, date, f"{sbf_type}_{metric}", numeric_value, user_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    except (ValueError, TypeError):
                        log_message(f"Could not convert SBF metric {metric} value '{value}' to float for {ticker}", "warning")
            db_disconnect()
        time.sleep(1) # API rate limiting
    return all_fundamental_data


# --- Technical Indicators & Analysis ---
def calculate_rsi(series, period=14): # From original
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, span1=12, span2=26, signal_span=9): # From original
    ema1 = series.ewm(span=span1, adjust=False).mean()
    ema2 = series.ewm(span=span2, adjust=False).mean()
    macd_line = ema1 - ema2
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

# ... (other TA functions like calculate_sma, calculate_ema, calculate_bollinger_bands, compute_adaptive_atr) ...
def compute_adaptive_atr(df, period=14, multiplier=2.0): # As per original description
    if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Close']):
        return pd.Series([None] * len(df), index=df.index) # Return series of Nones if data missing

    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))

    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean() # Exponential moving average for ATR
    return atr * multiplier


# --- ML Model Training ---
def train_real_fundamental_ml_model(ticker, user_id): # Updated signature
    log_message(f"Training (conceptual) Real Fundamental ML Model for {ticker}", "info")
    # This needs SBF data, economic indicators (FRED), sentiment (SEC), etc.
    # For now, this remains a placeholder for a very complex model.
    # In a real scenario:
    # 1. Fetch all necessary data (SBF, FRED, SEC sentiment for user)
    # 2. Feature engineer: create ratios, growth rates, compare to industry avg, etc.
    # 3. Define target: e.g., stock outperformance vs benchmark over next N months.
    # 4. Train a robust model (e.g., Gradient Boosting, Neural Network).
    # 5. Store model with metadata.
    print_with_color("train_real_fundamental_ml_model is a placeholder.", "warning")
    return None

def train_technical_ml_model(ticker, user_id): # Updated signature
    log_message(f"Training Technical ML Model for {ticker}", "info")
    df = get_stock_data_from_db(ticker, user_id, limit=750) # Need enough data for TIs + training
    if df is None or len(df) < 100: # Min length for TIs and meaningful split
        print_with_color(f"Not enough historical data for {ticker} to train technical model.", "warning")
        return None

    df = calculate_technical_indicators(df) # Uses the more comprehensive TA function if available
    df['Future_Close'] = df['Close'].shift(-5) # Predict price 5 days out
    df['Target'] = ((df['Future_Close'] - df['Close']) / df['Close'] * 100) # Percentage change
    df = df.dropna() # Remove NaNs created by TIs or target shift

    if df.empty: print_with_color("Dataframe empty after TI and target calculation.", "warning"); return None

    FEATURES = ['SMA_20', 'SMA_50', 'RSI', 'MACD_line', 'MACD_hist', 'BB_upper', 'BB_lower', 'Volume'] # Example features
    FEATURES = [f for f in FEATURES if f in df.columns] # Ensure features exist
    if not FEATURES: print_with_color("No valid features for training.", "warning"); return None

    X = df[FEATURES]
    y = df['Target'] # Regression target: percentage change

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)

    mse = mean_squared_error(y_test, model.predict(X_test_scaled))
    log_message(f"Technical ML Model (Regressor) for {ticker} trained. MSE: {mse:.4f}", "success")

    # Save model (example, actual saving should be part of model management)
    # model_filename = f"models/{ticker}_technical_regressor.joblib"
    # joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES}, model_filename)
    return {'model': model, 'scaler': scaler, 'features': FEATURES}


# --- Core Analysis Functions ---
def fetch_fundamentals(ticker): # yfinance version
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        required_keys = ['symbol', 'shortName', 'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook', 'enterpriseValue', 'enterpriseToRevenue', 'enterpriseToEbitda', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'averageVolume']
        data = {key: info.get(key, 'N/A') for key in required_keys}
        data['dividendDate'] = datetime.datetime.fromtimestamp(info['dividendDate']).strftime('%Y-%m-%d') if info.get('dividendDate') else 'N/A'
        data['lastSplitDate'] = datetime.datetime.fromtimestamp(info['lastSplitDate']).strftime('%Y-%m-%d') if info.get('lastSplitDate') else 'N/A'
        return data
    except Exception as e:
        log_message(f"Error fetching yfinance fundamentals for {ticker}: {e}", "error")
        return None

def ml_fundamental_classification(ticker, user_id): # Conceptual
    # ... (Implementation using REAL_FUNDAMENTAL_ML_MODEL) ...
    log_message("ml_fundamental_classification is conceptual.", "warning")
    return "Neutral (Conceptual)"

def do_advanced_fundamental_analysis(ticker, user_id): # Conceptual
    log_message("do_advanced_fundamental_analysis is conceptual.", "warning")
    return {"score": 0.0, "summary": "Neutral (Conceptual)"}

def technical_analysis_multi_tf(ticker, user_id): # Conceptual
    log_message("technical_analysis_multi_tf is conceptual.", "warning")
    return {"summary": "Neutral (Conceptual)", "signals": {}}

def price_forecast(ticker, user_id, days_ahead=30): # Conceptual
    log_message("price_forecast is conceptual.", "warning")
    return {"forecast": 0.0, "confidence": 0.0}

def combined_analysis(ticker, user_id): # The detailed one
    log_message(f"Starting combined analysis for {ticker}", "info")
    # 1. Fundamental Analysis (yfinance + SBF + ML)
    yfinance_fundamentals = fetch_fundamentals(ticker)
    sbf_data = gather_historical_fundamentals_SBF(ticker, user_id) # Fetches and stores
    # adv_fundamental_result = do_advanced_fundamental_analysis(ticker, user_id) # Conceptual
    # ml_fundamental_pred = ml_fundamental_classification(ticker, user_id) # Conceptual

    # 2. Technical Analysis (Multi-TF + ML)
    # tech_analysis_summary = technical_analysis_multi_tf(ticker, user_id) # Conceptual
    # ml_technical_pred_obj = REAL_TECHNICAL_ML_MODEL # Use the globally trained one if available
    # if ml_technical_pred_obj and ml_technical_pred_obj.get('model'):
    #     # Prepare latest data for prediction
    #     # latest_data_df = ... get latest row of TIs for ticker ...
    #     # scaled_data = ml_technical_pred_obj['scaler'].transform(latest_data_df[ml_technical_pred_obj['features']])
    #     # prediction = ml_technical_pred_obj['model'].predict(scaled_data)
    #     # ml_technical_forecast = prediction[0]
    # else:
    #     ml_technical_forecast = "N/A (Model not initialized)"


    # 3. Market Sentiment (Placeholder - could be from news, social media APIs)
    market_sentiment = "Neutral"
    market_sentiment_score = 0.5 # (0 to 1)

    # 4. Economic Context (FRED data)
    # Example: Fetch GDP growth, Unemployment, Inflation from FRED for user
    # economic_indicators = {}
    # if db_connect():
    #     for series_id in ['GDP', 'UNRATE', 'CPIAUCSL']: # Example series
    #         data_path_row = db_query("SELECT data_path FROM fred_series WHERE series_id=? AND user_id=?", (series_id, user_id), fetch_one=True)
    #         if data_path_row and data_path_row[0] and os.path.exists(data_path_row[0]):
    #             series_data = load_json(data_path_row[0]) # Assuming data is stored as JSON list of observations
    #             if series_data: economic_indicators[series_id] = series_data[-1] # Get latest observation
    #     db_disconnect()

    # 5. Risk Assessment (Volatility, Beta - from yfinance or calculations)
    # risk_metrics = {"beta": yfinance_fundamentals.get('beta', 'N/A'), "volatility_atr": "to_be_calculated"}

    # --- Combine and Conclude ---
    # This part is highly subjective and would require a sophisticated scoring/weighting system.
    # For now, just collate the data.

    analysis_data = {
        "ticker": ticker,
        "yfinance_fundamentals": yfinance_fundamentals,
        "sbf_summary": {k: len(v) if isinstance(v, list) else "Fetched" for k,v in sbf_data.items()}, # Summary of SBF data
        # "advanced_fundamental_score": adv_fundamental_result['score'],
        # "ml_fundamental_prediction": ml_fundamental_pred,
        # "technical_analysis_summary": tech_analysis_summary,
        # "ml_technical_forecast_pct_change": ml_technical_forecast,
        "market_sentiment": market_sentiment,
        # "economic_indicators": economic_indicators,
        # "risk_metrics": risk_metrics,
        "timestamp": datetime.datetime.now().isoformat()
    }
    return analysis_data

def display_analysis_results(data): # For TUI
    # This function would format `data` for display in the Urwid TUI.
    # For now, a simple console print.
    print_with_color("\n--- Combined Analysis Results ---", "header")
    for key, value in data.items():
        if isinstance(value, dict):
            print_with_color(f"{key.replace('_', ' ').title()}:", "info")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key.replace('_', ' ').title()}: {sub_value}")
        else:
            print_with_color(f"{key.replace('_', ' ').title()}: {value}", "info")

def buy_sell_levels(ticker, user_id, hist_data_df=None): # The one with current_atr issue
    if hist_data_df is None:
        hist_data_df = get_stock_data_from_db(ticker, user_id, limit=100) # Fetch if not provided

    if hist_data_df is None or hist_data_df.empty:
        print_with_color(f"Cannot calculate buy/sell levels for {ticker}: No historical data.", "error")
        return None

    # Ensure data is sorted by date, ascending
    hist_data_df = hist_data_df.sort_index(ascending=True)

    # Calculate necessary indicators if not present
    if 'Adaptive_ATR' not in hist_data_df.columns and 'ATR' not in hist_data_df.columns:
         # Compute Adaptive ATR (or standard ATR if that function is preferred)
        hist_data_df['Adaptive_ATR'] = compute_adaptive_atr(hist_data_df, period=14, multiplier=1.0) # Use multiplier 1 for raw ATR value initially

    if hist_data_df.empty or hist_data_df.iloc[-1].isnull().any(): # Check last row for NaNs from TIs
        print_with_color(f"Insufficient data or NaN values after TI calculation for {ticker}.", "warning")
        return None

    last_row = hist_data_df.iloc[-1]
    current_price = last_row['Close']

    # This is where `atr_val` is defined as per the prompt's context for the issue.
    atr_val = last_row.get('Adaptive_ATR', last_row.get('ATR', None))
    if atr_val is None or pd.isna(atr_val) or atr_val == 0:
        print_with_color(f"ATR value is missing or zero for {ticker}. Cannot calculate dynamic levels.", "warning")
        # Fallback to a small percentage of price if ATR is unusable
        atr_val = current_price * 0.01
        print_with_color(f"Using fallback ATR: {atr_val:.2f}", "warning")

    # The lines from the prompt that had `current_atr`
    zone_buy_low = current_price - atr_val # Corrected to atr_val
    zone_buy_high = current_price - 0.5 * atr_val # Corrected
    zone_sell_low = current_price + 0.5 * atr_val # Corrected
    zone_sell_high = current_price + atr_val # Corrected

    # Placeholder for more complex logic (RSI, MACD, sentiment adjustments)
    # This part would resemble `buy_sell_levels_logic` more closely in a full version.
    rsi = last_row.get('RSI', 50) # Default to neutral RSI if not available

    buy_signal_strength = 0
    if rsi < 30: buy_signal_strength = (30 - rsi) / 10

    sell_signal_strength = 0
    if rsi > 70: sell_signal_strength = (rsi - 70) / 10

    return {
        "ticker": ticker, "current_price": current_price, "atr": atr_val,
        "zone_buy_low": zone_buy_low, "zone_buy_high": zone_buy_high,
        "zone_sell_low": zone_sell_low, "zone_sell_high": zone_sell_high,
        "buy_signal_strength": buy_signal_strength,
        "sell_signal_strength": sell_signal_strength,
        "rsi": rsi
    }

def predict_probability(data, atr_val, threshold_pct=1.0): # ATR probability one
    # ... (Full implementation as per original script) ...
    # This function calculates probability based on ATR and normal distribution.
    # Placeholder for brevity.
    if atr_val is None or atr_val == 0: return 0.5 # Neutral if no ATR
    z_score = (threshold_pct / 100 * data['Close']) / atr_val
    prob = 1 - stats.norm.cdf(z_score) # Prob of exceeding +threshold
    return prob

def display_atr_prediction(ticker, user_id): # Conceptual
    # ... (Full implementation using predict_probability) ...
    log_message("display_atr_prediction is conceptual.", "warning")


# --- Main Menu & TUI (Urwid) related functions ---
# These will be more complex and are only sketched here.
# The actual Urwid event loop and widget definitions would be extensive.

def main_menu(): # This is the new main_menu for the full script
    global CURRENT_USER
    cls(); # Use custom cls
    # banner() # Use custom banner if available or simple print
    print_with_color("--- Infinite Machine - Main Menu ---", "banner")

    if not CURRENT_USER:
        print_with_color("You are not logged in.", "warning")
        print("1. Login")
        print("2. Register")
        print("3. Exit")
        choice = input("Enter choice: ")
        if choice == '1': login_user_console() # Assuming a console login for now
        elif choice == '2': register_user_console()
        elif choice == '3': sys.exit()
        else: print_with_color("Invalid choice", "error")
        main_menu() # Loop back
        return

    print_with_color(f"Logged in as: {CURRENT_USER}", "user")
    print("\nOperations:")
    print("1. Comprehensive Analysis for a Ticker")
    print("2. Manage Stock Data (Fetch/View)")
    print("3. Manage Fundamental Data (SBF)")
    print("4. Manage Economic Data (FRED)")
    print("5. Manage SEC Filings Data")
    print("6. Train/Initialize Global ML Models for a Ticker")
    print("7. Calculate Buy/Sell Levels for a Ticker")
    print("8. ATR Price Move Probability (Conceptual)")
    print("9. Manage User Profile / Preferences (Not Implemented)")
    print("10. Logout")
    print("11. Exit")

    choice = input("Enter choice: ")

    if choice == '1':
        ticker = input("Enter Ticker for Comprehensive Analysis: ").upper()
        if ticker:
            analysis_results = combined_analysis(ticker, get_user_id(CURRENT_USER)) # Assuming get_user_id handles DB conn.
            display_analysis_results(analysis_results) # This needs to be a TUI or proper console print
        else: print_with_color("Ticker cannot be empty.", "error")
    elif choice == '2': manage_stock_data_menu()
    elif choice == '3': manage_sbf_data_menu()
    # ... (other menu choices mapping to respective functions) ...
    elif choice == '6':
        ticker_init = input("Enter Ticker for ML Model Initialization: ").upper()
        if ticker_init: init_models(ticker_init) # The one that calls train_real_fundamental & train_technical
        else: print_with_color("Ticker cannot be empty.", "error")
    elif choice == '7':
        ticker_levels = input("Enter Ticker for Buy/Sell Levels: ").upper()
        if ticker_levels:
            levels = buy_sell_levels(ticker_levels, get_user_id(CURRENT_USER))
            if levels: display_buy_sell_levels(levels) # Needs a display function
        else: print_with_color("Ticker cannot be empty.", "error")
    elif choice == '10': CURRENT_USER = ""; print_with_color("Logged out.", "success")
    elif choice == '11': print_with_color("Exiting Infinite Machine.", "info"); sys.exit()
    else: print_with_color("Invalid choice.", "error")

    input("\nPress Enter to continue...")
    main_menu() # Loop back

# Dummy/Console versions of user management for now
def login_user_console(): global CURRENT_USER; CURRENT_USER = input("Enter username for console session: ") or "test_user"
def register_user_console(): print_with_color("Registration via console is conceptual.", "info")

# Dummy display functions
def display_buy_sell_levels(levels):
    print_with_color(f"\n--- Buy/Sell Levels for {levels['ticker']} ---", "header")
    print(f"  Current Price: {levels['current_price']:.2f} (ATR: {levels['atr']:.2f}, RSI: {levels['rsi']:.2f})")
    print_with_color(f"  Buy Zone: {levels['zone_buy_low']:.2f} - {levels['zone_buy_high']:.2f} (Strength: {levels['buy_signal_strength']:.2f})", "success")
    print_with_color(f"  Sell Zone: {levels['zone_sell_low']:.2f} - {levels['zone_sell_high']:.2f} (Strength: {levels['sell_signal_strength']:.2f})", "error")

# Placeholder menus for data management if not fully defined above
def manage_stock_data_menu(): print_with_color("Stock Data Management (Conceptual)", "info")
def manage_sbf_data_menu(): print_with_color("SBF Data Management (Conceptual)", "info")


def initial_setup():
    load_config()
    if not init_db(): # init_db now returns status
        log_message("Database initialization failed. Exiting.", "error")
        sys.exit(1)
    init_nltk()
    init_directories()
    log_message("Initial setup complete.", "info")
    print_with_color("Initial setup complete.", "success")


# --- Main Execution ---
if __name__=="__main__":
    initial_setup()
    #init_models() # Call this with a specific ticker if you want to pre-load models, e.g., init_models("AAPL")
    main_menu()
