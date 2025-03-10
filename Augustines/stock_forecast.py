# stock_forecast.py

import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import urwid
import requests

from sklearn.linear_model import LinearRegression
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Configuration for Alpha Vantage API
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '13LJ0VOBUDK3S7B7')  # Replace with your API key or set as an environment variable
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

PREDICTION_HISTORY_FILE = 'predictions_history.json'

##############################################################################
# Below are the modeling routines that were previously in advanced_modeling.py
##############################################################################

def create_advanced_features(hist_data, ticker, financials):
    """
    Example placeholder function to enrich hist_data with advanced features
    for use by forecasting models. You can expand this with fundamental or
    sentiment features, technical indicators, etc.
    """
    df = hist_data.copy()
    # Example: add some very basic columns
    df['Price_Change'] = df['Close'].diff()
    df['Rolling_Close_7'] = df['Close'].rolling(7).mean()
    df['Ticker'] = ticker

    # Suppose we store a couple of fundamentals in the DataFrame as well
    df['Fund_ROE'] = financials.get('return_on_equity', 0)
    df['Fund_DE'] = financials.get('debt_to_equity', 0)
    df.dropna(inplace=True)
    return df

def try_arima_models(hist_close):
    """
    Attempt multiple ARIMA orders and pick the best model based on AIC.
    """
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter('ignore', ConvergenceWarning)

    orders = [(1,1,1), (2,1,2), (5,1,0), (3,1,1),
              (1,1,2), (2,2,2), (4,1,2), (3,2,1)]
    best_aic = float('inf')
    best_model = None

    for order in orders:
        try:
            model = ARIMA(hist_close, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_model = model_fit
        except:
            continue
    return best_model

def fallback_linear_regression(hist_close, future_days):
    """
    Simple fallback: a linear regression that uses the date ordinal
    to predict price.
    """
    hist_close_ts = hist_close.to_timestamp().reset_index()
    hist_close_ts['Date_Ordinal'] = hist_close_ts['Date'].map(datetime.toordinal)
    X = hist_close_ts[['Date_Ordinal']]
    y = hist_close_ts['Close']

    lin_model = LinearRegression()
    lin_model.fit(X, y)

    future_date = hist_close_ts['Date'].iloc[-1] + timedelta(days=future_days)
    future_date_ordinal = np.array([[future_date.toordinal()]])
    forecast_price = lin_model.predict(future_date_ordinal)[0]
    return float(forecast_price)

def run_arima_model(hist_close):
    """
    Runs ARIMA model for forecasting. If none of the tried orders work,
    or an error occurs, fallback to the linear regression approach.
    Returns a dict of {14: forecastPriceIn14Days, 30: forecastPriceIn30Days}.
    """
    try:
        model_fit = try_arima_models(hist_close)
        if model_fit is not None:
            forecast_14 = model_fit.forecast(steps=14)[-1]
            forecast_30 = model_fit.forecast(steps=30)[-1]
            return {14: float(forecast_14), 30: float(forecast_30)}
        else:
            # fallback
            forecast_14 = fallback_linear_regression(hist_close, 14)
            forecast_30 = fallback_linear_regression(hist_close, 30)
            return {14: forecast_14, 30: forecast_30}
    except:
        forecast_14 = fallback_linear_regression(hist_close, 14)
        forecast_30 = fallback_linear_regression(hist_close, 30)
        return {14: forecast_14, 30: forecast_30}

def run_xgboost_model(hist_data_enriched):
    """
    Train and predict using an XGBoost model. This is a simple example that
    shifts Close by 14 days as the target, ignoring real sequence modeling.
    Returns a dict {14: priceIn14Days, 30: placeholderPriceIn30Days}.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    df = hist_data_enriched.copy()
    horizon = 14

    df['target_14d'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)

    X = df.drop('target_14d', axis=1).select_dtypes(include=[np.number])
    y = df['target_14d']

    if len(X) < 20:
        # Not enough data
        return {14: float(df['Close'].iloc[-1]), 30: float(df['Close'].iloc[-1])}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5,
                             learning_rate=0.05, objective='reg:squarederror')
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              early_stopping_rounds=10, verbose=False)

    # Predict 14-day
    last_row = X.iloc[[-1]]
    forecast_14 = model.predict(last_row)[0]

    # Placeholder for 30-day
    # e.g. just add the std dev of Close for a naive offset
    forecast_30 = float(forecast_14) + float(df['Close'].std())

    return {14: float(forecast_14), 30: forecast_30}

def run_bidirectional_lstm(hist_data_enriched):
    """
    Example Bidirectional LSTM model for time-series forecasting.
    Returns {14: predictedPriceIn14Days, 30: placeholderFor30Days}.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    df = hist_data_enriched.copy()
    horizon = 14
    df['target'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)

    X = df.select_dtypes(include=[np.number]).drop('target', axis=1)
    y = df['target']

    if len(X) < 30:
        # Not enough data for any serious training
        current_price = float(df['Close'].iloc[-1])
        return {14: current_price, 30: current_price}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_lstm = np.expand_dims(X_train_scaled, axis=1)
    X_test_lstm  = np.expand_dims(X_test_scaled, axis=1)

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False),
                            input_shape=(1, X_train_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train_lstm, y_train, epochs=30, batch_size=32,
              validation_split=0.1, callbacks=[early_stopping], verbose=0)

    # Forecast 14 days
    last_X = X.iloc[[-1]].values
    last_X_scaled = scaler.transform(last_X)
    last_X_lstm   = np.expand_dims(last_X_scaled, axis=1)
    forecast_14   = model.predict(last_X_lstm)[0][0]

    # Simple placeholder for 30 days
    forecast_30 = forecast_14 + float(df['Close'].std())

    return {14: float(forecast_14), 30: float(forecast_30)}

def ensemble_predictions(model_forecasts_list):
    """
    Given multiple model-forecast dicts (e.g. [{14: x, 30: y}, {...}, ...]),
    average each horizon's predictions to form an ensemble.
    """
    horizons = [14, 30]
    final_forecasts = {}
    for h in horizons:
        preds = [mf[h] for mf in model_forecasts_list if h in mf]
        if len(preds) > 0:
            final_forecasts[h] = float(np.mean(preds))
        else:
            final_forecasts[h] = np.nan
    return final_forecasts

##############################################################################
# Now the rest of the original stock_forecast.py code
##############################################################################

def load_predictions_history():
    if os.path.exists(PREDICTION_HISTORY_FILE):
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_predictions_history(data):
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_stock_data_yfinance(ticker):
    stock = yf.Ticker(ticker)
    periods = ['5y', '3y', '1y', '6mo', '3mo', '1mo']
    for period in periods:
        hist = stock.history(period=period)
        if not hist.empty:
            return hist
    raise ValueError(f"No historical data found for ticker '{ticker}' using yfinance.")

def get_stock_data_alpha_vantage(ticker):
    """
    Fetch historical stock data using Alpha Vantage API.
    """
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'outputsize': 'full',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        if "Note" in data:
            raise Exception("Alpha Vantage API rate limit reached. Please wait and try again.")
        elif "Error Message" in data:
            raise ValueError(f"Error fetching data for ticker '{ticker}' from Alpha Vantage.")
        else:
            raise Exception("Unexpected response from Alpha Vantage.")

    time_series = data['Time Series (Daily)']
    records = []
    for date_str, daily_data in time_series.items():
        records.append({
            'Date': pd.to_datetime(date_str),
            'Open': float(daily_data['1. open']),
            'High': float(daily_data['2. high']),
            'Low': float(daily_data['3. low']),
            'Close': float(daily_data['4. close']),
            'Volume': int(daily_data['6. volume'])
        })
    df = pd.DataFrame(records)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

def get_stock_data(ticker):
    """
    Attempt to fetch stock data using yfinance. If it fails for any reason,
    fallback to Alpha Vantage.
    """
    try:
        hist = get_stock_data_yfinance(ticker)
        print(f"Fetched data for {ticker} using yfinance.")
        return hist
    except Exception as e:
        print(f"yfinance failed for {ticker}: {e}")
        print("Attempting to fetch data via Alpha Vantage...")
        try:
            hist = get_stock_data_alpha_vantage(ticker)
            print(f"Fetched data for {ticker} using Alpha Vantage.")
            return hist
        except Exception as backup_e:
            print(f"Alpha Vantage also failed for {ticker}: {backup_e}")
            raise backup_e

def get_financials_yfinance(ticker):
    stock = yf.Ticker(ticker)
    financials = {}
    try:
        income_stmt = stock.financials
        financials['revenue'] = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
        financials['net_income'] = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else None

        balance_sheet = stock.balance_sheet
        financials['total_assets'] = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
        financials['total_liabilities'] = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else None
        financials['shareholder_equity'] = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else None

        cash_flow = stock.cashflow
        financials['operating_cash_flow'] = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cash_flow.index else None

        stock_info = stock.info
        financials['eps']        = stock_info.get('trailingEps')
        financials['eps_growth'] = stock_info.get('earningsQuarterlyGrowth')
        financials['pe_ratio']   = stock_info.get('trailingPE')
        financials['pb_ratio']   = stock_info.get('priceToBook')
        financials['free_cash_flow'] = stock_info.get('freeCashflow')

        # Basic ratio checks
        if financials['net_income'] is not None and financials['shareholder_equity'] not in [None, 0]:
            financials['return_on_equity'] = financials['net_income'] / financials['shareholder_equity']
        else:
            financials['return_on_equity'] = None

        if financials['total_liabilities'] is not None and financials['shareholder_equity'] not in [None, 0]:
            financials['debt_to_equity'] = financials['total_liabilities'] / financials['shareholder_equity']
        else:
            financials['debt_to_equity'] = None

    except Exception as e:
        print(f"Error retrieving financials for {ticker} with yfinance: {e}")
        raise e
    return financials

def get_financials_alpha_vantage(ticker):
    """
    Fetch financial data using Alpha Vantage API.
    """
    params = {
        'function': 'OVERVIEW',
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
    data = response.json()

    if not data:
        raise ValueError(f"No financial data found for ticker '{ticker}' via Alpha Vantage.")

    required_fields = [
        'RevenueTTM', 'NetIncomeTTM', 'TotalAssets', 'TotalLiabilities',
        'ReturnOnEquityTTM', 'DebtToEquity', 'EarningsPerShare', 'EarningsGrowthTTM',
        'PERatio', 'PriceToBookRatio', 'FreeCashFlow'
    ]

    financials = {}
    for field in required_fields:
        val = data.get(field, None)
        financials[field.lower()] = float(val) if val is not None else None

    # Map fields
    mapped = {
        'revenue': financials.get('revenuettm'),
        'net_income': financials.get('netincomettm'),
        'total_assets': financials.get('totalassets'),
        'total_liabilities': financials.get('totalliabilities'),
        'shareholder_equity': (financials.get('totalassets') - financials.get('totalliabilities'))
                               if financials.get('totalassets') and financials.get('totalliabilities') else None,
        'operating_cash_flow': financials.get('freecashflow'),  # no direct in alpha vantage
        'eps':        financials.get('earningspershare'),
        'eps_growth': financials.get('earningsgrowthttm'),
        'pe_ratio':   financials.get('peratio'),
        'pb_ratio':   financials.get('pricetobookratio'),
        'free_cash_flow': financials.get('freecashflow'),
        'return_on_equity': financials.get('returnonequityttm'),
        'debt_to_equity':   financials.get('debtequity')
    }
    return mapped

def get_financials(ticker):
    """
    Attempt yfinance first; if that fails, fallback to Alpha Vantage.
    """
    try:
        fin = get_financials_yfinance(ticker)
        print(f"Fetched financials for {ticker} (yfinance).")
        return fin
    except Exception as e:
        print(f"yfinance failed for {ticker} financials: {e}")
        print("Trying alpha vantage for financials...")
        try:
            fin = get_financials_alpha_vantage(ticker)
            print(f"Fetched financials for {ticker} via Alpha Vantage.")
            return fin
        except Exception as e2:
            print(f"Alpha Vantage also failed for {ticker} financials: {e2}")
            raise e2

def fundamental_analysis(financials):
    score = 0
    max_score = 8

    roe = financials.get('return_on_equity')
    if roe is not None and roe > 0.15:
        score += 1

    dte = financials.get('debt_to_equity')
    if dte is not None and dte < 0.5:
        score += 1

    net_income = financials.get('net_income')
    if net_income is not None and net_income > 0:
        score += 1

    ocf = financials.get('operating_cash_flow')
    if ocf is not None and ocf > 0:
        score += 1

    ta = financials.get('total_assets')
    tl = financials.get('total_liabilities')
    if ta is not None and tl is not None and ta > tl:
        score += 1

    eps_growth = financials.get('eps_growth')
    if eps_growth is not None and eps_growth > 0:
        score += 1

    pe_ratio = financials.get('pe_ratio')
    if pe_ratio is not None and pe_ratio < 25:
        score += 1

    fcf = financials.get('free_cash_flow')
    if fcf is not None and fcf > 0:
        score += 1

    return (score / max_score) * 10

def calculate_stochastic_oscillator(hist_data, period=14, d_period=3):
    df = hist_data.copy()
    df['LowestLow'] = df['Low'].rolling(period).min()
    df['HighestHigh'] = df['High'].rolling(period).max()
    df['%K'] = ((df['Close'] - df['LowestLow']) / (df['HighestHigh'] - df['LowestLow'])) * 100
    df['%D'] = df['%K'].rolling(d_period).mean()
    return df['%K'].iloc[-1], df['%D'].iloc[-1]

def calculate_ichimoku(hist_data):
    df = hist_data.copy()
    nine_high = df['High'].rolling(9).max()
    nine_low  = df['Low'].rolling(9).min()
    df['Conversion'] = (nine_high + nine_low) / 2

    twenty_six_high = df['High'].rolling(26).max()
    twenty_six_low  = df['Low'].rolling(26).min()
    df['Base'] = (twenty_six_high + twenty_six_low) / 2

    df['SpanA'] = ((df['Conversion'] + df['Base']) / 2).shift(26)
    fifty_two_high = df['High'].rolling(52).max()
    fifty_two_low  = df['Low'].rolling(52).min()
    df['SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    conv = df['Conversion'].iloc[-1]
    base = df['Base'].iloc[-1]
    spanA= df['SpanA'].iloc[-26] if len(df) > 26 else np.nan
    spanB= df['SpanB'].iloc[-26] if len(df) > 26 else np.nan
    return conv, base, spanA, spanB

def calculate_fib_retracements(hist_data, lookback=60):
    recent_data = hist_data.tail(lookback)
    swing_high = recent_data['High'].max()
    swing_low  = recent_data['Low'].min()
    diff = swing_high - swing_low
    return {
        '0.236': swing_high - diff * 0.236,
        '0.382': swing_high - diff * 0.382,
        '0.5':   swing_high - diff * 0.5,
        '0.618': swing_high - diff * 0.618,
        '0.786': swing_high - diff * 0.786
    }

def add_advanced_technical_indicators(hist_data):
    k_val, d_val = calculate_stochastic_oscillator(hist_data)
    conv, base, spanA, spanB = calculate_ichimoku(hist_data)
    fib_levels = calculate_fib_retracements(hist_data)
    return k_val, d_val, conv, base, spanA, spanB, fib_levels

def calculate_atr(hist_data, period=14):
    df = hist_data.copy()
    df['Previous_Close'] = df['Close'].shift(1)
    df['H-L']  = df['High'] - df['Low']
    df['H-Pc'] = abs(df['High'] - df['Previous_Close'])
    df['L-Pc'] = abs(df['Low']  - df['Previous_Close'])
    df['TR']   = df[['H-L','H-Pc','L-Pc']].max(axis=1)
    df['ATR']  = df['TR'].rolling(window=period).mean()
    return df['ATR'].iloc[-1]

def buy_sell_levels(hist_data):
    lookback = 60
    recent_data = hist_data.tail(lookback)
    support = recent_data['Close'].min()
    resistance = recent_data['Close'].max()
    return {
        'support': support,
        'resistance': resistance,
        'buy_zone': (support, support * 1.02),
        'sell_zone': (resistance * 0.98, resistance)
    }

def get_sector_etf(sector):
    sector_map = {
        'Energy': 'XLE',
        'Financial Services': 'XLF',
        'Financial': 'XLF',
        'Communication Services': 'XLC',
        'Industrials': 'XLI',
        'Healthcare': 'XLV',
        'Health Care': 'XLV',
        'Information Technology': 'XLK',
        'Technology': 'XLK',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Consumer Discretionary': 'XLY',
        'Real Estate': 'XLRE',
        'Materials': 'XLB'
    }
    return sector_map.get(sector, None)

def get_market_condition_yfinance():
    spy = yf.Ticker("SPY").history(period='1y')
    if len(spy) < 200:
        return None
    spy['MA200'] = spy['Close'].rolling(200).mean()
    latest_close = spy['Close'].iloc[-1]
    ma200        = spy['MA200'].iloc[-1]
    return (latest_close > ma200)

def get_market_condition_alpha_vantage():
    """
    Attempt to read SPY from Alpha Vantage, then see if price > 200 MA
    """
    try:
        hist = get_stock_data_alpha_vantage("SPY")
    except Exception as e:
        print(f"Failed to fetch SPY from alpha vantage: {e}")
        return None
    if len(hist) < 200:
        return None
    hist['MA200'] = hist['Close'].rolling(200).mean()
    latest_close = hist['Close'].iloc[-1]
    ma200        = hist['MA200'].iloc[-1]
    return (latest_close > ma200)

def get_market_condition():
    try:
        cond = get_market_condition_yfinance()
        return cond
    except Exception as e:
        print(f"yfinance failed for market condition: {e}")
        print("Trying alpha vantage for SPY condition...")
        try:
            cond2 = get_market_condition_alpha_vantage()
            return cond2
        except Exception as e2:
            print(f"Alpha vantage also failed for market condition: {e2}")
            return None

def get_sector_condition_yfinance(sector_etf):
    if not sector_etf:
        return None
    try:
        etf_data = yf.Ticker(sector_etf).history(period='1y')
        if len(etf_data) < 200:
            return None
        etf_data['MA200'] = etf_data['Close'].rolling(200).mean()
        latest_close = etf_data['Close'].iloc[-1]
        ma200        = etf_data['MA200'].iloc[-1]
        return (latest_close > ma200)
    except Exception as e:
        print(f"Failed yfinance for {sector_etf}: {e}")
        return None

def get_sector_condition_alpha_vantage(sector_etf):
    if not sector_etf:
        return None
    try:
        hist = get_stock_data_alpha_vantage(sector_etf)
        if len(hist) < 200:
            return None
        hist['MA200'] = hist['Close'].rolling(200).mean()
        latest_close = hist['Close'].iloc[-1]
        ma200        = hist['MA200'].iloc[-1]
        return (latest_close > ma200)
    except Exception as e:
        print(f"Failed alpha vantage for {sector_etf}: {e}")
        return None

def get_sector_condition(sector_etf):
    try:
        cond = get_sector_condition_yfinance(sector_etf)
        return cond
    except Exception as e:
        print(f"yfinance failed for {sector_etf} condition: {e}")
        print("Trying alpha vantage for sector ETF condition...")
        try:
            cond2 = get_sector_condition_alpha_vantage(sector_etf)
            return cond2
        except Exception as e2:
            print(f"Alpha vantage also failed for {sector_etf} condition: {e2}")
            return None

def adjust_probabilities(prob_up, prob_down, market_bullish, sector_bullish):
    if market_bullish is True and sector_bullish is True:
        prob_up   *= 1.1
        prob_down *= 0.9
    elif market_bullish is False and sector_bullish is False:
        prob_up   *= 0.9
        prob_down *= 1.1

    total = prob_up + prob_down
    if total > 0:
        prob_up   = (prob_up   / total) * 100.0
        prob_down = (prob_down / total) * 100.0
    else:
        # fallback
        prob_up   = 50.0
        prob_down = 50.0

    return prob_up, prob_down

def calculate_stochastic_signal(k_val, d_val):
    # Example: bullish if %K < 80, %K > %D, etc. This is just a sample approach.
    return (20 < k_val < 80) and (k_val > d_val)

def calculate_ichimoku_signal(hist_data):
    conv, base, spanA, spanB = calculate_ichimoku(hist_data)
    latest_close = hist_data['Close'].iloc[-1]
    if not np.isnan(spanA) and not np.isnan(spanB):
        cloud_top = max(spanA, spanB)
        return (latest_close > cloud_top)
    return False

def calculate_fib_signal(hist_data, fib_levels):
    latest_close = hist_data['Close'].iloc[-1]
    for lvl in fib_levels.values():
        if abs((latest_close - lvl)/lvl) < 0.01:
            return True
    return False

def add_more_technical_signals(hist_data, fib_levels):
    stoch_sig = calculate_stochastic_signal(*calculate_stochastic_oscillator(hist_data))
    ichi_sig  = calculate_ichimoku_signal(hist_data)
    fib_sig   = calculate_fib_signal(hist_data, fib_levels)
    return (stoch_sig, ichi_sig, fib_sig)

def technical_analysis(hist_data):
    score = 0
    max_score = 14

    hist_data['MA50']  = hist_data['Close'].rolling(50).mean()
    hist_data['MA200'] = hist_data['Close'].rolling(200).mean()
    latest_close = hist_data['Close'].iloc[-1]
    ma50         = hist_data['MA50'].iloc[-1]
    ma200        = hist_data['MA200'].iloc[-1]

    if latest_close > ma50:  score += 1
    if latest_close > ma200: score += 1
    if ma50 > ma200:         score += 1

    delta = hist_data['Close'].diff(1).dropna()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    period= 14
    gain  = up.rolling(period).mean()
    loss  = abs(down.rolling(period).mean())
    RS    = gain / loss
    RSI   = 100.0 - (100.0 / (1.0 + RS))
    latest_rsi = RSI.iloc[-1]
    if latest_rsi < 70: score += 1
    if latest_rsi > 50: score += 1

    exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
    hist_data['MACD'] = exp1 - exp2
    hist_data['Signal_Line'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
    if hist_data['MACD'].iloc[-1] > hist_data['Signal_Line'].iloc[-1]:
        score += 1

    hist_data['20_SMA'] = hist_data['Close'].rolling(20).mean()
    hist_data['STDDEV'] = hist_data['Close'].rolling(20).std()
    hist_data['Upper_Band'] = hist_data['20_SMA'] + (hist_data['STDDEV']*2)
    hist_data['Lower_Band'] = hist_data['20_SMA'] - (hist_data['STDDEV']*2)
    if latest_close < hist_data['Upper_Band'].iloc[-1]:
        score += 1

    hist_data['OBV'] = (np.sign(hist_data['Close'].diff()) * hist_data['Volume']).fillna(0).cumsum()
    if hist_data['OBV'].iloc[-1] > hist_data['OBV'].rolling(20).mean().iloc[-1]:
        score += 1

    # Extra advanced signals
    k_val, d_val, conv, base, spanA, spanB, fib_levels = add_advanced_technical_indicators(hist_data)
    if calculate_stochastic_signal(k_val, d_val):
        score += 1
    if calculate_ichimoku_signal(hist_data):
        score += 1
    if calculate_fib_signal(hist_data, fib_levels):
        score += 1

    more_signals = add_more_technical_signals(hist_data, fib_levels)
    for s in more_signals:
        if s:
            score += 1

    return (score / max_score)*10

##############################################################################
# High-level Forecasting & Probability
##############################################################################

def analyze_stock(ticker):
    """
    Main entry point: fetch data, fetch financials, create advanced features,
    run fundamental & technical analyses, then forecast with ARIMA, XGB, LSTM,
    and produce ensemble.
    """
    try:
        hist_data = get_stock_data(ticker)
    except ValueError as e:
        print(e)
        return None

    try:
        financials = get_financials(ticker)
    except Exception as e:
        print(f"Failed to retrieve financials for {ticker}: {e}")
        return None

    # Build advanced features
    hist_data_enriched = create_advanced_features(hist_data, ticker, financials)

    # Fundamental & Technical
    fundamental_score = fundamental_analysis(financials)
    technical_score   = technical_analysis(hist_data)

    # Forecasts from ARIMA, XGBoost, LSTM
    forecasts_linear = run_arima_model(hist_data['Close'])
    forecasts_xgb    = run_xgboost_model(hist_data_enriched)
    forecasts_bilstm = run_bidirectional_lstm(hist_data_enriched)

    # Ensemble
    forecasts_ensemble = ensemble_predictions([forecasts_linear, forecasts_xgb, forecasts_bilstm])

    current_price = hist_data['Close'].iloc[-1]
    results = {
        'ticker':    ticker,
        'current_price': float(current_price),
        'forecasts': forecasts_ensemble,
        'expected_return_percentages': {},
        'likelihood_score': 0.0,
        'hist_data': hist_data,  # in case you want to use it later
        'financials': financials
    }

    # Fill in return percentages
    for days_ahead, fprice in forecasts_ensemble.items():
        pct_inc = 100.0 * ((fprice - current_price)/current_price)
        results['expected_return_percentages'][days_ahead] = pct_inc

    # Weighted overall score
    overall_score = (0.6 * fundamental_score) + (0.4 * technical_score)
    results['likelihood_score'] = overall_score
    return results

def display_analysis_results(result):
    """
    Nicely print out results from analyze_stock.
    """
    if not result:
        print("Analysis could not be completed.")
        return

    print(f"\nForecast for {result['ticker']}:")
    print(f"Current Price: ${result['current_price']:.2f}")
    for days_ahead in sorted(result['forecasts'].keys()):
        fprice   = result['forecasts'][days_ahead]
        expret   = result['expected_return_percentages'][days_ahead]
        print(f"Forecast Price ({days_ahead} days): ${fprice:.2f}")
        print(f"Expected Return ({days_ahead} days): {expret:.2f}%")

    ls = result['likelihood_score']
    print(f"Likelihood Score: {ls:.2f}/10")

    # Simple example recommendation
    if ls >= 7:
        print("Recommendation: Consider entering a trade.")
        cp = result['current_price']
        stop_loss     = cp * 0.95
        profit_target = cp + (cp - stop_loss)*3
        print(f"Suggested Entry Price: ${cp:.2f}")
        print(f"Suggested Profit Target: ${profit_target:.2f}")
        print(f"Suggested Stop Loss: ${stop_loss:.2f}")
    else:
        print("Recommendation: The stock does not meet the criteria for entry at this time.")

def save_to_history(result):
    """
    Append analysis results to a 'history.log' file.
    """
    with open('history.log', 'a') as f:
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Ticker: {result['ticker']}\n")
        f.write(f"Current Price: ${result['current_price']:.2f}\n")
        for d, fp in result['forecasts'].items():
            expret = result['expected_return_percentages'][d]
            f.write(f"Forecast ({d} days): ${fp:.2f}\n")
            f.write(f"Expected Return ({d} days): {expret:.2f}%\n")
        f.write(f"Likelihood Score: {result['likelihood_score']:.2f}/10\n")
        f.write("-"*50 + "\n")

def predict_probability(hist_data, ticker):
    """
    For a given hist_data & ticker, attempt to compute the probability that
    bullish vs. bearish ATR-based targets get hit first, adjusting by market
    & sector conditions.
    """
    atr_value = calculate_atr(hist_data)
    if np.isnan(atr_value) or atr_value == 0:
        return None

    current_price = hist_data['Close'].iloc[-1]
    bullish_1ATR = current_price + atr_value
    bearish_1ATR = current_price - atr_value
    bullish_3ATR = current_price + 3*atr_value
    bearish_3ATR = current_price - 3*atr_value
    bullish_5ATR = current_price + 5*atr_value
    bearish_5ATR = current_price - 5*atr_value

    df = hist_data.copy()
    df['Next_High'] = df['High'].shift(-1)
    df['Next_Low']  = df['Low']. shift(-1)
    valid_data = df.dropna(subset=['Next_High','Next_Low'])
    if len(valid_data) == 0:
        return None

    def prob_move(factor):
        up_move = (valid_data['Next_High'] > valid_data['Close'] + factor*atr_value).mean()*100.0
        dn_move = (valid_data['Next_Low']  < valid_data['Close'] - factor*atr_value).mean()*100.0
        return up_move, dn_move

    prob_up_1,  prob_down_1  = prob_move(1)
    prob_up_3,  prob_down_3  = prob_move(3)
    prob_up_5,  prob_down_5  = prob_move(5)

    # Check sector
    stock = yf.Ticker(ticker)
    sector = stock.info.get('sector', None)
    sector_etf = get_sector_etf(sector)

    market_cond = get_market_condition()
    sector_cond = get_sector_condition(sector_etf)

    prob_up_1,  prob_down_1  = adjust_probabilities(prob_up_1,  prob_down_1,  market_cond, sector_cond)
    prob_up_3,  prob_down_3  = adjust_probabilities(prob_up_3,  prob_down_3,  market_cond, sector_cond)
    prob_up_5,  prob_down_5  = adjust_probabilities(prob_up_5,  prob_down_5,  market_cond, sector_cond)

    def likely_label(u, d):
        if u > d:
            return "Bullish target more likely first"
        elif d > u:
            return "Bearish target more likely first"
        else:
            return "Both equally likely"

    likely_1ATR = likely_label(prob_up_1, prob_down_1)
    likely_3ATR = likely_label(prob_up_3, prob_down_3)
    likely_5ATR = likely_label(prob_up_5, prob_down_5)

    total_prob_up   = prob_up_1 + prob_up_3 + prob_up_5
    total_prob_down = prob_down_1 + prob_down_3 + prob_down_5
    if total_prob_up > total_prob_down:
        overall_prediction = "Bullish"
    elif total_prob_down > total_prob_up:
        overall_prediction = "Bearish"
    else:
        overall_prediction = "Neutral"

    # Save to predictions_history
    predictions_history = load_predictions_history()
    if ticker not in predictions_history:
        predictions_history[ticker] = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'queried_price': float(current_price),
            'bullish_1ATR': float(bullish_1ATR),
            'bearish_1ATR': float(bearish_1ATR),
            'bullish_3ATR': float(bullish_3ATR),
            'bearish_3ATR': float(bearish_3ATR),
            'bullish_5ATR': float(bullish_5ATR),
            'bearish_5ATR': float(bearish_5ATR),
            'overall_prediction': overall_prediction
        }
        save_predictions_history(predictions_history)

    return {
        'prob_up_1':     prob_up_1,
        'prob_down_1':   prob_down_1,
        'bullish_1ATR':  bullish_1ATR,
        'bearish_1ATR':  bearish_1ATR,
        'likely_1ATR':   likely_1ATR,

        'prob_up_3':     prob_up_3,
        'prob_down_3':   prob_down_3,
        'bullish_3ATR':  bullish_3ATR,
        'bearish_3ATR':  bearish_3ATR,
        'likely_3ATR':   likely_3ATR,

        'prob_up_5':     prob_up_5,
        'prob_down_5':   prob_down_5,
        'bullish_5ATR':  bullish_5ATR,
        'bearish_5ATR':  bearish_5ATR,
        'likely_5ATR':   likely_5ATR,

        'overall_prediction': overall_prediction
    }

def prediction_history_cli():
    """
    Simple urwid UI showing the ATR-based predictions we saved in predictions_history.json
    """
    predictions_history = load_predictions_history()
    if not predictions_history:
        print("No Prediction History found. Query 'Predict Probability' first.")
        input("\nPress Enter to return to the main menu...")
        return

    palette = [
        ('titlebar', 'white,bold', ''),
        ('refresh button', 'dark green,bold', ''),
        ('delete button', 'dark cyan,bold', ''),
        ('quit button', 'dark red', ''),
        ('headers', 'light blue,bold', ''),
        ('body', 'white', ''),
        ('footer', 'white,bold', ''),
        ('hit', 'white,bold', 'dark green'),
        ('nohit', 'white', ''),
        ('bullish_col', 'dark green,bold', ''),
        ('bearish_col', 'dark red,bold', ''),
        ('overall_bullish', 'dark green,bold', ''),
        ('overall_bearish', 'dark red,bold', ''),
        ('overall_neutral', 'yellow,bold', '')
    ]

    header_text = urwid.Text(u' Prediction History', align='center')
    header = urwid.AttrMap(header_text, 'titlebar')

    menu = urwid.Text([
        u'Press (', ('refresh button', u'R'), u') to refresh. ',
        u'Press (', ('delete button', u'D'), u') to delete tickers that have hit 5th ATR. ',
        u'Press (', ('quit button', u'Q'), u') to return to main menu.'
    ])
    footer = urwid.AttrMap(menu, 'footer')

    body_text = urwid.Text(u'Fetching prediction history...', align='left')
    body_filler = urwid.Filler(body_text, valign='top')
    body_padding = urwid.Padding(body_filler, left=1, right=1)
    body = urwid.AttrMap(body_padding, 'body')
    layout = urwid.Frame(header=header, body=body, footer=footer)

    columns = [
        ('ticker',             'Ticker',           10),
        ('queried_price',      'Queried Price',    15),
        ('bullish_1ATR',       'Bullish 1',        10),
        ('bearish_1ATR',       'Bearish 1',        10),
        ('bullish_3ATR',       'Bullish 3',        10),
        ('bearish_3ATR',       'Bearish 3',        10),
        ('bullish_5ATR',       'Bullish 5',        10),
        ('bearish_5ATR',       'Bearish 5',        10),
        ('overall_prediction', 'Overall Prediction',20),
        ('status',             'Status',           15)
    ]

    def get_prediction_data():
        updates = []
        header_line = []
        for key, hdr, width in columns:
            header_line.append(('headers', hdr.ljust(width)))
        updates.extend(header_line)
        updates.append(('', '\n'))

        for ticker, info in predictions_history.items():
            qp  = info['queried_price']
            b1  = info['bullish_1ATR']
            be1 = info['bearish_1ATR']
            b3  = info['bullish_3ATR']
            be3 = info['bearish_3ATR']
            b5  = info['bullish_5ATR']
            be5 = info['bearish_5ATR']
            overall_pred = info.get('overall_prediction', 'Neutral')

            # Current price
            try:
                st = yf.Ticker(ticker)
                sd = st.history(period="1d")
                if sd.empty:
                    current_price = qp
                else:
                    current_price = sd['Close'].iloc[-1]
            except:
                current_price = qp

            # Check 5th ATR
            if current_price > b5:
                status = 'Bullish 5 ATR Hit'
                status_attr = 'hit'
            elif current_price < be5:
                status = 'Bearish 5 ATR Hit'
                status_attr = 'hit'
            else:
                status = 'In Progress'
                status_attr = 'nohit'

            line_data = {
                'ticker': ticker,
                'queried_price':  f'${qp:.2f}',
                'bullish_1ATR':   f'${b1:.2f}',
                'bearish_1ATR':   f'${be1:.2f}',
                'bullish_3ATR':   f'${b3:.2f}',
                'bearish_3ATR':   f'${be3:.2f}',
                'bullish_5ATR':   f'${b5:.2f}',
                'bearish_5ATR':   f'${be5:.2f}',
                'overall_prediction': overall_pred,
                'status': status
            }

            if overall_pred == "Bullish":
                overall_color = 'overall_bullish'
            elif overall_pred == "Bearish":
                overall_color = 'overall_bearish'
            else:
                overall_color = 'overall_neutral'

            line = []
            for colkey, hdr, width in columns:
                if 'bullish' in colkey:
                    col_attr = 'bullish_col'
                elif 'bearish' in colkey:
                    col_attr = 'bearish_col'
                elif colkey == 'overall_prediction':
                    if overall_pred == "Bullish":
                        col_attr = 'overall_bullish'
                    elif overall_pred == "Bearish":
                        col_attr = 'overall_bearish'
                    else:
                        col_attr = 'overall_neutral'
                else:
                    col_attr = status_attr

                val = line_data[colkey]
                line.append((col_attr, val.ljust(width)))

            line.append(('', '\n'))
            updates.extend(line)

        return updates

        # End get_prediction_data

    def handle_input(key):
        if key in ('R','r'):
            refresh()
        elif key in ('D','d'):
            delete_hit_tickers()
            refresh()
        elif key in ('Q','q'):
            raise urwid.ExitMainLoop()

    def refresh():
        body_text.set_text(get_prediction_data())

    def delete_hit_tickers():
        nonlocal predictions_history
        to_delete = []
        for tck, info in predictions_history.items():
            b5 = info['bullish_5ATR']
            be5= info['bearish_5ATR']
            try:
                st = yf.Ticker(tck)
                sd = st.history(period="1d")
                if sd.empty:
                    cp = info['queried_price']
                else:
                    cp = sd['Close'].iloc[-1]
            except:
                cp = info['queried_price']

            if cp > b5 or cp < be5:
                to_delete.append(tck)

        if to_delete:
            print(f"\nTickers that hit 5th ATR: {', '.join(to_delete)}")
            confirm = input("Remove these tickers from Prediction History? (yes/no): ").lower()
            if confirm == 'yes':
                for t in to_delete:
                    del predictions_history[t]
                save_predictions_history(predictions_history)
                print("Tickers removed from Prediction History.")
            else:
                print("Deletion canceled.")
        else:
            print("No tickers have hit the 5th ATR target.")

    refresh()
    main_loop = urwid.MainLoop(layout, palette, unhandled_input=handle_input)
    main_loop.run()

def view_history():
    """
    Simple text-based viewer for 'history.log' with minimal menu.
    """
    if not os.path.exists('history.log'):
        print("\nNo history found.")
        return

    while True:
        print("\n--- View Analysis History ---")
        print("1. View History")
        print("2. Save History to a Text File")
        print("3. Clear History")
        print("4. Return to Main Menu")
        choice = input("Please select an option (1-4): ")

        if choice == '1':
            with open('history.log','r') as f:
                print(f.read())

        elif choice == '2':
            fname = input("Enter filename (e.g. my_history.txt): ")
            try:
                with open('history.log','r') as src:
                    data = src.read()
                with open(fname,'w') as dst:
                    dst.write(data)
                print(f"History saved to {fname}")
            except Exception as e:
                print(f"Error saving history: {e}")

        elif choice == '3':
            confirm = input("Are you sure you want to clear the history? (yes/no): ").lower()
            if confirm == 'yes':
                os.remove('history.log')
                print("History cleared.")
                break
            else:
                print("Clear history cancelled.")

        elif choice == '4':
            break
        else:
            print("Invalid option. Please try again.")

def main_menu():
    """
    The main user-facing CLI loop for stock forecasting (unchanged).
    """
    while True:
        print("\n--- Stock Forecasting Tool ---")
        print("1. Analyze a Stock")
        print("2. Buy/Sell Levels")
        print("3. Predict Probability")
        print("4. View Analysis History")
        print("5. View Prediction History")
        print("6. Exit")
        choice = input("Please select an option (1-6): ")

        if choice == '1':
            ticker = input("Enter stock ticker symbol: ").upper()
            result = analyze_stock(ticker)
            display_analysis_results(result)
            if result:
                save_to_history(result)

        elif choice == '2':
            ticker = input("Enter stock ticker symbol: ").upper()
            try:
                hist_data = get_stock_data(ticker)
                levels = buy_sell_levels(hist_data)
                print("\n--- Buy/Sell Levels ---")
                print(f"Support Level: ${levels['support']:.2f}")
                print(f"Resistance Level: ${levels['resistance']:.2f}")
                print(f"Buy Zone: ${levels['buy_zone'][0]:.2f} - ${levels['buy_zone'][1]:.2f}")
                print(f"Sell Zone: ${levels['sell_zone'][0]:.2f} - ${levels['sell_zone'][1]:.2f}")
            except Exception as e:
                print(f"Error retrieving buy/sell levels for {ticker}: {e}")

        elif choice == '3':
            ticker = input("Enter the stock ticker symbol: ").upper()
            try:
                hist_data = get_stock_data(ticker)
                prob = predict_probability(hist_data, ticker)
                if prob is not None:
                    print("\n--- Predict Probability ---")
                    print("1 ATR Move:")
                    print(f"Bullish (1 ATR): ${prob['bullish_1ATR']:.2f}")
                    print(f"Bearish (1 ATR): ${prob['bearish_1ATR']:.2f}")
                    print(f"Prob. Bullish first (1 ATR): {prob['prob_up_1']:.2f}%")
                    print(f"Prob. Bearish first (1 ATR): {prob['prob_down_1']:.2f}%")
                    print(prob['likely_1ATR'])

                    print("\n3 ATR Move:")
                    print(f"Bullish (3 ATR): ${prob['bullish_3ATR']:.2f}")
                    print(f"Bearish (3 ATR): ${prob['bearish_3ATR']:.2f}")
                    print(f"Prob. Bullish first (3 ATR): {prob['prob_up_3']:.2f}%")
                    print(f"Prob. Bearish first (3 ATR): {prob['prob_down_3']:.2f}%")
                    print(prob['likely_3ATR'])

                    print("\n5 ATR Move:")
                    print(f"Bullish (5 ATR): ${prob['bullish_5ATR']:.2f}")
                    print(f"Bearish (5 ATR): ${prob['bearish_5ATR']:.2f}")
                    print(f"Prob. Bullish first (5 ATR): {prob['prob_up_5']:.2f}%")
                    print(f"Prob. Bearish first (5 ATR): {prob['prob_down_5']:.2f}%")
                    print(prob['likely_5ATR'])

                    print(f"\nOverall Prediction: {prob['overall_prediction']}")
                else:
                    print("Not enough data to compute probability.")
            except Exception as e:
                print(f"Error predicting probability for {ticker}: {e}")

        elif choice == '4':
            view_history()

        elif choice == '5':
            prediction_history_cli()

        elif choice == '6':
            print("Exiting the tool. Goodbye!")
            break
        else:
            print("Invalid option selected. Please try again.")


if __name__ == "__main__":
    main_menu()
