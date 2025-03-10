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

# Import advanced modeling tools from advanced_modeling.py
import advanced_modeling as am

import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Configuration for Alpha Vantage API
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_API_KEY')  # Replace with your actual API key or set as an environment variable
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

PREDICTION_HISTORY_FILE = 'predictions_history.json'

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
    Attempt to fetch stock data using yfinance. If it fails due to rate limits or other issues, fallback to Alpha Vantage.
    """
    try:
        hist = get_stock_data_yfinance(ticker)
        print(f"Fetched data for {ticker} using yfinance.")
        return hist
    except Exception as e:
        print(f"yfinance failed to fetch data for {ticker}: {e}")
        print("Attempting to fetch data using Alpha Vantage...")
        try:
            hist = get_stock_data_alpha_vantage(ticker)
            print(f"Fetched data for {ticker} using Alpha Vantage.")
            return hist
        except Exception as backup_e:
            print(f"Alpha Vantage failed to fetch data for {ticker}: {backup_e}")
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
        financials['eps'] = stock_info.get('trailingEps', None)
        financials['eps_growth'] = stock_info.get('earningsQuarterlyGrowth', None)
        financials['pe_ratio'] = stock_info.get('trailingPE', None)
        financials['pb_ratio'] = stock_info.get('priceToBook', None)
        financials['free_cash_flow'] = stock_info.get('freeCashflow', None)

        if financials['net_income'] is not None and financials['shareholder_equity'] not in [None, 0]:
            financials['return_on_equity'] = financials['net_income'] / financials['shareholder_equity']
        else:
            financials['return_on_equity'] = None

        if financials['total_liabilities'] is not None and financials['shareholder_equity'] not in [None, 0]:
            financials['debt_to_equity'] = financials['total_liabilities'] / financials['shareholder_equity']
        else:
            financials['debt_to_equity'] = None

    except Exception as e:
        print(f"Error retrieving financials for {ticker} using yfinance: {e}")
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
        raise ValueError(f"No financial data found for ticker '{ticker}' using Alpha Vantage.")

    required_fields = [
        'RevenueTTM', 'NetIncomeTTM', 'TotalAssets', 'TotalLiabilities',
        'ReturnOnEquityTTM', 'DebtToEquity', 'EarningsPerShare', 'EarningsGrowthTTM',
        'PERatio', 'PriceToBookRatio', 'FreeCashFlow'
    ]

    financials = {}
    for field in required_fields:
        financials[field.lower()] = float(data.get(field, np.nan)) if data.get(field, None) is not None else None

    # Map Alpha Vantage fields to your existing financials keys
    mapped_financials = {
        'revenue': financials.get('revenuettm'),
        'net_income': financials.get('netincomettm'),
        'total_assets': financials.get('totalassets'),
        'total_liabilities': financials.get('totalliabilities'),
        'shareholder_equity': financials.get('totalassets') - financials.get('totalliabilities') if financials.get('totalassets') and financials.get('totalliabilities') else None,
        'operating_cash_flow': financials.get('freecashflow'),  # Alpha Vantage doesn't provide operating cash flow directly
        'eps': financials.get('earningspershare'),
        'eps_growth': financials.get('earningsgrowthttm'),
        'pe_ratio': financials.get('peratio'),
        'pb_ratio': financials.get('pricetobookratio'),
        'free_cash_flow': financials.get('freecashflow'),
        'return_on_equity': financials.get('returnonequityttm'),
        'debt_to_equity': financials.get('debtequity')
    }

    return mapped_financials

def get_financials(ticker):
    """
    Attempt to fetch financials using yfinance. If it fails, fallback to Alpha Vantage.
    """
    try:
        financials = get_financials_yfinance(ticker)
        print(f"Fetched financials for {ticker} using yfinance.")
        return financials
    except Exception as e:
        print(f"yfinance failed to fetch financials for {ticker}: {e}")
        print("Attempting to fetch financials using Alpha Vantage...")
        try:
            financials = get_financials_alpha_vantage(ticker)
            print(f"Fetched financials for {ticker} using Alpha Vantage.")
            return financials
        except Exception as backup_e:
            print(f"Alpha Vantage failed to fetch financials for {ticker}: {backup_e}")
            raise backup_e

def fundamental_analysis(financials):
    score = 0
    max_score = 8

    roe = financials.get('return_on_equity', None)
    if roe is not None and roe > 0.15:
        score += 1

    dte = financials.get('debt_to_equity', None)
    if dte is not None and dte < 0.5:
        score += 1

    net_income = financials.get('net_income', None)
    if net_income is not None and net_income > 0:
        score += 1

    operating_cash_flow = financials.get('operating_cash_flow', None)
    if operating_cash_flow is not None and operating_cash_flow > 0:
        score += 1

    total_assets = financials.get('total_assets', None)
    total_liabilities = financials.get('total_liabilities', None)
    if total_assets is not None and total_liabilities is not None and total_assets > total_liabilities:
        score += 1

    eps_growth = financials.get('eps_growth', None)
    if eps_growth is not None and eps_growth > 0:
        score += 1

    pe_ratio = financials.get('pe_ratio', None)
    if pe_ratio is not None and pe_ratio < 25:
        score += 1

    free_cash_flow = financials.get('free_cash_flow', None)
    if free_cash_flow is not None and free_cash_flow > 0:
        score += 1

    fundamental_score = (score / max_score) * 10
    return fundamental_score

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
    nine_low = df['Low'].rolling(9).min()
    df['Conversion'] = (nine_high + nine_low) / 2

    twenty_six_high = df['High'].rolling(26).max()
    twenty_six_low = df['Low'].rolling(26).min()
    df['Base'] = (twenty_six_high + twenty_six_low) / 2

    df['SpanA'] = ((df['Conversion'] + df['Base']) / 2).shift(26)
    fifty_two_high = df['High'].rolling(52).max()
    fifty_two_low = df['Low'].rolling(52).min()
    df['SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    return df['Conversion'].iloc[-1], df['Base'].iloc[-1], df['SpanA'].iloc[-26] if len(df) > 26 else np.nan, df['SpanB'].iloc[-26] if len(df) > 26 else np.nan

def calculate_fib_retracements(hist_data, lookback=60):
    recent_data = hist_data.tail(lookback)
    swing_high = recent_data['High'].max()
    swing_low = recent_data['Low'].min()
    diff = swing_high - swing_low
    levels = {
        '0.236': swing_high - diff * 0.236,
        '0.382': swing_high - diff * 0.382,
        '0.5': swing_high - diff * 0.5,
        '0.618': swing_high - diff * 0.618,
        '0.786': swing_high - diff * 0.786
    }
    return levels

def add_advanced_technical_indicators(hist_data):
    k_val, d_val = calculate_stochastic_oscillator(hist_data)
    conv, base, spanA, spanB = calculate_ichimoku(hist_data)
    fib_levels = calculate_fib_retracements(hist_data)
    return k_val, d_val, conv, base, spanA, spanB, fib_levels

def calculate_atr(hist_data, period=14):
    df = hist_data.copy()
    df['Previous_Close'] = df['Close'].shift(1)
    df['H-L'] = df['High'] - df['Low']
    df['H-Pc'] = abs(df['High'] - df['Previous_Close'])
    df['L-Pc'] = abs(df['Low'] - df['Previous_Close'])
    df['TR'] = df[['H-L','H-Pc','L-Pc']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df['ATR'].iloc[-1]

def buy_sell_levels(hist_data):
    lookback = 60
    recent_data = hist_data.tail(lookback)
    close_prices = recent_data['Close']

    support = close_prices.min()
    resistance = close_prices.max()

    buy_zone_low = support
    buy_zone_high = support * 1.02
    sell_zone_low = resistance * 0.98
    sell_zone_high = resistance

    return {
        'support': support,
        'resistance': resistance,
        'buy_zone': (buy_zone_low, buy_zone_high),
        'sell_zone': (sell_zone_low, sell_zone_high)
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
    spy['MA200'] = spy['Close'].rolling(200).mean()
    if len(spy) < 200:
        return None
    latest_close = spy['Close'].iloc[-1]
    ma200 = spy['MA200'].iloc[-1]
    return latest_close > ma200

def get_market_condition_alpha_vantage():
    """
    Determine market condition using Alpha Vantage for SPY.
    """
    ticker = "SPY"
    try:
        hist = am.get_stock_data_alpha_vantage(ticker)
    except Exception as e:
        print(f"Failed to fetch SPY data from Alpha Vantage: {e}")
        return None

    hist['MA200'] = hist['Close'].rolling(200).mean()
    if len(hist) < 200:
        return None
    latest_close = hist['Close'].iloc[-1]
    ma200 = hist['MA200'].iloc[-1]
    return latest_close > ma200

def get_market_condition():
    """
    Attempt to get market condition using yfinance, fallback to Alpha Vantage.
    """
    try:
        condition = get_market_condition_yfinance()
        return condition
    except Exception as e:
        print(f"yfinance failed to get market condition: {e}")
        print("Attempting to get market condition using Alpha Vantage...")
        try:
            condition = get_market_condition_alpha_vantage()
            return condition
        except Exception as backup_e:
            print(f"Alpha Vantage failed to get market condition: {backup_e}")
            return None

def get_sector_condition_yfinance(sector_etf):
    if sector_etf is None:
        return None
    try:
        etf = yf.Ticker(sector_etf)
        etf_data = etf.history(period='1y')
    except Exception as e:
        print(f"yfinance failed to fetch sector ETF data for {sector_etf}: {e}")
        return None

    if len(etf_data) < 200:
        return None
    etf_data['MA200'] = etf_data['Close'].rolling(200).mean()
    latest_close = etf_data['Close'].iloc[-1]
    ma200 = etf_data['MA200'].iloc[-1]
    return latest_close > ma200

def get_sector_condition_alpha_vantage(sector_etf):
    """
    Determine sector condition using Alpha Vantage for the given sector ETF.
    """
    if sector_etf is None:
        return None
    try:
        hist = am.get_stock_data_alpha_vantage(sector_etf)
    except Exception as e:
        print(f"Failed to fetch sector ETF data for {sector_etf} from Alpha Vantage: {e}")
        return None

    hist['MA200'] = hist['Close'].rolling(200).mean()
    if len(hist) < 200:
        return None
    latest_close = hist['Close'].iloc[-1]
    ma200 = hist['MA200'].iloc[-1]
    return latest_close > ma200

def get_sector_condition(sector_etf):
    """
    Attempt to get sector condition using yfinance, fallback to Alpha Vantage.
    """
    try:
        condition = get_sector_condition_yfinance(sector_etf)
        return condition
    except Exception as e:
        print(f"yfinance failed to get sector condition for {sector_etf}: {e}")
        print("Attempting to get sector condition using Alpha Vantage...")
        try:
            condition = get_sector_condition_alpha_vantage(sector_etf)
            return condition
        except Exception as backup_e:
            print(f"Alpha Vantage failed to get sector condition for {sector_etf}: {backup_e}")
            return None

def adjust_probabilities(prob_up, prob_down, market_bullish, sector_bullish):
    if market_bullish is True and sector_bullish is True:
        prob_up *= 1.1
        prob_down *= 0.9
    elif market_bullish is False and sector_bullish is False:
        prob_up *= 0.9
        prob_down *= 1.1

    total = prob_up + prob_down
    if total > 0:
        prob_up = (prob_up / total) * 100.0
        prob_down = (prob_down / total) * 100.0
    else:
        prob_up = 50.0
        prob_down = 50.0

    return prob_up, prob_down

def calculate_stochastic_signal(k_val, d_val):
    return 20 < k_val < 80 and k_val > d_val

def calculate_ichimoku_signal(hist_data):
    conv, base, spanA, spanB = calculate_ichimoku(hist_data)
    latest_close = hist_data['Close'].iloc[-1]
    if not np.isnan(spanA) and not np.isnan(spanB):
        cloud_top = max(spanA, spanB)
        return latest_close > cloud_top
    return False

def calculate_fib_signal(hist_data, fib_levels):
    latest_close = hist_data['Close'].iloc[-1]
    for level in fib_levels.values():
        if abs((latest_close - level) / level) < 0.01:
            return True
    return False

def add_more_technical_signals(hist_data, fib_levels):
    stochastic_signal = calculate_stochastic_signal(*calculate_stochastic_oscillator(hist_data))
    ichimoku_signal = calculate_ichimoku_signal(hist_data)
    fib_signal = calculate_fib_signal(hist_data, fib_levels)
    return stochastic_signal, ichimoku_signal, fib_signal

def technical_analysis(hist_data):
    score = 0
    max_score = 14

    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
    latest_close = hist_data['Close'].iloc[-1]
    ma50 = hist_data['MA50'].iloc[-1]
    ma200 = hist_data['MA200'].iloc[-1]

    if latest_close > ma50:
        score += 1
    if latest_close > ma200:
        score += 1
    if ma50 > ma200:
        score += 1

    delta = hist_data['Close'].diff(1).dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    period = 14
    gain = up.rolling(window=period).mean()
    loss = abs(down.rolling(window=period).mean())
    RS = gain / loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    latest_RSI = RSI.iloc[-1]
    if latest_RSI < 70:
        score += 1
    if latest_RSI > 50:
        score += 1

    exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
    hist_data['MACD'] = exp1 - exp2
    hist_data['Signal_Line'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
    if hist_data['MACD'].iloc[-1] > hist_data['Signal_Line'].iloc[-1]:
        score += 1

    hist_data['20_SMA'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['STDDEV'] = hist_data['Close'].rolling(window=20).std()
    hist_data['Upper_Band'] = hist_data['20_SMA'] + (hist_data['STDDEV'] * 2)
    hist_data['Lower_Band'] = hist_data['20_SMA'] - (hist_data['STDDEV'] * 2)
    if latest_close < hist_data['Upper_Band'].iloc[-1]:
        score += 1

    hist_data['OBV'] = (np.sign(hist_data['Close'].diff()) * hist_data['Volume']).fillna(0).cumsum()
    if hist_data['OBV'].iloc[-1] > hist_data['OBV'].rolling(window=20).mean().iloc[-1]:
        score += 1

    k_val, d_val, conv, base, spanA, spanB, fib_levels = add_advanced_technical_indicators(hist_data)
    if calculate_stochastic_signal(k_val, d_val):
        score += 1
    if calculate_ichimoku_signal(hist_data):
        score += 1
    if calculate_fib_signal(hist_data, fib_levels):
        score += 1
    additional_signals = add_more_technical_signals(hist_data, fib_levels)
    for signal in additional_signals:
        if signal:
            score += 1

    technical_score = (score / max_score) * 10
    return technical_score

def try_arima_models(hist_close):
    """
    Attempt multiple ARIMA models and select the best based on AIC.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter('ignore', ConvergenceWarning)

    orders = [(1,1,1),(2,1,2),(5,1,0),(3,1,1),(1,1,2),(2,2,2),(4,1,2),(3,2,1)]
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
    Fallback linear regression model for forecasting.
    """
    from sklearn.linear_model import LinearRegression

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

def price_forecast(hist_data):
    hist_close = hist_data['Close']
    hist_close.index = pd.DatetimeIndex(hist_close.index).to_period('D')

    forecasts = {}
    future_days_list = [14, 30]

    # Run ensemble modeling using advanced_modeling.py
    forecasts_linear = am.run_arima_model(hist_close)
    forecasts_xgb = am.run_xgboost_model(hist_data_enriched=hist_data_enriched)  # Pass enriched data if available
    forecasts_bilstm = am.run_bidirectional_lstm(hist_data_enriched=hist_data_enriched)  # Pass enriched data if available

    # Ensemble predictions
    forecasts_ensemble = am.ensemble_predictions([forecasts_linear, forecasts_xgb, forecasts_bilstm])

    return forecasts_ensemble

def analyze_stock(ticker):
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

    # Create advanced features (technical, fundamental, sentiment placeholders)
    hist_data_enriched = am.create_advanced_features(hist_data, ticker, financials)

    # Example: run a baseline linear model as before
    fundamental_score = fundamental_analysis(financials)
    technical_score = technical_analysis(hist_data)
    forecasts_linear = am.run_arima_model(hist_data['Close'])

    # Now run advanced models for better accuracy
    # XGBoost model for regression
    forecasts_xgb = am.run_xgboost_model(hist_data_enriched)

    # Bidirectional LSTM model for improved sequence modeling
    forecasts_bilstm = am.run_bidirectional_lstm(hist_data_enriched)

    # Ensemble predictions from ARIMA, XGBoost, and Bidirectional LSTM
    forecasts_ensemble = am.ensemble_predictions([forecasts_linear, forecasts_xgb, forecasts_bilstm])

    current_price = hist_data['Close'].iloc[-1]

    results = {
        'ticker': ticker,
        'current_price': current_price,
        'forecasts': forecasts_ensemble,
        'expected_return_percentages': {},
        'likelihood_score': 0,
        'hist_data': hist_data,
        'financials': financials
    }

    for days_ahead, forecast_price in forecasts_ensemble.items():
        percentage_increase = ((forecast_price - current_price) / current_price) * 100
        results['expected_return_percentages'][days_ahead] = percentage_increase

    overall_score = (fundamental_score * 0.6) + (technical_score * 0.4)
    results['likelihood_score'] = overall_score

    return results

def save_to_history(result):
    with open('history.log', 'a') as file:
        file.write(f"Date: {datetime.now()}\n")
        file.write(f"Ticker: {result['ticker']}\n")
        file.write(f"Current Price: ${result['current_price']:.2f}\n")
        for days_ahead, forecast_price in result['forecasts'].items():
            expected_return = result['expected_return_percentages'][days_ahead]
            file.write(f"Forecast Price ({days_ahead} days): ${forecast_price:.2f}\n")
            file.write(f"Expected Return ({days_ahead} days): {expected_return:.2f}%\n")
        file.write(f"Likelihood Score: {result['likelihood_score']:.2f}/10\n")
        file.write("-" * 50 + "\n")

def predict_probability(hist_data, ticker):
    atr_value = calculate_atr(hist_data, period=14)
    if np.isnan(atr_value) or atr_value == 0:
        return None

    current_price = hist_data['Close'].iloc[-1]
    bullish_1ATR = current_price + atr_value
    bearish_1ATR = current_price - atr_value
    bullish_3ATR = current_price + 3 * atr_value
    bearish_3ATR = current_price - 3 * atr_value
    bullish_5ATR = current_price + 5 * atr_value
    bearish_5ATR = current_price - 5 * atr_value

    df = hist_data.copy()
    df['Next_High'] = df['High'].shift(-1)
    df['Next_Low'] = df['Low'].shift(-1)
    valid_data = df.dropna(subset=['Next_High', 'Next_Low'])

    if len(valid_data) == 0:
        return None

    def prob_move(factor):
        up_move = (valid_data['Next_High'] > valid_data['Close'] + factor * atr_value).mean() * 100.0
        down_move = (valid_data['Next_Low'] < valid_data['Close'] - factor * atr_value).mean() * 100.0
        return up_move, down_move

    prob_up_1, prob_down_1 = prob_move(1)
    prob_up_3, prob_down_3 = prob_move(3)
    prob_up_5, prob_down_5 = prob_move(5)

    stock = yf.Ticker(ticker)
    stock_info = stock.info
    sector = stock_info.get('sector', None)
    sector_etf = get_sector_etf(sector)
    market_condition = get_market_condition()
    sector_condition = get_sector_condition(sector_etf)

    prob_up_1, prob_down_1 = adjust_probabilities(prob_up_1, prob_down_1, market_condition, sector_condition)
    prob_up_3, prob_down_3 = adjust_probabilities(prob_up_3, prob_down_3, market_condition, sector_condition)
    prob_up_5, prob_down_5 = adjust_probabilities(prob_up_5, prob_down_5, market_condition, sector_condition)

    # Determine likely ATR targets
    likely_1ATR = "Bullish target more likely first" if prob_up_1 > prob_down_1 else ("Bearish target more likely first" if prob_down_1 > prob_up_1 else "Both equally likely")
    likely_3ATR = "Bullish target more likely first" if prob_up_3 > prob_down_3 else ("Bearish target more likely first" if prob_down_3 > prob_up_3 else "Both equally likely")
    likely_5ATR = "Bullish target more likely first" if prob_up_5 > prob_down_5 else ("Bearish target more likely first" if prob_down_5 > prob_up_5 else "Both equally likely")

    # Overall prediction based on all ATR levels
    total_prob_up = prob_up_1 + prob_up_3 + prob_up_5
    total_prob_down = prob_down_1 + prob_down_3 + prob_down_5

    if total_prob_up > total_prob_down:
        overall_prediction = "Bullish"
    elif total_prob_down > total_prob_up:
        overall_prediction = "Bearish"
    else:
        overall_prediction = "Neutral"

    # Add to prediction history if not already present
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
        'prob_up_1': prob_up_1,
        'prob_down_1': prob_down_1,
        'bullish_1ATR': bullish_1ATR,
        'bearish_1ATR': bearish_1ATR,
        'likely_1ATR': likely_1ATR,

        'prob_up_3': prob_up_3,
        'prob_down_3': prob_down_3,
        'bullish_3ATR': bullish_3ATR,
        'bearish_3ATR': bearish_3ATR,
        'likely_3ATR': likely_3ATR,

        'prob_up_5': prob_up_5,
        'prob_down_5': prob_down_5,
        'bullish_5ATR': bullish_5ATR,
        'bearish_5ATR': bearish_5ATR,
        'likely_5ATR': likely_5ATR,

        'overall_prediction': overall_prediction
    }

def display_analysis_results(result):
    if result:
        print(f"\nForecast for {result['ticker']}:")
        print(f"Current Price: ${result['current_price']:.2f}")
        for days_ahead in sorted(result['forecasts'].keys()):
            forecast_price = result['forecasts'][days_ahead]
            expected_return = result['expected_return_percentages'][days_ahead]
            print(f"Forecast Price ({days_ahead} days): ${forecast_price:.2f}")
            print(f"Expected Return ({days_ahead} days): {expected_return:.2f}%")
        print(f"Likelihood Score: {result['likelihood_score']:.2f}/10")
        if result['likelihood_score'] >= 7:
            print("Recommendation: Consider entering a trade.")
            stop_loss = result['current_price'] * 0.95
            profit_target = result['current_price'] + (result['current_price'] - stop_loss) * 3
            print(f"Suggested Entry Price: ${result['current_price']:.2f}")
            print(f"Suggested Profit Target: ${profit_target:.2f}")
            print(f"Suggested Stop Loss: ${stop_loss:.2f}")
        else:
            print("Recommendation: The stock does not meet the criteria for entry at this time.")
    else:
        print("Analysis could not be completed.")

def prediction_history_cli():
    predictions_history = load_predictions_history()
    if not predictions_history:
        print("No Prediction History found. Query 'Predict Probability' first.")
        input("\nPress Enter to return to the main menu...")
        return

    # Define color palette for urwid
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

    # Updated columns with renamed headers and color coding
    columns = [
        ('ticker', 'Ticker', 10),
        ('queried_price', 'Queried Price', 15),
        ('bullish_1ATR', 'Bullish 1', 10),
        ('bearish_1ATR', 'Bearish 1', 10),
        ('bullish_3ATR', 'Bullish 3', 10),
        ('bearish_3ATR', 'Bearish 3', 10),
        ('bullish_5ATR', 'Bullish 5', 10),
        ('bearish_5ATR', 'Bearish 5', 10),
        ('overall_prediction', 'Overall Prediction', 20),
        ('status', 'Status', 15)
    ]

    def get_prediction_data():
        updates = []
        header_line = []
        for key, header_text, width in columns:
            header_line.append(('headers', header_text.ljust(width)))
        updates.extend(header_line)
        updates.append(('', '\n'))

        for ticker, info in predictions_history.items():
            queried_price = info['queried_price']
            b1 = info['bullish_1ATR']
            be1 = info['bearish_1ATR']
            b3 = info['bullish_3ATR']
            be3 = info['bearish_3ATR']
            b5 = info['bullish_5ATR']
            be5 = info['bearish_5ATR']
            overall_prediction = info.get('overall_prediction', 'Neutral')

            # Current price
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="1d")
                if stock_data.empty:
                    current_price = queried_price
                else:
                    current_price = stock_data['Close'].iloc[-1]
            except:
                current_price = queried_price

            # Check if 5th ATR hit
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
                'queried_price': f'${queried_price:.2f}',
                'bullish_1ATR': f'${b1:.2f}',
                'bearish_1ATR': f'${be1:.2f}',
                'bullish_3ATR': f'${b3:.2f}',
                'bearish_3ATR': f'${be3:.2f}',
                'bullish_5ATR': f'${b5:.2f}',
                'bearish_5ATR': f'${be5:.2f}',
                'overall_prediction': overall_prediction,
                'status': status
            }

            # Determine color for overall prediction
            if overall_prediction == "Bullish":
                overall_color = 'overall_bullish'
            elif overall_prediction == "Bearish":
                overall_color = 'overall_bearish'
            else:
                overall_color = 'overall_neutral'

            line = []
            for key, header_text, width in columns:
                if 'bullish' in key:
                    col_attr = 'bullish_col'
                elif 'bearish' in key:
                    col_attr = 'bearish_col'
                elif key == 'overall_prediction':
                    if overall_prediction == "Bullish":
                        col_attr = 'overall_bullish'
                    elif overall_prediction == "Bearish":
                        col_attr = 'overall_bearish'
                    else:
                        col_attr = 'overall_neutral'
                else:
                    col_attr = status_attr
                line.append((col_attr, line_data[key].ljust(width)))
            line.append(('', '\n'))
            updates.extend(line)

        return updates

    def handle_input(key):
        if key in ('R', 'r'):
            refresh()
        elif key in ('D', 'd'):
            delete_hit_tickers()
            refresh()
        elif key in ('Q', 'q'):
            raise urwid.ExitMainLoop()

    def refresh():
        body_text.set_text(get_prediction_data())

    def delete_hit_tickers():
        nonlocal predictions_history
        to_delete = []
        for ticker, info in predictions_history.items():
            b5 = info['bullish_5ATR']
            be5 = info['bearish_5ATR']
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="1d")
                if stock_data.empty:
                    current_price = info['queried_price']
                else:
                    current_price = stock_data['Close'].iloc[-1]
            except:
                current_price = info['queried_price']

            if current_price > b5 or current_price < be5:
                to_delete.append(ticker)

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
    if os.path.exists('history.log'):
        while True:
            print("\n--- View Analysis History ---")
            print("1. View History")
            print("2. Save History to a Text File")
            print("3. Clear History")
            print("4. Return to Main Menu")
            choice = input("Please select an option (1-4): ")

            if choice == '1':
                with open('history.log', 'r') as file:
                    history = file.read()
                    print(history)
            elif choice == '2':
                filename = input("Enter the filename to save history (e.g., history.txt): ")
                try:
                    with open('history.log', 'r') as src_file:
                        data = src_file.read()
                    with open(filename, 'w') as dest_file:
                        dest_file.write(data)
                    print(f"History saved to {filename}")
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
                print("Invalid option selected. Please try again.")
    else:
        print("\nNo history found.")

def main_menu():
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
            ticker = input("Enter the stock ticker symbol: ").upper()
            result = analyze_stock(ticker)
            display_analysis_results(result)
            if result:
                save_to_history(result)
        elif choice == '2':
            ticker = input("Enter the stock ticker symbol: ").upper()
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
                    print(f"Bullish Target (1 ATR): ${prob['bullish_1ATR']:.2f}")
                    print(f"Bearish Target (1 ATR): ${prob['bearish_1ATR']:.2f}")
                    print(f"Probability of hitting Bullish Target first (1 ATR): {prob['prob_up_1']:.2f}%")
                    print(f"Probability of hitting Bearish Target first (1 ATR): {prob['prob_down_1']:.2f}%")
                    print(prob['likely_1ATR'])

                    print("\n3 ATR Move:")
                    print(f"Bullish Target (3 ATR): ${prob['bullish_3ATR']:.2f}")
                    print(f"Bearish Target (3 ATR): ${prob['bearish_3ATR']:.2f}")
                    print(f"Probability of hitting Bullish Target first (3 ATR): {prob['prob_up_3']:.2f}%")
                    print(f"Probability of hitting Bearish Target first (3 ATR): {prob['prob_down_3']:.2f}%")
                    print(prob['likely_3ATR'])

                    print("\n5 ATR Move:")
                    print(f"Bullish Target (5 ATR): ${prob['bullish_5ATR']:.2f}")
                    print(f"Bearish Target (5 ATR): ${prob['bearish_5ATR']:.2f}")
                    print(f"Probability of hitting Bullish Target first (5 ATR): {prob['prob_up_5']:.2f}%")
                    print(f"Probability of hitting Bearish Target first (5 ATR): {prob['prob_down_5']:.2f}%")
                    print(prob['likely_5ATR'])

                    print(f"\nOverall Prediction: {prob['overall_prediction']}")
                else:
                    print("Not enough data to compute probability.")
            except Exception as e:
                print(f"Error predicting probability for {ticker}: {e}")
        elif choice == '4':
            # View Analysis History
            view_history()
        elif choice == '5':
            # View Prediction History
            prediction_history_cli()
        elif choice == '6':
            print("Exiting the tool. Goodbye!")
            break
        else:
            print("Invalid option selected. Please try again.")

if __name__ == "__main__":
    main_menu()
