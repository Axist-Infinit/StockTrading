# rtscli_backtesting.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ANSI escape sequences for colored output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

###############################################
# Helper Functions
###############################################

def compute_ema(series, span):
    """Compute Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, window=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1 + rs))
    return rsi

def compute_atr(df, window=14):
    """Compute Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr

def compute_vwap(df):
    """Compute Volume Weighted Average Price."""
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap

def print_trades(trades_df):
    """
    Prints the trades DataFrame in a formatted table.
    """
    if trades_df.empty:
        print(f"{Colors.YELLOW}No trades to display.{Colors.RESET}")
        return
    print(trades_df.to_string(index=False))

def visualize_backtest_candlestick(ticker, df, trades_df, title, show=True, save_path=None):
    """
    Visualizes a candlestick chart with trade annotations.
    Only plots 'AnchoredVWAP', 'SMA_50', 'EMA_5', 'EMA_10' if they exist in df.
    """
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    if trades_df.empty:
        print(f"{Colors.YELLOW}No trades to visualize.{Colors.RESET}")
        return

    # Prepare buy/sell signals
    buy_signals = trades_df[['EntryDate', 'EntryPrice']]
    sell_signals = trades_df[['ExitDate', 'ExitPrice']]

    addplots = []

    # AnchoredVWAP
    if 'AnchoredVWAP' in df.columns:
        addplots.append(mpf.make_addplot(df['AnchoredVWAP'], color='orange', width=1.0, ylabel='VWAP'))

    # SMA_50
    if 'SMA_50' in df.columns:
        addplots.append(mpf.make_addplot(df['SMA_50'], color='blue', width=1.0))

    # EMA_5
    if 'EMA_5' in df.columns:
        addplots.append(mpf.make_addplot(df['EMA_5'], color='purple', width=1.0))

    # EMA_10
    if 'EMA_10' in df.columns:
        addplots.append(mpf.make_addplot(df['EMA_10'], color='green', width=1.0))

    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    fig, axlist = mpf.plot(
        df,
        type='candle',
        style=s,
        title=title,
        addplot=addplots,
        returnfig=True
    )

    ax = axlist[0]

    # Plot buy signals
    for _, row in buy_signals.iterrows():
        ax.scatter(row['EntryDate'], row['EntryPrice'], marker='^', color='green', s=100, label='Buy')

    # Plot sell signals
    for _, row in sell_signals.iterrows():
        ax.scatter(row['ExitDate'], row['ExitPrice'], marker='v', color='red', s=100, label='Sell')

    # De-duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def save_backtest_result(ticker, strategy, months, total_pnl, num_trades, win_rate, avg_pnl, avg_holding_time):
    """
    Saves the backtest results to a CSV file.
    """
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{ticker}_{strategy}_backtest.csv")
    
    result = {
        "Ticker": ticker,
        "Strategy": strategy,
        "Months": months,
        "Total_PnL": total_pnl,
        "Number_of_Trades": num_trades,
        "Win_Rate": win_rate,
        "Average_PnL": avg_pnl,
        "Average_Holding_Time_Days": avg_holding_time
    }
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.append(result, ignore_index=True)
    else:
        df = pd.DataFrame([result])
    
    df.to_csv(file_path, index=False)

def load_data(ticker, start_date, end_date):
    """
    Loads historical data for the given ticker and date range.
    """
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            group_by='column'
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"{Colors.RED}Error loading data for {ticker}: {e}{Colors.RESET}")
        return pd.DataFrame()

###############################################
# Backtest Strategy Functions
###############################################

def backtest_swing_strategy(ticker, df):
    """
    Backtests the Swing Strategy on the provided DataFrame.
    """
    if df.empty or len(df) < 50:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['EMA_7'] = compute_ema(df['Close'], 7)
    df['EMA_21'] = compute_ema(df['Close'], 21)
    df['EMA_50'] = compute_ema(df['Close'], 50)
    df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    df['ATR'] = compute_atr(df, 14)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_ema(df['Close'], 12) - compute_ema(df['Close'], 26)
    df['MACD_Signal'] = compute_ema(df['MACD'], 9)
    df['MACD_Bullish'] = df['MACD'] > df['MACD_Signal']
    
    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            if close_price < 1:
                continue
            if pd.isna(row['Avg_Volume']) or row['Avg_Volume'] < 50000:
                continue

            # Swing condition: EMA alignment and price above EMAs
            uptrend = (row['EMA_7'] > row['EMA_21']) and (row['EMA_21'] > row['EMA_50'])
            price_above_ema7 = close_price > row['EMA_7']
            rsi_ok = row['RSI'] < 70
            macd_bullish = row['MACD_Bullish']
            stochastic_ok = True  # Placeholder for actual stochastic calculation

            buy_signal = uptrend and price_above_ema7 and rsi_ok and macd_bullish and stochastic_ok
            if buy_signal:
                in_position = True
                entry_price = close_price
                entry_date = date
                entry_index = i
                atr_val = row['ATR']
                stop_loss = entry_price - atr_val
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "Swing",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "Swing",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)


def backtest_pullback_strategy(ticker, df):
    """
    Backtests the Pullback Strategy on the provided DataFrame.
    """
    if df.empty or len(df) < 50:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['EMA_10'] = compute_ema(df['Close'], 10)
    df['EMA_20'] = compute_ema(df['Close'], 20)
    df['EMA_50'] = compute_ema(df['Close'], 50)
    df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
    df['ATR'] = compute_atr(df, 14)
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            if close_price < 1:
                continue
            if pd.isna(row['Volume_Avg']) or row['Volume_Avg'] < 50000:
                continue

            # Pullback condition: price <= any of [10 EMA, 20 EMA, 50 EMA]
            pullback = (close_price <= row['EMA_10']) or (close_price <= row['EMA_20']) or (close_price <= row['EMA_50'])
            # Bounce condition: close_price >= 2% above 10 or 20 EMA
            bounce = (close_price >= row['EMA_10'] * 1.02) or \
                     (close_price >= row['EMA_20'] * 1.02)
            # Volume confirmation
            volume_ok = (row['Volume'] > 0.8 * row['Volume_Avg'])

            buy_signal = pullback and bounce and volume_ok
            if buy_signal:
                in_position = True
                entry_price = close_price
                entry_date = date
                entry_index = i
                candidate1 = low_price * 0.995
                candidate2 = row['EMA_10']
                stop_loss = min(candidate1, candidate2)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "Pullback",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "Pullback",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)


def backtest_breakout_strategy(ticker, df):
    """
    Backtests the Breakout Strategy on the provided DataFrame.
    """
    if df.empty or len(df) < 40:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['EMA_10'] = compute_ema(df['Close'], 10)
    df['EMA_20'] = compute_ema(df['Close'], 20)
    df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
    df['ATR'] = compute_atr(df, 14)
    
    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            if close_price < 1:
                continue
            if pd.isna(row['Volume_Avg']) or row['Volume_Avg'] < 50000:
                continue

            if i < 20:
                continue
            last_20_high = df['High'].iloc[i-20:i].max()
            # Require 2% breakout
            if high_price > last_20_high * 1.02:
                vol_breakout = (row['Volume'] > row['Volume_Avg'])
                if vol_breakout:
                    in_position = True
                    entry_price = close_price
                    entry_date = date
                    entry_index = i
                    candidate1 = low_price * 0.995
                    candidate2 = row['EMA_10']
                    stop_loss = min(candidate1, candidate2)
                    risk_per_share = entry_price - stop_loss
                    profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "Breakout",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "Breakout",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)


def backtest_vwap_swing_trade_strategy(ticker, df):
    """
    Backtests the VWAP Swing Trade Strategy on the provided DataFrame.
    Anchored VWAP is calculated from the most significant swing low.
    """
    if df.empty or len(df) < 30:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_TP_Vol'] = (df['Typical_Price'] * df['Volume']).cumsum()
    df['Cumulative_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Vol']

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_5']  = compute_ema(df['Close'], 5)
    df['EMA_10'] = compute_ema(df['Close'], 10)
    df['RSI']    = compute_rsi(df['Close'], 14)
    df['ATR']    = compute_atr(df, 14)
    df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

    # Identify the most significant swing low (minimum low) in the last ~3 months (approx 90 days)
    recent_period = min(len(df), 90)
    swing_low_date = df['Low'].iloc[-recent_period:].idxmin()
    if pd.isna(swing_low_date):
        swing_low_date = df.index[0]  # Fallback to first date if not found

    # Calculate Anchored VWAP from swing_low_date onwards
    df_slice = df.loc[swing_low_date:].copy()
    df_slice['Typical_Price'] = (df_slice['High'] + df_slice['Low'] + df_slice['Close']) / 3
    df_slice['Cumulative_TP_Vol'] = (df_slice['Typical_Price'] * df_slice['Volume']).cumsum()
    df_slice['Cumulative_Vol'] = df_slice['Volume'].cumsum()
    df_slice['AnchoredVWAP'] = df_slice['Cumulative_TP_Vol'] / df_slice['Cumulative_Vol']

    # Merge Anchored VWAP back into the main df
    df['AnchoredVWAP'] = np.nan
    df.loc[df_slice.index, 'AnchoredVWAP'] = df_slice['AnchoredVWAP']
    df['AnchoredVWAP'].ffill(inplace=True)

    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            # VWAP Swing Trade conditions
            if pd.isna(row['AnchoredVWAP']) or pd.isna(row['SMA_50']) or pd.isna(row['EMA_5']) or pd.isna(row['EMA_10']):
                continue

            price_above_vwap   = (close_price > row['AnchoredVWAP'])
            price_above_sma50  = (close_price > row['SMA_50'])
            ema_crossover      = (row['EMA_5'] > row['EMA_10'])
            rsi_ok             = (50 <= row['RSI'] <= 85)
            volume_ok          = (row['Volume'] > row['Avg_Volume'])

            buy_signal = (price_above_vwap and price_above_sma50 and ema_crossover and rsi_ok and volume_ok)
            if buy_signal:
                in_position = True
                entry_price = close_price
                entry_date = date
                entry_index = i
                atr_val = row['ATR']
                stop_loss = entry_price - atr_val
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "VWAP",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "VWAP",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)


def backtest_anchored_vwap_strategy(ticker, df):
    """
    Backtests the Anchored VWAP Strategy on the provided DataFrame.
    Anchored VWAP is calculated from the first day of each month.
    """
    if df.empty or len(df) < 60:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['ATR'] = compute_atr(df, 14)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['EMA_5'] = compute_ema(df['Close'], 5)
    df['EMA_10'] = compute_ema(df['Close'], 10)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    
    # Anchored VWAP Calculation from the first day of each month
    df['Month'] = df.index.to_period('M')
    unique_months = df['Month'].unique()

    # Initialize Anchored VWAP column
    df['AnchoredVWAP'] = np.nan

    for month in unique_months:
        month_data = df[df['Month'] == month]
        if month_data.empty:
            continue
        anchor_date = month_data.index[0]
        # Calculate Anchored VWAP from the anchor_date onwards
        tp = (month_data['High'] + month_data['Low'] + month_data['Close']) / 3
        cum_tp_vol = (tp * month_data['Volume']).cumsum()
        cum_vol = month_data['Volume'].cumsum()
        vwap = cum_tp_vol / cum_vol

        df.loc[anchor_date:, 'AnchoredVWAP'] = vwap

    # Forward fill Anchored VWAP
    df['AnchoredVWAP'].ffill(inplace=True)

    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            # Anchored VWAP Buy Conditions
            if pd.isna(row['AnchoredVWAP']) or pd.isna(row['SMA_50']) or pd.isna(row['EMA_5']) or pd.isna(row['EMA_10']):
                continue

            price_above_vwap   = (close_price > row['AnchoredVWAP'])
            price_above_sma50  = (close_price > row['SMA_50'])
            ema_crossover      = (row['EMA_5'] > row['EMA_10'])
            rsi_ok             = (50 <= row['RSI'] <= 85)
            volume_ok          = (row['Volume'] > row['Avg_Volume'])

            buy_signal = (price_above_vwap and price_above_sma50 and ema_crossover and rsi_ok and volume_ok)
            if buy_signal:
                in_position = True
                entry_price = close_price
                entry_date = date
                entry_index = i
                atr_val = row['ATR']
                stop_loss = entry_price - (1.5 * atr_val)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "AnchoredVWAP",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "AnchoredVWAP",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)


def backtest_enhanced_vwap_strategy(ticker, df):
    """
    Backtests the Enhanced Anchored VWAP Strategy on the provided DataFrame.
    Incorporates multiple timeframes and candlestick pattern confirmations.
    """
    if df.empty or len(df) < 60:
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    # Indicators
    df['ATR'] = compute_atr(df, 14)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['EMA_5'] = compute_ema(df['Close'], 5)
    df['EMA_10'] = compute_ema(df['Close'], 10)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    
    # Anchored VWAP Calculation from the first day of each month
    df['Month'] = df.index.to_period('M')
    unique_months = df['Month'].unique()

    # Initialize Anchored VWAP column
    df['AnchoredVWAP'] = np.nan

    for month in unique_months:
        month_data = df[df['Month'] == month]
        if month_data.empty:
            continue
        anchor_date = month_data.index[0]
        # Calculate Anchored VWAP from the anchor_date onwards
        tp = (month_data['High'] + month_data['Low'] + month_data['Close']) / 3
        cum_tp_vol = (tp * month_data['Volume']).cumsum()
        cum_vol = month_data['Volume'].cumsum()
        vwap = cum_tp_vol / cum_vol

        df.loc[anchor_date:, 'AnchoredVWAP'] = vwap

    # Forward fill Anchored VWAP
    df['AnchoredVWAP'].ffill(inplace=True)

    trades = []
    in_position = False
    entry_price = stop_loss = profit_target = None
    entry_date = None
    entry_index = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        close_price = row['Close']
        high_price = row['High']
        low_price  = row['Low']

        if not in_position:
            # Enhanced Anchored VWAP Buy Conditions
            if pd.isna(row['AnchoredVWAP']) or pd.isna(row['SMA_50']) or pd.isna(row['EMA_5']) or pd.isna(row['EMA_10']):
                continue

            price_above_vwap   = (close_price > row['AnchoredVWAP'])
            price_above_sma50  = (close_price > row['SMA_50'])
            ema_crossover      = (row['EMA_5'] > row['EMA_10'])
            rsi_ok             = (50 <= row['RSI'] <= 85)
            volume_ok          = (row['Volume'] > row['Avg_Volume'])

            # Candlestick pattern confirmation (e.g., bullish engulfing)
            # For simplicity, we'll assume a bullish engulfing pattern on the current day
            if i < 1:
                candlestick_ok = False
            else:
                prev_close = df.iloc[i-1]['Close']
                prev_open = df.iloc[i-1]['Open']
                curr_close = row['Close']
                curr_open = row['Open']
                candlestick_ok = (curr_open < prev_close) and (curr_close > prev_open)

            buy_signal = (price_above_vwap and price_above_sma50 and ema_crossover and rsi_ok and volume_ok and candlestick_ok)
            if buy_signal:
                in_position = True
                entry_price = close_price
                entry_date = date
                entry_index = i
                atr_val = row['ATR']
                stop_loss = entry_price - (1.5 * atr_val)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + 2 * risk_per_share
        else:
            # In a trade => check stop or profit target
            exit_price  = None
            exit_reason = None
            holding_time = None
            if low_price <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
            elif high_price >= profit_target:
                exit_price = profit_target
                exit_reason = "TARGET"

            if exit_price is not None:
                exit_date = date
                holding_time = (exit_date - entry_date).days
                trades.append({
                    "Strategy": "EnhancedVWAP",
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitReason": exit_reason,
                    "PnL": exit_price - entry_price,
                    "Holding_Time": holding_time
                })
                in_position = False
                entry_price = stop_loss = profit_target = None
                entry_date = None
                entry_index = None

    # If still in position at the end
    if in_position and entry_price is not None:
        last_close = df.iloc[-1]['Close']
        last_date = df.iloc[-1].name
        holding_time = (last_date - entry_date).days
        trades.append({
            "Strategy": "EnhancedVWAP",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": last_date,
            "ExitPrice": last_close,
            "ExitReason": "EndOfData",
            "PnL": last_close - entry_price,
            "Holding_Time": holding_time
        })

    if not trades:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    return pd.DataFrame(trades)

###############################################
# Consolidated Backtest Function
###############################################

def backtest_all_strategies(ticker, start_date, end_date):
    """
    Runs all strategies (Swing, Pullback, Breakout, VWAP, AnchoredVWAP, EnhancedVWAP)
    over [start_date, end_date]. Concatenates results into
    one DataFrame, then prints and visualizes.
    Also saves each strategy's backtest results individually.
    """
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        print(f"{Colors.RED}No data available for {ticker} in the specified timeframe.{Colors.RESET}")
        return pd.DataFrame(columns=[
            "Strategy","EntryDate","EntryPrice","ExitDate","ExitPrice","ExitReason","PnL","Holding_Time"
        ])

    df.sort_index(inplace=True)

    # All strategies
    strategies = {
        "Swing": backtest_swing_strategy,
        "Pullback": backtest_pullback_strategy,
        "Breakout": backtest_breakout_strategy,
        "VWAP": backtest_vwap_swing_trade_strategy,
        "AnchoredVWAP": backtest_anchored_vwap_strategy,
        "EnhancedVWAP": backtest_enhanced_vwap_strategy
    }

    combined_trades = pd.DataFrame()

    for strategy_name, strategy_func in strategies.items():
        print(f"\n{Colors.MAGENTA}Running {strategy_name} Strategy...{Colors.RESET}")
        trades_df = strategy_func(ticker, df.copy())
        if not trades_df.empty:
            # Accumulate trades
            combined_trades = pd.concat([combined_trades, trades_df], ignore_index=True)

            # Compute and save metrics
            total_pnl = trades_df['PnL'].sum()
            avg_pnl = trades_df['PnL'].mean()
            num_trades = len(trades_df)
            win_trades = trades_df[trades_df['PnL'] > 0]
            win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
            avg_holding_time = trades_df['Holding_Time'].mean()

            save_backtest_result(
                ticker=ticker,
                strategy=strategy_name,
                months=(end_date - start_date).days // 30,
                total_pnl=total_pnl,
                num_trades=num_trades,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                avg_holding_time=avg_holding_time
            )
        else:
            print(f"{Colors.YELLOW}No trades executed for {strategy_name} Strategy.{Colors.RESET}")

    if combined_trades.empty:
        print(f"{Colors.RED}No trades found across all strategies for {ticker} during this period.{Colors.RESET}")
        return combined_trades

    # ---------------------------------------------------------------------
    #  Round numeric columns to 2 decimals before printing & visualizing
    # ---------------------------------------------------------------------
    combined_trades['EntryPrice'] = combined_trades['EntryPrice'].round(2)
    combined_trades['ExitPrice']  = combined_trades['ExitPrice'].round(2)
    combined_trades['PnL']        = combined_trades['PnL'].round(2)

    # Print consolidated trades
    print(f"\n{Colors.CYAN}Consolidated Trade History for {ticker}:{Colors.RESET}")
    print_trades(combined_trades)

    # Now visualize
    visualize_backtest_candlestick(
        ticker=ticker,
        df=df,
        trades_df=combined_trades,
        title=f"{ticker} Backtest Chart - All Strategies",
        show=True,
        save_path=None
    )

    return combined_trades

###############################################
# Trade History Function
###############################################

def view_trade_history(trades):
    """
    Prints a summarized overall trade history in the requested format:

    Symbol  Entry Date  Exit Date   Entry Price Exit Price  Strategy    PnL

    `trades` is expected to be a list of dicts or a DataFrame with the required columns.
    """
    if isinstance(trades, pd.DataFrame):
        trades = trades.to_dict('records')

    if not trades:
        print(f"{Colors.YELLOW}No trades to display.{Colors.RESET}")
        return

    print("\nSymbol   Entry Date   Exit Date    Entry Price  Exit Price   Strategy           PnL")
    print("-" * 80)
    for t in trades:
        symbol       = t.get('Ticker', 'N/A')  # Assuming 'Ticker' is present
        entry_date   = t.get('EntryDate', 'N/A')
        exit_date    = t.get('ExitDate', 'N/A')
        entry_price  = t.get('EntryPrice', 'N/A')
        exit_price   = t.get('ExitPrice', 'N/A')
        strategy     = t.get('Strategy', 'N/A')
        pnl          = t.get('PnL', 'N/A')

        # Format dates if they are datetime objects
        if isinstance(entry_date, pd.Timestamp):
            entry_date_str = entry_date.strftime('%Y-%m-%d')
        else:
            entry_date_str = str(entry_date)

        if isinstance(exit_date, pd.Timestamp):
            exit_date_str = exit_date.strftime('%Y-%m-%d')
        else:
            exit_date_str = str(exit_date)

        print(f"{symbol:<8} {entry_date_str:<12} {exit_date_str:<12} "
              f"${entry_price:<11.2f} ${exit_price:<10.2f} {strategy:<18} {pnl:.2f}")

###############################################
# Run Backtest Menu Function
###############################################

def run_backtest_menu(strategy, ticker, months):
    """
    Runs a backtest for the specified strategy (Swing, Pullback, Breakout,
    VWAP, AnchoredVWAP, EnhancedVWAP, or All) on the given ticker and months range.

    :param strategy: "Swing", "Pullback", "Breakout", "VWAP", "AnchoredVWAP", "EnhancedVWAP", or "All"
    :param ticker: e.g. "AAPL"
    :param months: e.g. 6
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)

    if strategy == "All":
        backtest_all_strategies(ticker, start_date, end_date)
        return

    df = load_data(ticker, start_date, end_date)
    if df.empty:
        print(f"{Colors.RED}No data available for {ticker} in the specified timeframe.{Colors.RESET}")
        return

    df.sort_index(inplace=True)

    # Mapping strategy names to functions
    strategy_functions = {
        "Swing": backtest_swing_strategy,
        "Pullback": backtest_pullback_strategy,
        "Breakout": backtest_breakout_strategy,
        "VWAP": backtest_vwap_swing_trade_strategy,
        "AnchoredVWAP": backtest_anchored_vwap_strategy,
        "EnhancedVWAP": backtest_enhanced_vwap_strategy
    }

    if strategy not in strategy_functions:
        print(f"{Colors.RED}Unknown strategy '{strategy}'. Please specify a valid strategy.{Colors.RESET}")
        return

    print(f"\n{Colors.MAGENTA}Running {strategy} Strategy...{Colors.RESET}")
    trades_df = strategy_functions[strategy](ticker, df.copy())

    if not trades_df.empty:
        print_trades(trades_df)
        # Calculate metrics
        total_pnl = trades_df['PnL'].sum()
        avg_pnl = trades_df['PnL'].mean()
        num_trades = len(trades_df)
        win_trades = trades_df[trades_df['PnL'] > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        avg_holding_time = trades_df['Holding_Time'].mean()

        # Save backtest result
        save_backtest_result(
            ticker=ticker,
            strategy=strategy,
            months=months,
            total_pnl=total_pnl,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_holding_time=avg_holding_time
        )

        # Visualize
        visualize_backtest_candlestick(
            ticker=ticker,
            df=df,
            trades_df=trades_df,
            title=f"{ticker} Backtest Chart - {strategy} Strategy",
            show=True,
            save_path=None
        )
    else:
        print(f"{Colors.YELLOW}No trades executed for {strategy} Strategy.{Colors.RESET}")

###############################################
# Run All Strategies Function for "All" Option
###############################################

def run_all_strategies_option(ticker, months):
    """
    Runs all strategies for a given ticker and timeframe.
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    backtest_all_strategies(ticker, start_date, end_date)

###############################################
# Backtest Menu Interface
###############################################

def run_backtest_menu_interface():
    """
    Interface function to handle user inputs for backtesting.
    """
    while True:
        print("\nStrategy Backtester:")
        print("1. Swing Strategy")
        print("2. Pullback Strategy")
        print("3. Breakout Strategy")
        print("4. Simple VWAP Swing Trade Strategy")
        print("5. Enhanced VWAP Strategy")
        print("6. Anchored VWAP Strategy")
        print("7. Backtest All Strategies")
        print("0. Return to Main Menu")

        choice = input("Select a strategy to backtest: ").strip()
        if choice == '0':
            return  # Return to main menu

        ticker = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
        months_str = input("Enter the number of months to backtest (e.g., 3, 6, 12): ").strip()

        # Convert months_str -> months
        try:
            months = int(months_str)
            if months <= 0:
                raise ValueError
        except ValueError:
            print(f"{Colors.RED}Invalid number of months. Please enter a positive integer.{Colors.RESET}")
            continue

        if choice == '1':
            run_backtest_menu(strategy="Swing", ticker=ticker, months=months)
        elif choice == '2':
            run_backtest_menu(strategy="Pullback", ticker=ticker, months=months)
        elif choice == '3':
            run_backtest_menu(strategy="Breakout", ticker=ticker, months=months)
        elif choice == '4':
            run_backtest_menu(strategy="VWAP", ticker=ticker, months=months)
        elif choice == '5':
            run_backtest_menu(strategy="EnhancedVWAP", ticker=ticker, months=months)
        elif choice == '6':
            run_backtest_menu(strategy="AnchoredVWAP", ticker=ticker, months=months)
        elif choice == '7':
            run_backtest_menu(strategy="All", ticker=ticker, months=months)
        else:
            print(f"{Colors.RED}Invalid option. Please select a valid strategy number.{Colors.RESET}")

###############################################
# Trade History Function
###############################################

def view_trade_history(trades):
    """
    Prints a summarized overall trade history in the requested format:

    Symbol  Entry Date  Exit Date   Entry Price Exit Price  Strategy    PnL

    `trades` is expected to be a list of dicts or a DataFrame with the required columns.
    """
    if isinstance(trades, pd.DataFrame):
        trades = trades.to_dict('records')

    if not trades:
        print(f"{Colors.YELLOW}No trades to display.{Colors.RESET}")
        return

    print("\nSymbol   Entry Date   Exit Date    Entry Price  Exit Price   Strategy           PnL")
    print("-" * 80)
    for t in trades:
        symbol       = t.get('Ticker', 'N/A')  # Assuming 'Ticker' is present
        entry_date   = t.get('EntryDate', 'N/A')
        exit_date    = t.get('ExitDate', 'N/A')
        entry_price  = t.get('EntryPrice', 'N/A')
        exit_price   = t.get('ExitPrice', 'N/A')
        strategy     = t.get('Strategy', 'N/A')
        pnl          = t.get('PnL', 'N/A')

        # Format dates if they are datetime objects
        if isinstance(entry_date, pd.Timestamp):
            entry_date_str = entry_date.strftime('%Y-%m-%d')
        else:
            entry_date_str = str(entry_date)

        if isinstance(exit_date, pd.Timestamp):
            exit_date_str = exit_date.strftime('%Y-%m-%d')
        else:
            exit_date_str = str(exit_date)

        # Ensure entry_price and exit_price are floats for formatting
        try:
            entry_price_float = float(entry_price)
        except:
            entry_price_float = 0.0

        try:
            exit_price_float = float(exit_price)
        except:
            exit_price_float = 0.0

        try:
            pnl_float = float(pnl)
        except:
            pnl_float = 0.0

        print(f"{symbol:<8} {entry_date_str:<12} {exit_date_str:<12} "
              f"${entry_price_float:<11.2f} ${exit_price_float:<10.2f} {strategy:<18} {pnl_float:.2f}")

###############################################
# Main Execution and Additional Functions
###############################################

def analyze_all_backtests():
    """
    Analyzes all backtest results stored in the backtest_results directory.
    """
    results_dir = "backtest_results"
    if not os.path.exists(results_dir):
        print(f"{Colors.RED}No backtest results found.{Colors.RESET}")
        return

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"{Colors.RED}No backtest result CSV files found in {results_dir}.{Colors.RESET}")
        return

    all_results = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join(results_dir, file)
        df = pd.read_csv(file_path)
        all_results = pd.concat([all_results, df], ignore_index=True)

    if all_results.empty:
        print(f"{Colors.RED}No data to analyze in backtest results.{Colors.RESET}")
        return

    # Display summary statistics
    print(f"\n{Colors.CYAN}Backtest Results Summary:{Colors.RESET}")
    summary = all_results.groupby(['Strategy']).agg({
        'Total_PnL': 'sum',
        'Number_of_Trades': 'sum',
        'Win_Rate': 'mean',
        'Average_PnL': 'mean',
        'Average_Holding_Time_Days': 'mean'
    }).reset_index()

    print(summary.to_string(index=False))

    # Optionally, visualize the summary
    try:
        import matplotlib.pyplot as plt

        strategies = summary['Strategy']
        total_pnls = summary['Total_PnL']
        win_rates = summary['Win_Rate']

        fig, ax1 = plt.subplots(figsize=(10,6))

        color = 'tab:blue'
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Total PnL', color=color)
        ax1.bar(strategies, total_pnls, color=color, alpha=0.6, label='Total PnL')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(strategies, rotation=45, ha='right')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Win Rate', color=color)  # we already handled the x-label with ax1
        ax2.plot(strategies, win_rates, color=color, marker='o', label='Win Rate')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Backtest Strategies Summary")
        plt.show()
    except ImportError:
        print(f"{Colors.YELLOW}Matplotlib not installed. Skipping visualization.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error during visualization: {e}{Colors.RESET}")

def run_all_strategies_option(ticker, months):
    """
    Runs all strategies for a given ticker and timeframe.
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    backtest_all_strategies(ticker, start_date, end_date)

###############################################
# Main Function
###############################################

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RTS CLI Backtesting Tool")
    parser.add_argument('--strategy', type=str, help="Strategy to backtest: Swing, Pullback, Breakout, VWAP, AnchoredVWAP, EnhancedVWAP, All")
    parser.add_argument('--ticker', type=str, help="Stock ticker symbol, e.g., AAPL")
    parser.add_argument('--months', type=int, default=6, help="Number of months of historical data to use")
    parser.add_argument('--view-history', action='store_true', help="View summarized trade history")
    args = parser.parse_args()

    if args.view_history:
        # Load all holdings and display trade history
        # Assuming holdings are stored in backtest_results directory
        results_dir = "backtest_results"
        if not os.path.exists(results_dir):
            print(f"{Colors.RED}No backtest results found.{Colors.RESET}")
        else:
            all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
            if not all_files:
                print(f"{Colors.RED}No backtest result CSV files found in {results_dir}.{Colors.RESET}")
            else:
                all_trades = pd.DataFrame()
                for file in all_files:
                    file_path = os.path.join(results_dir, file)
                    df = pd.read_csv(file_path)
                    all_trades = pd.concat([all_trades, df], ignore_index=True)
                view_trade_history(all_trades)
    else:
        # Interactive Backtest Menu
        run_backtest_menu_interface()

if __name__ == "__main__":
    main()
