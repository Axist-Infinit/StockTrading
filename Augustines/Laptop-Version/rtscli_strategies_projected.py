# rtscli_strategies_projected.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ANSI escape sequences for colors (duplicated or refactored here)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

# Example function to compute RSI — you can reuse the utilities from rtscli_strategies or define new ones
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_projected_swing_strategy(only_buy_signals=False, watchlist_name=None, stock_list=None):
    """
    Re-located from rtscli_strategies.py

    If stock_list is provided, we can skip watchlist loading logic (optional).
    Otherwise, you'll have to replicate the watchlist loading or pass in externally.
    """
    print(f"{Colors.CYAN}Running Projected Swing Trading Strategy...{Colors.RESET}")

    # Example logic — adapt or keep as is
    # Below is the same logic you had before, minus direct references
    # to watchlist loads if you handle them externally.
    if stock_list is None:
        print(f"{Colors.RED}No stock list provided. Please pass in a list of tickers or handle watchlist logic.{Colors.RESET}")
        return

    ema_periods = [7, 21, 50]
    for symbol in stock_list:
        try:
            df = yf.download(symbol, period="1y", progress=False)
            if df.empty:
                print(f"{Colors.RED}{symbol}: No data found.{Colors.RESET}")
                continue

            # Example calculations
            for period in ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

            df['RSI'] = compute_rsi(df['Close'], window=14)
            df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()  # Simplified for illustration

            latest = df.iloc[-1]
            current_price = latest['Close']
            atr = latest['ATR']

            potential_buy_level = latest['EMA_7']
            stop_loss = potential_buy_level - (1 * atr)
            profit_target = potential_buy_level + (2 * atr)
            expected_gain = ((profit_target - potential_buy_level) / potential_buy_level) * 100

            # Print out the results
            print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET}")
            print(f"  {Colors.YELLOW}Current Price: {Colors.RESET}${current_price:.2f}")
            print(f"  {Colors.GREEN}Potential Buy Level: {Colors.RESET}${potential_buy_level:.2f}")
            print(f"  {Colors.RED}Stop Loss: {Colors.RESET}${stop_loss:.2f}")
            print(f"  {Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f}")
            print(f"  {Colors.CYAN}Expected Gain: {Colors.RESET}{expected_gain:.2f}%")
            print(f"  {Colors.YELLOW}RSI: {Colors.RESET}{latest['RSI']:.2f}")

        except Exception as e:
            print(f"{Colors.RED}{symbol}: An error occurred - {e}{Colors.RESET}")


def run_projected_vwap_swing_trade_strategy(only_buy_signals=False, watchlist_name=None, stock_list=None):
    """
    Re-located from rtscli_strategies.py
    Similar approach as run_projected_swing_strategy
    """
    print(f"{Colors.CYAN}Running Projected VWAP Swing Trade Strategy...{Colors.RESET}")

    if stock_list is None:
        print(f"{Colors.RED}No stock list provided. Please pass in a list of tickers or handle watchlist logic.{Colors.RESET}")
        return

    for symbol in stock_list:
        try:
            df = yf.download(symbol, period="6mo", interval='1d', progress=False)
            if df.empty:
                print(f"{Colors.RED}{symbol}: No data found.{Colors.RESET}")
                continue

            # Example calculations
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()  # Simplified

            latest = df.iloc[-1]
            current_price = latest['Close']
            atr = latest['ATR']

            potential_buy_level = latest['VWAP']
            stop_loss = potential_buy_level - (1.5 * atr)
            profit_target = potential_buy_level + (2 * atr * 1.5)
            expected_gain = ((profit_target - potential_buy_level) / potential_buy_level) * 100

            # Print out the results
            print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET}")
            print(f"  {Colors.YELLOW}Current Price: {Colors.RESET}${current_price:.2f}")
            print(f"  {Colors.GREEN}Potential Buy Level: {Colors.RESET}${potential_buy_level:.2f}")
            print(f"  {Colors.RED}Stop Loss: {Colors.RESET}${stop_loss:.2f} (1.5 ATR)")
            print(f"  {Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f}")
            print(f"  {Colors.CYAN}Expected Gain: {Colors.RESET}{expected_gain:.2f}%")

        except Exception as e:
            print(f"{Colors.RED}{symbol}: An error occurred - {e}{Colors.RESET}")
