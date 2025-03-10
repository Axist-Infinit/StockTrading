# rtscli_strategies_patterns.py

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from rtscli_utils import load_holdings, save_holdings, load_watchlist, select_watchlist

# ANSI escape sequences for colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

##############################################
# RSI and Helper Functions
##############################################
def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI) for a given price series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def safe_get_price_column(df):
    """
    Try to get 'Close', 'Adj Close', or 'AdjClose' in a case-insensitive manner.
    If found, return that series. 
    Otherwise, return None.
    """
    for col in df.columns:
        if col.lower() in ['close', 'adj close', 'adjclose']:
            return df[col]
    return None

def ensure_columns(df, cols):
    """
    Check if all required columns exist in the DataFrame.
    Returns (True, []) if all columns are present, or (False, [missing_cols]) otherwise.
    """
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        return False, missing_cols
    return True, []

def check_trading_conditions(df, symbol, only_buy_signals):
    """
    Check volume, trend, and RSI conditions before pattern detection.
    Returns True if all conditions are met, False otherwise.
    """
    # Check moving averages for bullish trend
    if df['MA20'].iloc[-1] < df['MA50'].iloc[-1]:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: Moving averages do not indicate a bullish trend.{Colors.RESET}")
        return False

    # Check RSI for bullish momentum
    if df['RSI'].iloc[-1] < 50:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: RSI indicates weak momentum.{Colors.RESET}")
        return False

    # Check for high volume (1.5x average)
    if df['Volume'].iloc[-1] < df['AvgVolume'].iloc[-1] * 1.5:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: Volume not high enough to confirm the signal.{Colors.RESET}")
        return False

    return True

def check_earnings(symbol, only_buy_signals):
    """
    Check for upcoming earnings within 7 days.
    Returns True if no earnings are upcoming within the next 7 days, False otherwise.
    """
    try:
        earnings = yf.Ticker(symbol).calendar
        if not earnings.empty:
            # The first column's name is the date (by default)
            earnings_date = earnings.columns[0]
            earnings_datetime = earnings.loc['Earnings Date', earnings_date]
            if pd.notnull(earnings_datetime):
                earnings_date = pd.to_datetime(earnings_datetime)
                today = pd.Timestamp.today()
                if today <= earnings_date <= today + pd.Timedelta(days=7):
                    if not only_buy_signals:
                        print(f"{Colors.RED}{symbol}: Upcoming earnings report on {earnings_date.date()}. Skipping trade.{Colors.RESET}")
                    return False
    except Exception:
        # If earnings data not available or any other issue, proceed
        print(f"{Colors.MAGENTA}{symbol}: Could not retrieve earnings data. Proceeding with trade checks.{Colors.RESET}")
    return True

def prepare_data(df, symbol, only_buy_signals):
    """
    Prepare the DataFrame with required columns and indicators.
    Returns the prepared DataFrame or None if preparation fails.
    """
    if df.empty:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: No data found.{Colors.RESET}")
        return None

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # If user stored "AdjClose" but not "Close", rename it:
    if "Close" not in df.columns and "AdjClose" in df.columns:
        df.rename(columns={"AdjClose":"Close"}, inplace=True)

    # Try to get price column
    price = safe_get_price_column(df)
    if price is None:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: Neither 'Close' nor 'Adj Close' column found in data.{Colors.RESET}")
        return None
    else:
        # Ensure we have a final 'Close' column
        df['Close'] = price

    # Ensure required columns
    required_cols = ['High', 'Low', 'Volume', 'Close']
    cols_present, missing_cols = ensure_columns(df, required_cols)
    if not cols_present:
        if not only_buy_signals:
            print(f"{Colors.RED}{symbol}: Missing columns: {', '.join(missing_cols)}.{Colors.RESET}")
        return None

    # Compute MAs, RSI, Average Volume, ATR
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()

    # Calculate ATR
    close_shifted = df['Close'].shift()
    df['TR'] = df.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - close_shifted[row.name]) if pd.notnull(close_shifted[row.name]) else (row['High'] - row['Low']),
            abs(row['Low'] - close_shifted[row.name]) if pd.notnull(close_shifted[row.name]) else (row['High'] - row['Low'])
        ),
        axis=1
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    return df

##############################################
# Inverse Head and Shoulders Strategy
##############################################
def run_inverse_head_and_shoulders_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    """
    Run the Inverse Head and Shoulders trading strategy on the watchlist.
    """
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    for symbol in stock_list:
        try:
            # If data_cache is available and has this symbol, use it
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                # Otherwise, just download 6 months
                df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)

            df = prepare_data(df, symbol, only_buy_signals)
            if df is None:
                continue

            # Check trading conditions
            if not check_trading_conditions(df, symbol, only_buy_signals):
                continue

            # Check for upcoming earnings
            if not check_earnings(symbol, only_buy_signals):
                continue

            # Identify local minima (troughs) and maxima (peaks)
            price_values = df['Close'].values
            peaks, _ = find_peaks(price_values)
            troughs, _ = find_peaks(-price_values)

            if len(troughs) < 3:
                if not only_buy_signals:
                    print(f"{Colors.RED}{symbol}: Not enough troughs for Inverse Head and Shoulders pattern.{Colors.RESET}")
                continue

            pattern_found = False
            for i in range(1, len(troughs) - 1):
                left_shoulder = price_values[troughs[i - 1]]
                head = price_values[troughs[i]]
                right_shoulder = price_values[troughs[i + 1]]

                # Basic pattern checks
                if (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05 and
                    (left_shoulder > head * 1.1) and (right_shoulder > head * 1.1)):

                    # Get the peaks after each trough
                    peaks_after_left = peaks[peaks > troughs[i - 1]]
                    peaks_after_head = peaks[peaks > troughs[i]]
                    if len(peaks_after_left) == 0 or len(peaks_after_head) == 0:
                        continue

                    neckline_left = price_values[peaks_after_left[0]]
                    neckline_right = price_values[peaks_after_head[0]]
                    neckline = (neckline_left + neckline_right) / 2

                    # Check for breakout above the neckline
                    if price_values[-1] > neckline:
                        current_price = price_values[-1]
                        atr = df['ATR'].iloc[-1]
                        stop_loss = head
                        profit_target = current_price + (neckline - head) * 2

                        print(f"\n{Colors.GREEN}{symbol} - Inverse Head and Shoulders detected.{Colors.RESET}")
                        print(f"  {Colors.YELLOW}Current Price: {Colors.RESET}${current_price:.2f}")
                        print(f"  {Colors.RED}Stop Loss: {Colors.RESET}${stop_loss:.2f} (Below head)")
                        print(f"  {Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f}")
                        print(f"  {Colors.MAGENTA}RSI: {Colors.RESET}{df['RSI'].iloc[-1]:.2f}")
                        print(f"  {Colors.MAGENTA}Volume: {Colors.RESET}{df['Volume'].iloc[-1]} (Avg: {df['AvgVolume'].iloc[-1]:.0f})")
                        print(f"  {Colors.MAGENTA}ATR: {Colors.RESET}${atr:.2f}")

                        if not only_buy_signals:
                            add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                            if add_holding == 'yes':
                                holdings = load_holdings()
                                try:
                                    price_paid = float(input(
                                        f"Enter price paid for {symbol} (default is current price ${current_price:.2f}): "
                                    ) or current_price)
                                    number_of_shares = int(input("Enter number of shares: "))
                                    holding = {
                                        'symbol': symbol,
                                        'price_paid': price_paid,
                                        'shares': number_of_shares,
                                        'stop_loss': stop_loss,
                                        'profit_target': profit_target,
                                        'strategy': 'Inverse H+S'
                                    }
                                    holdings.append(holding)
                                    save_holdings(holdings)
                                    print(f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} and profit target at ${profit_target:.2f}.{Colors.RESET}")
                                except ValueError:
                                    print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
                        pattern_found = True
                        break  # Move to next symbol after finding the pattern

            if not pattern_found and not only_buy_signals:
                print(f"{Colors.RED}{symbol}: No Inverse Head and Shoulders pattern detected.{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}{symbol}: An error occurred - {e}{Colors.RESET}")

        if not only_buy_signals:
            input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")


##############################################
# Rectangle Pattern Strategy
##############################################
def run_rectangle_pattern_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    """
    Run the Rectangle Pattern trading strategy on the watchlist.
    """
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    for symbol in stock_list:
        try:
            # If data_cache is available, use it; otherwise download
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)

            df = prepare_data(df, symbol, only_buy_signals)
            if df is None:
                continue

            # Check trading conditions
            if not check_trading_conditions(df, symbol, only_buy_signals):
                continue

            # Check for upcoming earnings
            if not check_earnings(symbol, only_buy_signals):
                continue

            # Calculate rolling max/min to find rectangle boundaries
            rolling_max = df['High'].rolling(window=20).max()
            rolling_min = df['Low'].rolling(window=20).min()

            recent_high = rolling_max.iloc[-1]
            recent_low = rolling_min.iloc[-1]
            price_range = recent_high - recent_low
            if price_range == 0:
                if not only_buy_signals:
                    print(f"{Colors.RED}{symbol}: Price range is zero. Skipping.{Colors.RESET}")
                continue

            price = df['Close']
            within_range = price[-20:].between(recent_low * 0.99, recent_high * 1.01).all()

            # Breakout above the rectangle
            if within_range and price.iloc[-1] > recent_high:
                current_price = price.iloc[-1]
                atr = df['ATR'].iloc[-1]
                stop_loss = recent_low
                profit_target = current_price + (recent_high - recent_low) * 2

                print(f"\n{Colors.GREEN}{symbol} - Rectangle pattern breakout detected.{Colors.RESET}")
                print(f"  {Colors.YELLOW}Current Price: {Colors.RESET}${current_price:.2f}")
                print(f"  {Colors.RED}Stop Loss: {Colors.RESET}${stop_loss:.2f} (Below rectangle support)")
                print(f"  {Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f}")
                print(f"  {Colors.MAGENTA}RSI: {Colors.RESET}{df['RSI'].iloc[-1]:.2f}")
                print(f"  {Colors.MAGENTA}Volume: {Colors.RESET}{df['Volume'].iloc[-1]} (Avg: {df['AvgVolume'].iloc[-1]:.0f})")
                print(f"  {Colors.MAGENTA}ATR: {Colors.RESET}${atr:.2f}")

                if not only_buy_signals:
                    add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                    if add_holding == 'yes':
                        holdings = load_holdings()
                        try:
                            price_paid = float(input(
                                f"Enter price paid for {symbol} (default=${current_price:.2f}): "
                            ) or current_price)
                            number_of_shares = int(input("Enter number of shares: "))
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': number_of_shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'Rectangle Pattern'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} and profit target at ${profit_target:.2f}.{Colors.RESET}")
                        except ValueError:
                            print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
            else:
                if not only_buy_signals:
                    print(f"{Colors.RED}{symbol}: No Rectangle pattern breakout detected.{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}{symbol}: An error occurred - {e}{Colors.RESET}")

        if not only_buy_signals:
            input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")


##############################################
# Ascending Channel Strategy
##############################################
def run_ascending_channel_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    """
    Run the Ascending Channel trading strategy on the watchlist.
    """
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    for symbol in stock_list:
        try:
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)

            df = prepare_data(df, symbol, only_buy_signals)
            if df is None:
                continue

            # Check trading conditions
            if not check_trading_conditions(df, symbol, only_buy_signals):
                continue

            # Check for upcoming earnings
            if not check_earnings(symbol, only_buy_signals):
                continue

            # Identify peaks and troughs
            price_values = df['Close'].values
            peaks, _ = find_peaks(price_values)
            troughs, _ = find_peaks(-price_values)

            if len(peaks) < 2 or len(troughs) < 2:
                if not only_buy_signals:
                    print(f"{Colors.RED}{symbol}: Not enough peaks or troughs for Ascending Channel pattern.{Colors.RESET}")
                continue

            # Fit lines to last two peaks + last two troughs
            peak_prices = price_values[peaks[-2:]]
            trough_prices = price_values[troughs[-2:]]
            peak_times = peaks[-2:]
            trough_times = troughs[-2:]

            peak_slope = 0
            if len(peak_times) == 2 and (peak_times[1] != peak_times[0]):
                peak_slope = (peak_prices[1] - peak_prices[0]) / (peak_times[1] - peak_times[0])
            trough_slope = 0
            if len(trough_times) == 2 and (trough_times[1] != trough_times[0]):
                trough_slope = (trough_prices[1] - trough_prices[0]) / (trough_times[1] - trough_times[0])

            # Check if slopes are both positive and ~similar
            if peak_slope > 0 and trough_slope > 0 and abs(peak_slope - trough_slope) / abs(peak_slope) < 0.2:
                # Check if current price is near the lower channel line
                projected_trough = trough_prices[1] + trough_slope * (len(price_values) - trough_times[1])
                current_price = price_values[-1]

                if current_price <= projected_trough * 1.02:
                    atr = df['ATR'].iloc[-1]
                    stop_loss = current_price - (1.5 * atr)
                    profit_target = current_price + (3 * atr)

                    print(f"\n{Colors.GREEN}{symbol} - Ascending Channel pattern detected.{Colors.RESET}")
                    print(f"  {Colors.YELLOW}Current Price: {Colors.RESET}${current_price:.2f}")
                    print(f"  {Colors.RED}Stop Loss: {Colors.RESET}${stop_loss:.2f} (1.5 ATR below entry)")
                    print(f"  {Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f}")
                    print(f"  {Colors.MAGENTA}RSI: {Colors.RESET}{df['RSI'].iloc[-1]:.2f}")
                    print(f"  {Colors.MAGENTA}Volume: {Colors.RESET}{df['Volume'].iloc[-1]} (Avg: {df['AvgVolume'].iloc[-1]:.0f})")
                    print(f"  {Colors.MAGENTA}ATR: {Colors.RESET}${atr:.2f}")

                    if not only_buy_signals:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            try:
                                price_paid = float(input(
                                    f"Enter price paid for {symbol} (default=${current_price:.2f}): "
                                ) or current_price)
                                number_of_shares = int(input("Enter number of shares: "))
                                holding = {
                                    'symbol': symbol,
                                    'price_paid': price_paid,
                                    'shares': number_of_shares,
                                    'stop_loss': stop_loss,
                                    'profit_target': profit_target,
                                    'strategy': 'Ascending Channel'
                                }
                                holdings.append(holding)
                                save_holdings(holdings)
                                print(f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} and profit target at ${profit_target:.2f}.{Colors.RESET}")
                            except ValueError:
                                print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
                else:
                    if not only_buy_signals:
                        print(f"{Colors.RED}{symbol}: Price not near lower channel line; no ascending channel buy signal.{Colors.RESET}")
            else:
                if not only_buy_signals:
                    print(f"{Colors.RED}{symbol}: No Ascending Channel pattern detected (slopes mismatch).{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}{symbol}: An error occurred - {e}{Colors.RESET}")

        if not only_buy_signals:
            input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")
