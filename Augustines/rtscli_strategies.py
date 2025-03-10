# rtscli_strategies.py

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import stock_forecast
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import from your local modules
from rtscli_utils import load_holdings, save_holdings, load_watchlist, select_watchlist
from rtscli_strategies_patterns import (
    run_inverse_head_and_shoulders_strategy,
    run_rectangle_pattern_strategy,
    run_ascending_channel_strategy
)

###################################################
# GLOBAL CACHE & HELPER FUNCTION
###################################################

# Dictionary that holds in-memory data for each ticker
SHARED_DATA = {}

def get_or_update_stock_data(symbol, start_date, end_date):
    import os
    import pandas as pd
    import yfinance as yf
    from datetime import datetime

    folder_path = "stock-data"
    os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(folder_path, f"{symbol}.csv")

    # Convert any datetime objects to strings
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    def safe_yf_download(sym, start, end):
        """Download from yfinance with a try/except to avoid clutter."""
        try:
            return yf.download(sym, start=start, end=end, progress=False)
        except Exception as e:
            print(f"[WARN] Skipping {sym} due to yfinance error: {e}")
            return pd.DataFrame()

    # 1) If CSV doesn't exist, download fresh
    if not os.path.isfile(file_path):
        df = safe_yf_download(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame()
        try:
            df.to_csv(file_path)
        except PermissionError as pe:
            print(f"[ERROR] Permission denied while creating {file_path} for {symbol}: {pe}")
            return pd.DataFrame()
        return df

    # 2) Read existing CSV
    try:
        existing_df = pd.read_csv(
            file_path,
            skiprows=3,
            names=["Date", "AdjClose", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )
    except Exception as e:
        print(f"[WARN] Could not parse CSV for {symbol}: {e}")
        try:
            existing_df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        except PermissionError as pe:
            print(f"[ERROR] Permission denied while reading {file_path} for {symbol}: {pe}")
            return pd.DataFrame()

    if isinstance(existing_df.columns, pd.MultiIndex):
        existing_df.columns = existing_df.columns.get_level_values(-1)

    # Remove duplicates
    existing_df = existing_df[~existing_df.index.duplicated(keep='first')].sort_index()

    # 3) Possibly download missing older/newer data
    desired_start_ts = pd.to_datetime(start_date)
    desired_end_ts = pd.to_datetime(end_date)
    existing_start = existing_df.index.min()
    existing_end = existing_df.index.max()
    downloaded_something = False

    # Older data
    if (existing_start is not None) and (desired_start_ts < existing_start):
        fetch_end = existing_start - pd.Timedelta(days=1)
        older_data = safe_yf_download(symbol, desired_start_ts, fetch_end)
        if not older_data.empty:
            older_data = older_data[~older_data.index.duplicated(keep='first')]
            if older_data.index.tz is not None:
                older_data.index = older_data.index.tz_convert(None)
            existing_df = pd.concat([older_data, existing_df], axis=0).drop_duplicates()
            downloaded_something = True

    # Newer data
    if (existing_end is not None) and (desired_end_ts > existing_end):
        fetch_start = existing_end + pd.Timedelta(days=1)
        newer_data = safe_yf_download(symbol, fetch_start, desired_end_ts)
        if not newer_data.empty:
            newer_data = newer_data[~newer_data.index.duplicated(keep='first')]
            if newer_data.index.tz is not None:
                newer_data.index = newer_data.index.tz_convert(None)
            existing_df = pd.concat([existing_df, newer_data], axis=0).drop_duplicates()
            downloaded_something = True

    if downloaded_something:
        existing_df.sort_index(inplace=True)
        try:
            existing_df.to_csv(file_path)
        except PermissionError as pe:
            print(f"[ERROR] Permission denied while updating {file_path} for {symbol}: {pe}")

    final_slice = existing_df.loc[desired_start_ts:desired_end_ts]
    return final_slice

###################################################
# ANSI escape sequences for colors
###################################################
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

###################################################
# SHARED FUNCTIONS
###################################################
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1 + rs))
    return rsi

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr

def compute_vwap(df):
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap

def view_trade_history(trades):
    if isinstance(trades, pd.DataFrame):
        trades = trades.to_dict('records')

    print("\nSymbol   Entry Date   Exit Date    Entry Price  Exit Price   Shares   Strategy           Result")
    print("-" * 90)
    for t in trades:
        symbol = t.get('symbol', 'N/A')
        entry_date = t.get('EntryDate', 'N/A')
        exit_date = t.get('ExitDate', 'N/A')
        entry_price = t.get('EntryPrice', 'N/A')
        exit_price = t.get('ExitPrice', 'N/A')
        shares = t.get('Shares', 'N/A')
        strategy = t.get('Strategy', 'Manual')
        result = t.get('PnL', 'N/A')

        if isinstance(entry_date, pd.Timestamp):
            entry_date_str = entry_date.strftime('%Y-%m-%d')
        else:
            entry_date_str = str(entry_date)

        if isinstance(exit_date, pd.Timestamp):
            exit_date_str = exit_date.strftime('%Y-%m-%d')
        else:
            exit_date_str = str(exit_date)

        print(f"{symbol:<8} {entry_date_str:<12} {exit_date_str:<12} "
              f"${entry_price:<11} ${exit_price:<10} {shares:<7} {strategy:<18} {result}")


###################################################
# Strategy Functions
###################################################

def run_swing_strategy_console(only_buy_signals=False, watchlist_name=None, data_cache=None):
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    ema_periods = [7, 21, 50]

    for symbol in stock_list:
        reasons = []
        try:
            # Use the data cache if provided, otherwise fetch fresh
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(years=2)
                df = get_or_update_stock_data(symbol, start_date, end_date)

            # Flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 252:
                reasons.append("Not enough data to perform calculations.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Calculate indicators
            df['52_week_high'] = df['High'].rolling(window=252, min_periods=1).max()
            for period in ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            window_length = 14
            roll_up = up.rolling(window=window_length).mean()
            roll_down = down.rolling(window=window_length).mean()
            rs = roll_up / roll_down
            df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['L14'] = df['Low'].rolling(window=14).min()
            df['H14'] = df['High'].rolling(window=14).max()
            denominator = df['H14'] - df['L14']
            denominator = denominator.replace(0, 1e-9)
            df['%K'] = 100 * ((df['Close'] - df['L14']) / denominator)
            df['%D'] = df['%K'].rolling(window=3).mean()
            df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

            latest = df.iloc[-1]
            current_price = latest['Close']
            high_52_week = latest['52_week_high']
            pct_below_52wk_high = ((high_52_week - current_price) / high_52_week) * 100

            # Threshold for below 52-week high
            threshold_pct = 10.0
            price_below_52wk_high = pct_below_52wk_high >= threshold_pct
            uptrend = latest['EMA_7'] > latest['EMA_21'] > latest['EMA_50']
            price_above_ema7 = current_price > latest['EMA_7']
            rsi_ok = latest['RSI'] < 70
            volume_ok = latest['Volume'] > latest['Avg_Volume']
            macd_bullish = latest['MACD'] > latest['MACD_Signal']
            stochastic_ok = latest['%K'] < 80

            # Earnings check
            stock = yf.Ticker(symbol)
            earnings_date = None
            try:
                cal = stock.calendar.transpose()
                if not cal.empty and 'Earnings Date' in cal.columns:
                    earnings_date = cal['Earnings Date'].values[0]
                    if pd.isnull(earnings_date):
                        earnings_date = None
            except Exception:
                pass
            earnings_ok = True
            if earnings_date:
                days_until_earnings = (earnings_date - datetime.now()).days
                earnings_ok = days_until_earnings > 14
                if not earnings_ok:
                    reasons.append("Earnings within 14 days.")

            # Higher timeframe trend
            df_weekly = df.resample('W').last()
            df_weekly['EMA_7'] = df_weekly['Close'].ewm(span=7, adjust=False).mean()
            df_weekly['EMA_21'] = df_weekly['Close'].ewm(span=21, adjust=False).mean()
            higher_timeframe_trend = df_weekly['EMA_7'].iloc[-1] > df_weekly['EMA_21'].iloc[-1]

            buy_signal = (
                uptrend and price_above_ema7 and rsi_ok and volume_ok and
                macd_bullish and stochastic_ok and earnings_ok and
                higher_timeframe_trend and price_below_52wk_high
            )

            if buy_signal:
                print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - Current Price: ${current_price:.2f}")
                atr = latest['ATR']
                stop_loss = current_price - atr
                profit_target = current_price + (2 * atr)
                expected_gain = ((profit_target - current_price) / current_price) * 100
                print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
                print(f"  Buy at: ${current_price:.2f}")
                print(f"  Stop Loss: ${stop_loss:.2f} (1 ATR below entry)")
                print(f"  Profit Target: ${profit_target:.2f} (2:1 Reward-to-Risk)")
                print(f"  Expected Gain: {expected_gain:.2f}%")
                print(f"  Current Price is {pct_below_52wk_high:.2f}% below the 52-week high of ${high_52_week:.2f}")

                if not only_buy_signals:
                    try:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            price_paid_input = input(
                                f"Enter price paid for {symbol} (default is current price ${current_price:.2f}): "
                            )
                            price_paid = float(price_paid_input) if price_paid_input else current_price
                            number_of_shares = int(input("Enter number of shares: "))
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': number_of_shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'Swing'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(
                                f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} "
                                f"and profit target at ${profit_target:.2f}.{Colors.RESET}"
                            )
                    except ValueError:
                        print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
            else:
                if not uptrend:
                    reasons.append("EMAs not in uptrend alignment.")
                if not price_above_ema7:
                    reasons.append("Price not above EMA 7.")
                if not rsi_ok:
                    reasons.append("RSI is 70 or higher.")
                if not volume_ok:
                    reasons.append("Volume below average.")
                if not macd_bullish:
                    reasons.append("MACD not bullish.")
                if not stochastic_ok:
                    reasons.append("Stochastic %K is 80 or higher.")
                if not higher_timeframe_trend:
                    reasons.append("Higher timeframe trend not bullish.")
                if not price_below_52wk_high:
                    reasons.append(
                        f"Price is too close to 52-week high ({pct_below_52wk_high:.2f}% below)."
                    )
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")

        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            print(f"{Colors.RED}An error occurred while processing {Colors.MAGENTA}{symbol}{Colors.RED}: {', '.join(reasons)}{Colors.RESET}")


def run_pullback_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    if not only_buy_signals:
        print(f"{Colors.CYAN}Checking market conditions...{Colors.RESET}")

    indices = ['SPY', 'QQQ', 'IWM', 'MDY']
    market_uptrend_count = 0
    for index in indices:
        try:
            # Minimal data fetch for the index
            if data_cache and index in data_cache:
                index_data = data_cache[index].copy()
            else:
                index_data = yf.download(index, period="2y", progress=False, group_by='column')

            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)
            if index_data.empty:
                print(f"{Colors.RED}Could not retrieve data for {index}.{Colors.RESET}")
                continue

            index_data['50_MA'] = index_data['Close'].rolling(window=50).mean()
            latest_close = index_data['Close'].iloc[-1]
            ma_50 = index_data['50_MA'].iloc[-1]

            if pd.notna(latest_close) and pd.notna(ma_50):
                if latest_close > ma_50:
                    market_uptrend_count += 1
        except Exception as e:
            print(f"{Colors.RED}Error processing {index}: {e}{Colors.RESET}")
            continue

    # Check market trend
    if market_uptrend_count >= 2:
        if not only_buy_signals:
            print(f"{Colors.GREEN}Market is in a confirmed uptrend.{Colors.RESET}")
    else:
        if not only_buy_signals:
            print(f"{Colors.RED}Market conditions are not suitable for the Pullback Strategy.{Colors.RESET}")
        return

    for symbol in stock_list:
        reasons = []
        try:
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(months=6)
                df = get_or_update_stock_data(symbol, start_date, end_date)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 126:
                reasons.append("Not enough data to perform calculations.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            buy_signal = True
            df['20_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['10_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['50_EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
            current_price = df['Close'].iloc[-1]
            if current_price < 1:
                reasons.append("Price below $1.")
                buy_signal = False

            avg_volume = df['Volume_Avg'].iloc[-1]
            if avg_volume < 50000:
                reasons.append("Average volume below 50k.")
                buy_signal = False

            df['1M_Change'] = df['Close'].pct_change(periods=20) * 100
            df['3M_Change'] = df['Close'].pct_change(periods=60) * 100
            df['6M_Change'] = df['Close'].pct_change(periods=126) * 100
            one_month_change = df['1M_Change'].iloc[-1]
            three_month_change = df['3M_Change'].iloc[-1]
            six_month_change = df['6M_Change'].iloc[-1]
            momentum = False
            if (one_month_change >= 10) or (three_month_change >= 20) or (six_month_change >= 30):
                momentum = True
            if not momentum:
                reasons.append("Does not meet momentum criteria.")
                buy_signal = False

            pullback = False
            if (current_price <= df['10_EMA'].iloc[-1]) or \
               (current_price <= df['20_EMA'].iloc[-1]) or \
               (current_price <= df['50_EMA'].iloc[-1]):
                pullback = True
            if not pullback:
                reasons.append("Not in a pullback phase.")
                buy_signal = False

            bounce = False
            ema_list = ['10_EMA', '20_EMA']
            for ema in ema_list:
                ema_value = df[ema].iloc[-1]
                if current_price >= ema_value * 1.02:
                    bounce = True
                    break
            if not bounce:
                reasons.append("Has not bounced 2% from EMA.")
                buy_signal = False

            volume_ok = df['Volume'].iloc[-1] >= 0.8 * df['Volume'].rolling(window=20).mean().iloc[-1]
            if not volume_ok:
                reasons.append("Volume confirmation not met.")
                buy_signal = False

            # Candlestick patterns
            def is_bullish_engulfing(df):
                if len(df) < 2:
                    return False
                prev_open = df['Open'].iloc[-2]
                prev_close = df['Close'].iloc[-2]
                curr_open = df['Open'].iloc[-1]
                curr_close = df['Close'].iloc[-1]
                return (curr_open < prev_close) and (curr_close > prev_open)

            def is_hammer(df):
                if len(df) < 1:
                    return False
                curr_open = df['Open'].iloc[-1]
                curr_close = df['Close'].iloc[-1]
                curr_low = df['Low'].iloc[-1]
                curr_high = df['High'].iloc[-1]
                body = abs(curr_close - curr_open)
                lower_wick = min(curr_open, curr_close) - curr_low
                upper_wick = curr_high - max(curr_open, curr_close)
                return (lower_wick > 2 * body) and (upper_wick < body)

            def is_piercing_line(df):
                if len(df) < 2:
                    return False
                prev_close = df['Close'].iloc[-2]
                prev_open = df['Open'].iloc[-2]
                curr_open = df['Open'].iloc[-1]
                curr_close = df['Close'].iloc[-1]
                mid_point = (prev_open + prev_close) / 2
                return (prev_close < prev_open) and (curr_open < prev_close) and (curr_close > mid_point)

            def is_morning_star(df):
                if len(df) < 3:
                    return False
                first = df.iloc[-3]
                second = df.iloc[-2]
                third = df.iloc[-1]
                return (first['Close'] < first['Open']) and (second['Close'] < second['Open']) and \
                       (third['Close'] > third['Open']) and (third['Close'] > (first['Close'] + first['Open']) / 2)

            def is_bullish_harami(df):
                if len(df) < 2:
                    return False
                prev_open = df['Open'].iloc[-2]
                prev_close = df['Close'].iloc[-2]
                curr_open = df['Open'].iloc[-1]
                curr_close = df['Close'].iloc[-1]
                return (prev_close < prev_open) and (curr_close > curr_open) and \
                       (curr_open > prev_close) and (curr_close < prev_open)

            def is_inverted_hammer(df):
                if len(df) < 1:
                    return False
                curr_open = df['Open'].iloc[-1]
                curr_close = df['Close'].iloc[-1]
                curr_low = df['Low'].iloc[-1]
                curr_high = df['High'].iloc[-1]
                body = abs(curr_close - curr_open)
                upper_wick = curr_high - max(curr_open, curr_close)
                lower_wick = min(curr_open, curr_close) - curr_low
                return (upper_wick > 2 * body) and (lower_wick < body)

            candlestick_signal = (
                is_bullish_engulfing(df) or
                is_hammer(df) or
                is_piercing_line(df) or
                is_morning_star(df) or
                is_bullish_harami(df) or
                is_inverted_hammer(df)
            )

            if not candlestick_signal:
                reasons.append("No bullish candlestick pattern.")
                buy_signal = False

            if buy_signal:
                print(f"\n{Colors.GREEN}{symbol} meets all the criteria for the Pullback Strategy.{Colors.RESET}")
                entry_price = current_price
                entry_day_low = df['Low'].iloc[-1]
                ema_10 = df['10_EMA'].iloc[-1]
                stop_loss_candidate1 = entry_day_low * 0.995
                stop_loss_candidate2 = ema_10
                stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + (risk_per_share * 2)

                if not only_buy_signals:
                    try:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            price_paid_input = input(
                                f"Enter price paid for {symbol} (default is current price ${entry_price:.2f}): "
                            )
                            price_paid = float(price_paid_input) if price_paid_input else entry_price
                            number_of_shares = int(input("Enter number of shares: "))
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': number_of_shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'Pullback'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(
                                f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} "
                                f"and profit target at ${profit_target:.2f}.{Colors.RESET}"
                            )
                    except ValueError:
                        print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
            else:
                reason_str = "; ".join(reasons) if reasons else "Does not meet criteria."
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
        except Exception as e:
            reasons.append(f"Error processing data: {e}")
            print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
            continue


def run_breakout_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    if not only_buy_signals:
        print(f"{Colors.CYAN}Checking market conditions...{Colors.RESET}")

    indices = ['SPY', 'QQQ', 'IWM', 'MDY']
    market_uptrend_count = 0
    for index in indices:
        try:
            if data_cache and index in data_cache:
                index_data = data_cache[index].copy()
            else:
                index_data = yf.download(index, period="2y", progress=False, group_by='column')

            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)
            if index_data.empty:
                print(f"{Colors.RED}Could not retrieve data for {index}.{Colors.RESET}")
                continue

            index_data['50_MA'] = index_data['Close'].rolling(window=50).mean()
            latest_close = index_data['Close'].iloc[-1]
            ma_50 = index_data['50_MA'].iloc[-1]
            if pd.notna(latest_close) and pd.notna(ma_50):
                if latest_close > ma_50:
                    market_uptrend_count += 1
        except Exception as e:
            print(f"{Colors.RED}Error processing {index}: {e}{Colors.RESET}")
            continue

    if market_uptrend_count >= 2:
        if not only_buy_signals:
            print(f"{Colors.GREEN}Market is in a confirmed uptrend.{Colors.RESET}")
    else:
        if not only_buy_signals:
            print(f"{Colors.RED}Market conditions are not suitable for the Breakout Strategy.{Colors.RESET}")
        return

    for symbol in stock_list:
        reasons = []
        try:
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(months=6)
                df = get_or_update_stock_data(symbol, start_date, end_date)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 60:
                reasons.append("Not enough data to perform calculations.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            buy_signal = True
            df['20_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['10_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
            current_price = df['Close'].iloc[-1]
            if current_price < 1:
                reasons.append("Price below $1.")
                buy_signal = False

            avg_volume = df['Volume_Avg'].iloc[-1]
            if avg_volume < 50000:
                reasons.append("Average volume below 50k.")
                buy_signal = False

            if len(df) < 22:
                reasons.append("Not enough data to check volatility contraction.")
                buy_signal = False
            else:
                volatility = df['Close'].rolling(window=10).std()
                recent_volatility = volatility.iloc[-1]
                previous_volatility = volatility.iloc[-11]
                volatility_contraction = recent_volatility < previous_volatility

                volume_rolling = df['Volume'].rolling(window=10).mean()
                recent_volume = volume_rolling.iloc[-1]
                previous_volume = volume_rolling.iloc[-11]
                volume_contraction = recent_volume < previous_volume

                if not (volatility_contraction or volume_contraction):
                    reasons.append("No volatility or volume contraction.")
                    buy_signal = False

            if not ((df['Close'].iloc[-3:] > df['10_EMA'].iloc[-3:]).all() and
                    (df['Close'].iloc[-3:] > df['20_EMA'].iloc[-3:]).all()):
                reasons.append("Not in tight range above EMAs.")
                buy_signal = False

            if len(df) < 21:
                reasons.append("Not enough data to check breakout criteria.")
                buy_signal = False
            else:
                df['20D_High'] = df['High'].rolling(window=20).max()
                previous_20d_high = df['20D_High'].iloc[-2]
                new_20d_high = df['20D_High'].iloc[-1]
                breakout = False
                if new_20d_high > previous_20d_high:
                    breakout_percentage = ((new_20d_high - previous_20d_high) / previous_20d_high) * 100
                    if breakout_percentage >= 2:
                        breakout = True

                if not breakout:
                    reasons.append("No 2% breakout to new 20-day highs.")
                    buy_signal = False

                breakout_volume = df['Volume'].iloc[-1] > df['Volume'].rolling(window=20).mean().iloc[-1]
                if not breakout_volume:
                    reasons.append("Breakout volume confirmation not met.")
                    buy_signal = False

            if buy_signal:
                print(f"\n{Colors.GREEN}{symbol} meets all the criteria for the Breakout Strategy.{Colors.RESET}")
                entry_price = current_price
                entry_day_low = df['Low'].iloc[-1]
                ema_10 = df['10_EMA'].iloc[-1]
                stop_loss_candidate1 = entry_day_low * 0.995
                stop_loss_candidate2 = ema_10
                stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + (risk_per_share * 2)

                print(f"{Colors.CYAN}Entry Price: {Colors.RESET}${entry_price:.2f}")
                print(f"{Colors.CYAN}Stop Loss: {Colors.RESET}${stop_loss:.2f} (Lower of 0.5% below entry day's low or 10 EMA)")
                print(f"{Colors.CYAN}Profit Target: {Colors.RESET}${profit_target:.2f} (2:1 Reward-to-Risk)")

                if not only_buy_signals:
                    add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                    if add_holding == 'yes':
                        holdings = load_holdings()
                        try:
                            price_paid = float(input(
                                f"Enter price paid for {symbol} (default is current price ${entry_price:.2f}): "
                            ) or entry_price)
                            number_of_shares = int(input("Enter number of shares: "))
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': number_of_shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'Breakout'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(
                                f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} "
                                f"and profit target at ${profit_target:.2f}.{Colors.RESET}"
                            )
                        except ValueError:
                            print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
            else:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
            continue


def run_vwap_swing_trade_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    """
    VWAP Swing Trade Strategy with fixes for indexer compatibility.
    """
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    import yfinance as yf
    from datetime import datetime

    for symbol in stock_list:
        reasons = []
        try:
            # Use cached data if available, else fetch fresh
            if data_cache and symbol in data_cache:
                df = data_cache[symbol].copy()
            else:
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(months=6)
                df = get_or_update_stock_data(symbol, start_date, end_date)

            # Flatten multi-index columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Check if DataFrame has sufficient data
            if df.empty or len(df) < 30:
                reasons.append("Not enough data to perform calculations.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Verify required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                # Fallback: re-download fresh data if required columns are missing
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(months=6)
                temp_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if temp_df.empty:
                    reasons.append(f"Missing columns: {', '.join(missing)}. Also failed to re-download data.")
                    print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                    continue
                df = temp_df.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Re-check for missing columns after download
                missing = [c for c in required if c not in df.columns]
                if missing:
                    reasons.append(f"Missing columns after re-download: {', '.join(missing)}.")
                    print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                    continue

            # Proceed with strategy calculations
            recent_period = min(len(df), 90)
            swing_low_idx = df['Low'].iloc[-recent_period:].idxmin()
            if pd.isna(swing_low_idx):
                reasons.append("No valid swing low found.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            anchor_date = swing_low_idx
            df_slice = df.loc[anchor_date:].copy()
            df_slice['Typical_Price'] = (df_slice['High'] + df_slice['Low'] + df_slice['Close']) / 3
            df_slice['Cum_TP_Vol'] = (df_slice['Typical_Price'] * df_slice['Volume']).cumsum()
            df_slice['Cum_Vol'] = df_slice['Volume'].cumsum()
            df_slice['AnchoredVWAP'] = df_slice['Cum_TP_Vol'] / df_slice['Cum_Vol']

            overlap_index = df_slice.index.intersection(df.index)
            df['AnchoredVWAP'] = np.nan
            df.loc[overlap_index, 'AnchoredVWAP'] = df_slice.loc[overlap_index, 'AnchoredVWAP']
            df['AnchoredVWAP'].ffill(inplace=True)

            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['EMA_5'] = compute_ema(df['Close'], 5)
            df['EMA_10'] = compute_ema(df['Close'], 10)
            df['RSI'] = compute_rsi(df['Close'], 14)
            df['ATR'] = compute_atr(df, 14)
            df['Avg_Volume'] = df['Volume'].rolling(window=20, min_periods=1).mean()

            latest = df.iloc[-1]
            missing_indicators = [col for col in ['AnchoredVWAP','SMA_50','EMA_5','EMA_10'] if pd.isna(latest[col])]
            if missing_indicators:
                reasons.append(f"Missing indicators: {', '.join(missing_indicators)}.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                # Attempt fallback for AnchoredVWAP
                if 'AnchoredVWAP' in missing_indicators:
                    df.loc[df.index[-1], 'AnchoredVWAP'] = df.loc[df.index[-1], 'Close']
                    latest = df.iloc[-1]  
                    missing_indicators = [col for col in ['AnchoredVWAP','SMA_50','EMA_5','EMA_10'] if pd.isna(latest[col])]
                    if not missing_indicators:
                        reasons = []
                if missing_indicators:
                    print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                    continue

            current_price = latest['Close']
            price_above_vwap = current_price > latest['AnchoredVWAP']
            price_above_sma50 = current_price > latest['SMA_50']
            ema_crossover = latest['EMA_5'] > latest['EMA_10']
            rsi_ok = 50 <= latest['RSI'] <= 85
            volume_ok = latest['Volume'] > latest['Avg_Volume']

            buy_signal = (price_above_vwap and price_above_sma50 and ema_crossover and rsi_ok and volume_ok)

            if buy_signal:
                print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - VWAP Swing Trade Buy Signal.")
                atr_val = latest['ATR'] if not pd.isna(latest['ATR']) else 0
                stop_loss = current_price - (1.5 * atr_val)
                profit_target = current_price + (2 * (current_price - stop_loss))
                expected_gain = ((profit_target - current_price) / current_price) * 100

                print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
                print(f"  Buy at:        ${current_price:.2f}")
                print(f"  Stop Loss:     ${stop_loss:.2f} (1.5 * ATR below entry)")
                print(f"  Profit Target: ${profit_target:.2f} (2:1 R:R)")
                print(f"  Expected Gain: {expected_gain:.2f}%")

                if not only_buy_signals:
                    try:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            price_paid_input = input(
                                f"Enter price paid for {symbol} (default is current price ${current_price:.2f}): "
                            )
                            price_paid = float(price_paid_input) if price_paid_input else current_price
                            number_of_shares = int(input("Enter number of shares: "))
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': number_of_shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'VWAP'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(f"{Colors.GREEN}Added {symbol} to holdings with stop loss={stop_loss:.2f}, profit target={profit_target:.2f}.{Colors.RESET}")
                    except ValueError:
                        print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
            else:
                if not price_above_vwap:
                    reasons.append("Price not above anchored VWAP.")
                if not price_above_sma50:
                    reasons.append("Price not above 50-day SMA.")
                if not ema_crossover:
                    reasons.append("EMA 5 not above EMA 10.")
                if not rsi_ok:
                    reasons.append("RSI not in [50, 85].")
                if not volume_ok:
                    reasons.append("Volume not above average.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")

        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")

###################################################
# Patterns from rtscli_strategies_patterns (imported)
###################################################
# We won't rewrite those here, but we must allow them to accept `data_cache`


###################################################
# ENHANCED ANCHORED VWAP STRATEGY
###################################################
def run_anchored_vwap_strategy(only_buy_signals=False, watchlist_name=None, data_cache=None):
    def safe_scalar(val):
        return val if pd.notna(val) else None

    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks first.{Colors.RESET}")
        return

    base_pivot_lookback = 5
    monthly_vol_threshold = 0.05

    def is_bullish_engulfing(df):
        if len(df) < 2:
            return False
        prev_open = df['Open'].iloc[-2]
        prev_close = df['Close'].iloc[-2]
        curr_open = df['Open'].iloc[-1]
        curr_close = df['Close'].iloc[-1]
        return (curr_open < prev_close) and (curr_close > prev_open)

    def is_hammer(df):
        if len(df) < 1:
            return False
        row = df.iloc[-1]
        body = abs(row['Close'] - row['Open'])
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        upper_wick = row['High'] - max(row['Open'], row['Close'])
        return (lower_wick > 2 * body) and (upper_wick < body)

    def any_bullish_pattern(df):
        return is_bullish_engulfing(df) or is_hammer(df)

    def download_data(symbol, interval, period="2y"):
        try:
            df_ = yf.download(symbol, period=period, interval=interval, progress=False, group_by='column')
            if not df_.empty:
                df_ = df_[~df_.index.duplicated(keep='first')]
            return df_
        except Exception:
            return pd.DataFrame()

    def find_pivot_lows(df, lookback=5, num_pivots=2):
        if df.empty:
            return []
        pivot_dates = []
        tmp = df.copy()
        for _ in range(num_pivots):
            rolling_min = tmp['Close'].rolling(window=lookback).min()
            pivot_idx = rolling_min.idxmin()
            if isinstance(pivot_idx, pd.Series):
                pivot_idx = pivot_idx.iloc[0]
            if pd.isna(pivot_idx) or pivot_idx not in tmp.index:
                break
            pivot_dates.append(pivot_idx)
            pivot_loc = tmp.index.get_loc(pivot_idx)
            mask_start = max(0, pivot_loc - lookback)
            mask_end = min(len(tmp), pivot_loc + lookback)
            tmp.iloc[mask_start:mask_end] = np.nan
        return pivot_dates

    def anchored_vwap(df, anchor_date):
        import pandas as pd

        # Ensure flat columns if MultiIndex columns exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure DataFrame index is unique
        df = df[~df.index.duplicated(keep='first')]

        # If anchor_date is a Series, pick the first element
        if isinstance(anchor_date, pd.Series):
            anchor_date = anchor_date.iloc[0]

        # Ensure anchor_date exists in the DataFrame, adjust if necessary
        if anchor_date not in df.index:
            valid_idx = df.index[df.index >= anchor_date]
            if len(valid_idx) == 0:
                return pd.Series(index=df.index, dtype=float)
            anchor_date = valid_idx[0]

        # Create a slice from anchor_date onward
        slice_ = df.loc[anchor_date:].copy()
        if slice_.empty:
            return pd.Series(index=df.index, dtype=float)

        # Calculate Typical Price and cumulative sums for VWAP calculation
        tp = (slice_['High'] + slice_['Low'] + slice_['Close']) / 3
        cumsum_tp_vol = (tp * slice_['Volume']).cumsum()
        cumsum_vol = slice_['Volume'].cumsum()
        vwap_vals = cumsum_tp_vol / cumsum_vol

        # Reindex the VWAP values to the original DataFrame's index and forward-fill
        return vwap_vals.reindex(df.index).ffill()

    for symbol in stock_list:
        reasons = []
        try:
            # 1) Download monthly data for volatility check
            monthly_df_for_vol = download_data(symbol, interval="1mo", period="5y")
            if not monthly_df_for_vol.empty and len(monthly_df_for_vol) >= 5:
                monthly_df_for_vol['Returns'] = monthly_df_for_vol['Close'].pct_change()
                vol = monthly_df_for_vol['Returns'].std()
                if vol > monthly_vol_threshold:
                    base_pivot_lookback = 8

            # 2) Weighted monthly anchored VWAP
            monthly_df = download_data(symbol, interval="1mo", period="5y")
            if monthly_df.empty or len(monthly_df) < base_pivot_lookback:
                reasons.append("Not enough monthly data.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            monthly_pivots = find_pivot_lows(monthly_df, lookback=base_pivot_lookback, num_pivots=2)
            if not monthly_pivots:
                reasons.append("No valid monthly pivots found.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            monthly_total_vwap = pd.Series(0, index=monthly_df.index, dtype=float)
            weights = [0.3, 0.7]
            for i, piv_date in enumerate(monthly_pivots):
                wvwap = anchored_vwap(monthly_df, piv_date)
                monthly_total_vwap += wvwap * weights[i]
            monthly_df['AnchoredVWAP_Monthly'] = monthly_total_vwap

            # 3) Weighted weekly anchored VWAP
            weekly_df = download_data(symbol, interval="1wk", period="2y")
            if weekly_df.empty or len(weekly_df) < 8:
                reasons.append("Not enough weekly data.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            weekly_pivots = find_pivot_lows(weekly_df, lookback=8, num_pivots=2)
            if not weekly_pivots:
                reasons.append("No valid weekly pivots found.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            weekly_total_vwap = pd.Series(0, index=weekly_df.index, dtype=float)
            weights = [0.4, 0.6]
            for i, piv_date in enumerate(weekly_pivots):
                wvwap = anchored_vwap(weekly_df, piv_date)
                weekly_total_vwap += wvwap * weights[i]
            weekly_df['AnchoredVWAP_Weekly'] = weekly_total_vwap

            # 4) Weighted daily anchored VWAP
            daily_df = download_data(symbol, interval="1d", period="1y")
            if daily_df.empty or len(daily_df) < 20:
                reasons.append("Not enough daily data.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            daily_pivots = find_pivot_lows(daily_df, lookback=20, num_pivots=2)
            if not daily_pivots:
                reasons.append("No valid daily pivots found.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            daily_total_vwap = pd.Series(0, index=daily_df.index, dtype=float)
            weights = [0.5, 0.5]
            for i, piv_date in enumerate(daily_pivots):
                dvwap = anchored_vwap(daily_df, piv_date)
                daily_total_vwap += dvwap * weights[i]
            daily_df['AnchoredVWAP_Daily'] = daily_total_vwap

            # Safely extract final close/VWAP to scalars
            monthly_close = safe_scalar(monthly_df['Close'].iloc[-1])
            monthly_awv   = safe_scalar(monthly_df['AnchoredVWAP_Monthly'].iloc[-1])
            weekly_close  = safe_scalar(weekly_df['Close'].iloc[-1])
            weekly_awv    = safe_scalar(weekly_df['AnchoredVWAP_Weekly'].iloc[-1])
            daily_close   = safe_scalar(daily_df['Close'].iloc[-1])
            daily_awv     = safe_scalar(daily_df['AnchoredVWAP_Daily'].iloc[-1])

            # Validate we have numeric data
            if any(x is None for x in [monthly_close, monthly_awv, weekly_close, weekly_awv, daily_close, daily_awv]):
                reasons.append("Missing monthly/weekly/daily close or VWAP values.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Evaluate multi-timeframe checks
            if monthly_close <= monthly_awv:
                reasons.append("Price below Weighted Monthly Anchored VWAP.")
            if weekly_close <= weekly_awv:
                reasons.append("Price below Weighted Weekly Anchored VWAP.")
            if daily_close <= daily_awv:
                reasons.append("Price below Weighted Daily Anchored VWAP.")

            # Quick candlestick check on daily
            if len(daily_df) >= 2:
                candlestick_ok = any_bullish_pattern(daily_df.tail(2))
            else:
                candlestick_ok = False

            if not candlestick_ok:
                reasons.append("No bullish candlestick signal on daily chart.")

            buy_signal = (len(reasons) == 0)

            if buy_signal:
                print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - Multi-Timeframe Anchored VWAP Buy Signal")

                # Compute ATR on daily
                daily_df['ATR'] = compute_atr(daily_df, 14)
                if not daily_df['ATR'].isna().all():
                    daily_atr_latest = daily_df['ATR'].iloc[-1]
                else:
                    daily_atr_latest = 0.0

                stop_loss = daily_close - 1.5 * daily_atr_latest
                risk_ps = daily_close - stop_loss
                profit_target = daily_close + (2 * risk_ps)
                exp_gain_pct = ((profit_target - daily_close) / daily_close) * 100.0

                print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
                print(f"  Current Price:    ${daily_close:.2f}")
                print(f"  Stop Loss:        ${stop_loss:.2f} (1.5 x Daily ATR)")
                print(f"  Profit Target:    ${profit_target:.2f} (2:1 R:R)")
                print(f"  Expected Gain:    {exp_gain_pct:.2f}%")

                if not only_buy_signals:
                    try:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            price_paid_str = input(f"Enter price paid (default = ${daily_close:.2f}): ")
                            price_paid = float(price_paid_str) if price_paid_str else daily_close
                            shares = int(input("Enter number of shares: ") or 0)
                            holding = {
                                'symbol': symbol,
                                'price_paid': price_paid,
                                'shares': shares,
                                'stop_loss': stop_loss,
                                'profit_target': profit_target,
                                'strategy': 'AnchoredVWAP'
                            }
                            holdings.append(holding)
                            save_holdings(holdings)
                            print(f"{Colors.GREEN}Added {symbol} with SL={stop_loss:.2f}, PT={profit_target:.2f}.{Colors.RESET}")
                    except ValueError:
                        print(f"{Colors.RED}Invalid input. Skipping addition.{Colors.RESET}")
            else:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")

        except Exception as exc:
            print(f"{Colors.RED}{symbol}: No buy signal - {exc}{Colors.RESET}")
            continue

###################################################
# RUN ALL STRATEGIES
###################################################
def run_all_strategies():
    watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        print(f"{Colors.RED}Your {watchlist_name} watchlist is empty. Please add stocks first.{Colors.RESET}")
        return

    print("\nRunning All Strategies...")

    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=2)

    global SHARED_DATA
    SHARED_DATA.clear()

    # Fetch each ticker once, store in SHARED_DATA
    for symbol in stock_list:
        try:
            df = get_or_update_stock_data(symbol, start_date, end_date)
            if df.empty:
                print(f"{Colors.YELLOW}{symbol}: Skipping empty dataset from yfinance.{Colors.RESET}")
                continue
            SHARED_DATA[symbol] = df
        except PermissionError as pe:
            print(f"{Colors.RED}Permission error downloading {symbol}: {pe}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error downloading {symbol} - {e}{Colors.RESET}")

    print("\nSwing Trading Strategy Buy Signals:")
    run_swing_strategy_console(
        only_buy_signals=False, 
        watchlist_name=watchlist_name, 
        data_cache=SHARED_DATA
    )

    print("\nPullback Strategy Buy Signals:")
    run_pullback_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nBreakout Strategy Buy Signals:")
    run_breakout_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nVWAP Swing Trade Strategy Buy Signals:")
    run_vwap_swing_trade_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nInverse Head and Shoulders Strategy Buy Signals:")
    run_inverse_head_and_shoulders_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nRectangle Pattern Strategy Buy Signals:")
    run_rectangle_pattern_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nAscending Channel Strategy Buy Signals:")
    run_ascending_channel_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    print("\nAnchored VWAP Strategy Buy Signals:")
    run_anchored_vwap_strategy(
        only_buy_signals=False,
        watchlist_name=watchlist_name,
        data_cache=SHARED_DATA
    )

    input("\nAnalysis complete. Press Enter to return to the main menu...")




###################################################
# 100-BAGGER COMPLEX STRATEGY (NOT in run_all_strategies)
###################################################
def run_100bagger_complex_strategy(only_buy_signals=False, watchlist_name=None):
    """
    This complex strategy identifies potential 100-bagger stocks based on
    multi-timeframe uptrends (monthly, weekly) and advanced daily signals
    like anchored VWAP, RSI, MACD, Bollinger Bands, and Heikin Ashi candles.
    """
    # Assuming Colors, select_watchlist, load_watchlist, etc., are defined globally
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist '{watchlist_name}' is empty. Please add stocks first.{Colors.RESET}")
        return

    for symbol in stock_list:
        reasons = []
        try:
            # Calculate start and end dates for 52 months precisely
            end_date = datetime.now()
            start_date = end_date - relativedelta(months=52)

            # Download Monthly Data
            monthly_df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1mo",
                progress=False
            )
            if isinstance(monthly_df.columns, pd.MultiIndex):
                monthly_df.columns = monthly_df.columns.get_level_values(0)
            
            # Debug: Print number of months fetched
            num_months = len(monthly_df)
            print(f"{symbol}: Fetched {num_months} months of data.")
            logging.info(f"{symbol}: Fetched {num_months} months of monthly data.")

            if monthly_df.empty:
                reasons.append("Monthly data is empty.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            elif num_months < 50:
                reasons.append(f"Only {num_months} months of data available; 50 required.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Calculate EMAs on monthly data
            monthly_df['EMA_20'] = monthly_df['Close'].ewm(span=20, adjust=False).mean()
            monthly_df['EMA_50'] = monthly_df['Close'].ewm(span=50, adjust=False).mean()
            latest_monthly = monthly_df.iloc[-1]

            if not (latest_monthly['Close'] > latest_monthly['EMA_20'] > latest_monthly['EMA_50']):
                reasons.append("No monthly uptrend (Close > EMA_20 > EMA_50).")

            # Download Weekly Data
            weekly_df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1wk",
                progress=False
            )
            if isinstance(weekly_df.columns, pd.MultiIndex):
                weekly_df.columns = weekly_df.columns.get_level_values(0)
            
            if weekly_df.empty:
                reasons.append("Insufficient weekly data.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            weekly_df['EMA_20'] = weekly_df['Close'].ewm(span=20, adjust=False).mean()
            weekly_df['EMA_50'] = weekly_df['Close'].ewm(span=50, adjust=False).mean()
            latest_weekly = weekly_df.iloc[-1]

            if not (latest_weekly['Close'] > latest_weekly['EMA_20'] > latest_weekly['EMA_50']):
                reasons.append("No weekly uptrend (Close > EMA_20 > EMA_50).")

            if reasons:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Download Daily Data (1 year range)
            daily_df = yf.download(
                symbol,
                start=(end_date - relativedelta(years=1)).strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d",
                progress=False
            )
            if isinstance(daily_df.columns, pd.MultiIndex):
                daily_df.columns = daily_df.columns.get_level_values(0)
            
            if daily_df.empty or len(daily_df) < 60:
                reasons.append("Insufficient daily data.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Anchored VWAP from last ~60-day pivot low
            pivot_lookback = min(len(daily_df), 60)
            pivot_low_idx = daily_df['Low'].iloc[-pivot_lookback:].idxmin()
            if pd.isna(pivot_low_idx):
                reasons.append("No pivot low found for anchored VWAP.")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            anchor_slice = daily_df.loc[pivot_low_idx:].copy()
            anchor_slice['TP'] = (anchor_slice['High'] + anchor_slice['Low'] + anchor_slice['Close']) / 3
            anchor_slice['Cum_TP_Vol'] = (anchor_slice['TP'] * anchor_slice['Volume']).cumsum()
            anchor_slice['Cum_Vol'] = anchor_slice['Volume'].cumsum()
            anchor_slice['AnchoredVWAP'] = anchor_slice['Cum_TP_Vol'] / anchor_slice['Cum_Vol']

            daily_df['AnchoredVWAP'] = np.nan
            daily_df.loc[anchor_slice.index, 'AnchoredVWAP'] = anchor_slice['AnchoredVWAP']
            daily_df['AnchoredVWAP'].ffill(inplace=True)

            # Calculate indicators on daily data
            daily_df['RSI_14'] = 100.0 - (100.0 / (1.0 + calculate_rsi(daily_df['Close'], 14)))
            fast_ema = daily_df['Close'].ewm(span=12, adjust=False).mean()
            slow_ema = daily_df['Close'].ewm(span=26, adjust=False).mean()
            daily_df['MACD'] = fast_ema - slow_ema
            daily_df['MACD_Signal'] = daily_df['MACD'].ewm(span=9, adjust=False).mean()
            daily_df['ATR_14'] = compute_atr(daily_df, 14)
            bb_mid = daily_df['Close'].rolling(window=20).mean()
            bb_std = daily_df['Close'].rolling(window=20).std()
            daily_df['BB_Upper'] = bb_mid + 2 * bb_std
            daily_df['BB_Lower'] = bb_mid - 2 * bb_std

            # Heikin Ashi Candles calculation
            ha_df = daily_df.copy()
            ha_df['HA_Close'] = (daily_df['Open'] + daily_df['High'] + daily_df['Low'] + daily_df['Close']) / 4
            ha_open_vals = [(daily_df['Open'].iloc[0] + daily_df['Close'].iloc[0]) / 2]
            for i in range(1, len(daily_df)):
                ha_open_vals.append((ha_open_vals[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
            ha_df['HA_Open'] = ha_open_vals
            daily_df['HA_Close'] = ha_df['HA_Close']
            daily_df['HA_Open'] = ha_df['HA_Open']

            latest_row = daily_df.iloc[-1]
            if latest_row['Close'] <= latest_row['AnchoredVWAP']:
                reasons.append("Daily price below anchored VWAP.")
            if latest_row['RSI_14'] >= 70:
                reasons.append("RSI >= 70 (overbought).")
            if latest_row['MACD'] <= latest_row['MACD_Signal']:
                reasons.append("MACD <= Signal (not bullish).")
            if latest_row['HA_Close'] <= latest_row['HA_Open']:
                reasons.append("Heikin Ashi not bullish (HA_Close <= HA_Open).")

            daily_df['EMA_10'] = daily_df['Close'].ewm(span=10, adjust=False).mean()
            if latest_row['Close'] < latest_row['EMA_10']:
                reasons.append("Price < EMA_10.")

            if reasons:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Construct Buy Signal
            daily_price = latest_row['Close']
            atr_val = latest_row['ATR_14'] if pd.notna(latest_row['ATR_14']) else 0
            stop_loss = daily_price - (1.5 * atr_val)
            profit_target = daily_price + (2 * (daily_price - stop_loss))

            print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - 100-Bagger Complex Strategy Buy Signal")
            print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
            print(f"  Current Price:   ${daily_price:.2f}")
            print(f"  AnchoredVWAP:    ${latest_row['AnchoredVWAP']:.2f}")
            print(f"  Stop Loss:       ${stop_loss:.2f}  (1.5 x ATR below)")
            print(f"  Profit Target:   ${profit_target:.2f}  (2:1 Reward-to-Risk)")
            print(f"  RSI_14:          {latest_row['RSI_14']:.2f}")
            print(f"  MACD:            {latest_row['MACD']:.2f}, Signal: {latest_row['MACD_Signal']:.2f}")

            if not only_buy_signals:
                try:
                    add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                    if add_holding == 'yes':
                        holdings = load_holdings()
                        price_paid_input = input(f"Enter price paid for {symbol} (default = ${daily_price:.2f}): ")
                        price_paid = float(price_paid_input) if price_paid_input else daily_price
                        shares = int(input("Enter number of shares: "))
                        holding_entry = {
                            'symbol': symbol,
                            'price_paid': price_paid,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'profit_target': profit_target,
                            'strategy': '100Bagger'
                        }
                        holdings.append(holding_entry)
                        save_holdings(holdings)
                        print(f"{Colors.GREEN}Added {symbol} to holdings with stop_loss=${stop_loss:.2f}, profit_target=${profit_target:.2f}.{Colors.RESET}")
                        logging.info(f"{symbol}: Added to holdings.")
                except ValueError:
                    print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
                    logging.warning(f"{symbol}: Invalid input during holding addition.")

        except Exception as exc:
            print(f"{Colors.RED}{symbol}: No buy signal - {exc}{Colors.RESET}")
            logging.error(f"{symbol}: No buy signal - {exc}")
            continue


###################################################
# MAIN
###################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTS CLI Strategies Tool")
    parser.add_argument('--strategy', type=str, required=True,
                        help="Strategy to run: Swing, Pullback, Breakout, VWAP, AnchoredVWAP, All, "
                             "InverseH&S, Rectangle, AscendingChannel, 100bagger")
    parser.add_argument('--watchlist', type=str, default=None, help="Watchlist name to use")
    parser.add_argument('--only-buy-signals', action='store_true',
                        help="Run strategies to display buy signals without adding to holdings")
    parser.add_argument('--view-history', action='store_true', help="View summarized trade history")
    args = parser.parse_args()

    if args.view_history:
        holdings = load_holdings()
        view_trade_history(holdings)
    else:
        strategy = args.strategy.lower()
        if strategy == "all":
            run_all_strategies()
        elif strategy == "swing":
            run_swing_strategy_console(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy == "pullback":
            run_pullback_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy == "breakout":
            run_breakout_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy == "vwap":
            run_vwap_swing_trade_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy == "anchoredvwap":
            run_anchored_vwap_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy in ["inversehs", "inverseheadandshoulders"]:
            run_inverse_head_and_shoulders_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy in ["rectangle", "rectanglepattern"]:
            run_rectangle_pattern_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy in ["ascendingchannel", "ascending"]:
            run_ascending_channel_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        elif strategy == "100bagger":
            run_100bagger_complex_strategy(only_buy_signals=args.only_buy_signals, watchlist_name=args.watchlist)
        else:
            print(f"{Colors.RED}Unknown strategy '{args.strategy}'. Please specify a valid strategy.{Colors.RESET}")