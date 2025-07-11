# rtscli_strategies.py

import yfinance as yf
import pandas as pd
import numpy as np
import stock_forecast
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from rtscli_utils import load_holdings, save_holdings, load_watchlist, select_watchlist

# Import the new pattern-based strategies
from rtscli_strategies_patterns import (
    run_inverse_head_and_shoulders_strategy,
    run_rectangle_pattern_strategy,
    run_ascending_channel_strategy
)

# ANSI escape sequences for colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'


###################################################
# SHARED FUNCTIONS (Integrated from second script)
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
    """
    Basic VWAP calculation for reference or simpler VWAP-based strategies.
    """
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap


###################################################
# Additional Utility Functions
###################################################
# Trade History Viewing Function
def view_trade_history(trades):
    """
    Prints a summarized overall trade history in the requested format:
      Symbol  Entry Date  Exit Date   Entry Price Exit Price  Shares  Strategy    Result
    """
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
# Strategy Functions (From First Script and Enhanced)
###################################################

def run_swing_strategy_console(only_buy_signals=False, watchlist_name=None):
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    ema_periods = [7, 21, 50]
    buy_signals = []  # List to store buy signal tickers

    for symbol in stock_list:
        reasons = []
        try:
            # Adjust date range
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(years=2)
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                group_by='column'
            )

            # Flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 252:
                reasons.append("Not enough data to perform calculations.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
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
                buy_signals.append(symbol)

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
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            reason_str = "; ".join(reasons)
            print(f"{Colors.RED}An error occurred while processing {Colors.MAGENTA}{symbol}{Colors.RED}: {reason_str}{Colors.RESET}")

def run_pullback_strategy(only_buy_signals=False, watchlist_name=None):
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
            index_data = yf.download(index, period="2y", progress=False, group_by='column')
            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)
            if index_data.empty:
                print(f"{Colors.RED}Could not retrieve data for {index}.{Colors.RESET}")
                continue

            index_data['50_MA'] = index_data['Close'].rolling(window=50).mean()
            latest_close = index_data['Close'].iloc[-1]
            ma_50 = index_data['50_MA'].iloc[-1]

            if isinstance(latest_close, pd.Series):
                latest_close = latest_close.iloc[0]
            if isinstance(ma_50, pd.Series):
                ma_50 = ma_50.iloc[0]

            if pd.notna(latest_close) and pd.notna(ma_50):
                if latest_close > ma_50:
                    market_uptrend_count += 1
            else:
                print(f"{Colors.YELLOW}Skipping {index} due to missing data for moving averages.{Colors.RESET}")
                continue
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

    qualifying_stocks = []
    buy_signals = []  # Initialize buy_signals

    for symbol in stock_list:
        reasons = []
        try:
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(months=6)
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                group_by='column'
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 126:
                reasons.append("Not enough data to perform calculations.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
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
                return (prev_close < prev_open) and (curr_close > curr_open) and (curr_open > prev_close) and (curr_close < prev_open)

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
                qualifying_stocks.append(symbol)
                buy_signals.append(symbol)
                entry_price = current_price
                entry_day_low = df['Low'].iloc[-1]
                ema_10 = df['10_EMA'].iloc[-1]
                stop_loss_candidate1 = entry_day_low * 0.995
                stop_loss_candidate2 = ema_10
                stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
                risk_per_share = entry_price - stop_loss
                profit_target = entry_price + (risk_per_share * 2)
                buy_signals.append(symbol)

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
            reason_str = "; ".join(reasons)
            print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
            continue

    # After processing all stocks
    if buy_signals and not only_buy_signals:
        choice = input("Would you like to run 'Predict Probability' on these buy signal tickers? (yes/no): ").lower()
        if choice == 'yes':
            for ticker in buy_signals:
                try:
                    hist_data = stock_forecast.get_stock_data(ticker)
                    prob = stock_forecast.predict_probability(hist_data, ticker)
                    if prob is not None:
                        print(f"\n--- Predict Probability for {ticker} ---")
                        print("1 ATR Move:")
                        print(f"Bullish Target (1 ATR): ${prob['bullish_1ATR']:.2f}")
                        print(f"Bearish Target (1 ATR): ${prob['bearish_1ATR']:.2f}")
                        print(f"Probability Bullish First (1 ATR): {prob['prob_up_1']:.2f}%")
                        print(f"Probability Bearish First (1 ATR): {prob['prob_down_1']:.2f}%")
                        print(prob['likely_1ATR'])

                        print("\n3 ATR Move:")
                        print(f"Bullish Target (3 ATR): ${prob['bullish_3ATR']:.2f}")
                        print(f"Bearish Target (3 ATR): ${prob['bearish_3ATR']:.2f}")
                        print(f"Probability Bullish First (3 ATR): {prob['prob_up_3']:.2f}%")
                        print(f"Probability Bearish First (3 ATR): {prob['prob_down_3']:.2f}%")
                        print(prob['likely_3ATR'])

                        print("\n5 ATR Move:")
                        print(f"Bullish Target (5 ATR): ${prob['bullish_5ATR']:.2f}")
                        print(f"Bearish Target (5 ATR): ${prob['bearish_5ATR']:.2f}")
                        print(f"Probability Bullish First (5 ATR): {prob['prob_up_5']:.2f}%")
                        print(f"Probability Bearish First (5 ATR): {prob['prob_down_5']:.2f}%")
                        print(prob['likely_5ATR'])

                        print(f"\nOverall Prediction: {prob['overall_prediction']}")
                    else:
                        print(f"Not enough data for Predict Probability on {ticker}.")
                except Exception as e:
                    print(f"Error predicting probability for {ticker}: {e}")
        input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")


def run_breakout_strategy(only_buy_signals=False, watchlist_name=None):
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
            index_data = yf.download(index, period="2y", progress=False, group_by='column')
            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)
            if index_data.empty:
                print(f"{Colors.RED}Could not retrieve data for {index}.{Colors.RESET}")
                continue

            index_data['50_MA'] = index_data['Close'].rolling(window=50).mean()
            latest_close = index_data['Close'].iloc[-1]
            ma_50 = index_data['50_MA'].iloc[-1]

            if isinstance(latest_close, pd.Series):
                latest_close = latest_close.iloc[0]
            if isinstance(ma_50, pd.Series):
                ma_50 = ma_50.iloc[0]

            if pd.notna(latest_close) and pd.notna(ma_50):
                if latest_close > ma_50:
                    market_uptrend_count += 1
            else:
                print(f"{Colors.YELLOW}Skipping {index} due to missing data for moving averages.{Colors.RESET}")
                continue
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

    qualifying_stocks = []
    buy_signals = []

    for symbol in stock_list:
        reasons = []
        try:
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(months=6)
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                group_by='column'
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 60:
                reasons.append("Not enough data to perform calculations.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
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

                volume = df['Volume'].rolling(window=10).mean()
                recent_volume = volume.iloc[-1]
                previous_volume = volume.iloc[-11]
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
                qualifying_stocks.append(symbol)
                buy_signals.append(symbol)
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
                reason_str = "; ".join(reasons) if reasons else "Does not meet criteria."
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            reason_str = "; ".join(reasons)
            print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
            continue

    # After processing all stocks
    if buy_signals and not only_buy_signals:
        choice = input("Would you like to run 'Predict Probability' on these buy signal tickers? (yes/no): ").lower()
        if choice == 'yes':
            for ticker in buy_signals:
                try:
                    hist_data = stock_forecast.get_stock_data(ticker)
                    prob = stock_forecast.predict_probability(hist_data, ticker)
                    if prob is not None:
                        print(f"\n--- Predict Probability for {ticker} ---")
                        # Print out the typical details from your model...
                        print(f"Probability of Success: {prob * 100:.2f}%")
                    else:
                        print(f"Not enough data for Predict Probability on {ticker}.")
                except Exception as ex:
                    print(f"Error predicting probability for {ticker}: {ex}")
        input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")


def run_vwap_swing_trade_strategy(only_buy_signals=False, watchlist_name=None):
    """
    Runs a VWAP Swing Trade Strategy, anchoring VWAP to the most significant swing low
    in the last ~6 months of data. Mirroring the backtest code.
    """
    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    buy_signals = []

    for symbol in stock_list:
        reasons = []
        try:
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(months=6)
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                group_by='column'
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 30:
                reasons.append("Not enough data to perform calculations.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                reasons.append(f"Missing columns: {', '.join(missing_columns)}.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            # 1) Identify the "Most Significant Swing Low" in the last X bars
            #    For demonstration, let's define "swing low" as the minimum Low
            #    in the last ~3 months. You can refine this logic as needed.
            recent_period = min(len(df), 90)  # look back ~3 months
            swing_low_idx = df['Low'].iloc[-recent_period:].idxmin()
            if pd.isna(swing_low_idx):
                reasons.append("No valid swing low found.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            # 2) Compute anchored VWAP from that swing low date
            anchor_date = swing_low_idx
            df_slice = df.loc[anchor_date:].copy()
            df_slice['Typical_Price'] = (df_slice['High'] + df_slice['Low'] + df_slice['Close']) / 3
            df_slice['Cum_TP_Vol'] = (df_slice['Typical_Price'] * df_slice['Volume']).cumsum()
            df_slice['Cum_Vol'] = df_slice['Volume'].cumsum()
            df_slice['AnchoredVWAP'] = df_slice['Cum_TP_Vol'] / df_slice['Cum_Vol']

            # Merge anchored VWAP back into the main df
            df['AnchoredVWAP'] = np.nan
            df.loc[df_slice.index, 'AnchoredVWAP'] = df_slice['AnchoredVWAP']
            df['AnchoredVWAP'].ffill(inplace=True)

            # 3) Additional Indicators
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_5'] = compute_ema(df['Close'], 5)
            df['EMA_10'] = compute_ema(df['Close'], 10)
            df['RSI'] = compute_rsi(df['Close'], 14)
            df['ATR'] = compute_atr(df, 14)
            df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

            latest = df.iloc[-1]
            if pd.isna(latest['AnchoredVWAP']) or pd.isna(latest['SMA_50']) or pd.isna(latest['EMA_5']) or pd.isna(latest['EMA_10']):
                reasons.append("Missing indicators.")
                buy_signal = False
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
                continue

            current_price = latest['Close']
            price_above_vwap = current_price > latest['AnchoredVWAP']
            price_above_sma50 = current_price > latest['SMA_50']
            ema_crossover = latest['EMA_5'] > latest['EMA_10']
            rsi_ok = 50 <= latest['RSI'] <= 85
            volume_ok = latest['Volume'] > latest['Avg_Volume']

            buy_signal = price_above_vwap and price_above_sma50 and ema_crossover and rsi_ok and volume_ok

            if buy_signal:
                print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - VWAP Swing Trade Buy Signal (Anchored to last major swing low).")
                atr_val = latest['ATR'] if not pd.isna(latest['ATR']) else 0
                stop_loss = current_price - (1.5 * atr_val)
                profit_target = current_price + (2 * (current_price - stop_loss))
                expected_gain = ((profit_target - current_price) / current_price) * 100

                print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
                print(f"  Buy at:        ${current_price:.2f}")
                print(f"  Stop Loss:     ${stop_loss:.2f} (1.5 * ATR below entry)")
                print(f"  Profit Target: ${profit_target:.2f} (2:1 R:R)")
                print(f"  Expected Gain: {expected_gain:.2f}%")

                buy_signals.append(symbol)

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
                            print(
                                f"{Colors.GREEN}Added {symbol} to holdings with stop loss at ${stop_loss:.2f} "
                                f"and profit target at ${profit_target:.2f}.{Colors.RESET}"
                            )
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
                reason_str = "; ".join(reasons)
                print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
        except Exception as e:
            reasons.append(f"An error occurred: {e}")
            reason_str = "; ".join(reasons)
            print(f"{Colors.RED}{symbol}: No buy signal - {reason_str}{Colors.RESET}")
            continue

    # Optional: Prompt for Probability
    if buy_signals and not only_buy_signals:
        choice = input("Would you like to run 'Predict Probability' on these buy signal tickers? (yes/no): ").lower()
        if choice == 'yes':
            for ticker in buy_signals:
                try:
                    hist_data = stock_forecast.get_stock_data(ticker)
                    prob = stock_forecast.predict_probability(hist_data, ticker)
                    if prob is not None:
                        print(f"\n--- Predict Probability for {ticker} ---")
                        # Print out the typical details from your model...
                        print(f"Probability of Success: {prob * 100:.2f}%")
                    else:
                        print(f"Not enough data for Predict Probability on {ticker}.")
                except Exception as ex:
                    print(f"Error predicting probability for {ticker}: {ex}")
        input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")



###################################################
# Unified Enhanced Anchored VWAP Strategy
###################################################
def run_anchored_vwap_strategy(only_buy_signals=False, watchlist_name=None):
    """
    Enhanced Anchored VWAP Strategy (merged standard + advanced logic):
    - Uses multiple timeframes (Monthly, Weekly, Daily, 4H, 1H) to anchor VWAPs
      at multiple pivot points and apply Weighted Anchoring.
    - Incorporates candlestick patterns as an additional confirmation filter.
    - Demonstrates a dynamic parameter approach (adaptive lookback + placeholders for ML).
    """

    if watchlist_name is None:
        watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        if not only_buy_signals:
            print(f"{Colors.RED}Your watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    buy_signals = []

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
        curr_open = df['Open'].iloc[-1]
        curr_close = df['Close'].iloc[-1]
        curr_low = df['Low'].iloc[-1]
        curr_high = df['High'].iloc[-1]
        body = abs(curr_close - curr_open)
        lower_wick = min(curr_open, curr_close) - curr_low
        upper_wick = curr_high - max(curr_open, curr_close)
        return (lower_wick > 2 * body) and (upper_wick < body)

    def any_bullish_pattern(df):
        return is_bullish_engulfing(df) or is_hammer(df)

    def download_data(symbol, interval, period="2y"):
        """
        Downloads data. Resamples to 4H if needed.
        """
        try:
            if interval == '4h':
                df_1h = yf.download(
                    symbol, period=period, interval="60m",
                    progress=False, group_by='column'
                )
                if df_1h.empty:
                    return pd.DataFrame()
                df_1h.index = df_1h.index.tz_localize(None)
                df_4h = df_1h.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna(subset=['Open','High','Low','Close'])
                return df_4h
            else:
                df_ = yf.download(
                    symbol, period=period, interval=interval,
                    progress=False, group_by='column'
                )
                return df_
        except Exception:
            return pd.DataFrame()

    def find_pivot_lows(df, lookback=5, num_pivots=2):
        if df.empty:
            return []
        pivot_dates = []
        temp_df = df.copy()
        for _ in range(num_pivots):
            rolling_min = temp_df['Close'].rolling(window=lookback).min()
            pivot_idxmin = rolling_min.idxmin()
            if pivot_idxmin not in temp_df.index:
                break
            pivot_dates.append(pivot_idxmin)
            pivot_loc = temp_df.index.get_loc(pivot_idxmin)
            mask_start = max(0, pivot_loc - lookback)
            mask_end = min(len(temp_df), pivot_loc + lookback)
            temp_df.iloc[mask_start:mask_end] = np.nan
        return pivot_dates

    def anchored_vwap(df, anchor_date):
        """
        Anchored VWAP from anchor_date to end of DataFrame
        """
        if anchor_date not in df.index:
            valid_dates = df.index[df.index >= anchor_date]
            if len(valid_dates) == 0:
                return pd.Series(index=df.index, dtype=float)
            anchor_date = valid_dates[0]

        df_slice = df.loc[anchor_date:]
        tp = (df_slice['High'] + df_slice['Low'] + df_slice['Close']) / 3
        cum_tp_vol = (tp * df_slice['Volume']).cumsum()
        cum_vol = df_slice['Volume'].cumsum()
        vwap_values = cum_tp_vol / cum_vol

        anchored_vwap_series = pd.Series(index=df.index, dtype=float)
        anchored_vwap_series.loc[df_slice.index] = vwap_values
        return anchored_vwap_series

    for symbol in stock_list:
        reasons = []
        try:
            # Quick volatility check on monthly data
            monthly_df_for_vol = download_data(symbol, interval="1mo", period="5y")
            if not monthly_df_for_vol.empty and len(monthly_df_for_vol) >= 5:
                monthly_df_for_vol['Returns'] = monthly_df_for_vol['Close'].pct_change()
                vol = monthly_df_for_vol['Returns'].std()
                if vol > monthly_vol_threshold:
                    base_pivot_lookback = 8

            # 1) Monthly
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
            for i, pivot_date in enumerate(monthly_pivots):
                single_vwap = anchored_vwap(monthly_df, pivot_date).fillna(method='ffill')
                monthly_total_vwap += single_vwap * weights[i]
            monthly_df['AnchoredVWAP_Monthly'] = monthly_total_vwap

            # 2) Weekly
            weekly_df = download_data(symbol, interval="1wk", period="2y")
            if weekly_df.empty:
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
            for i, pivot_date in enumerate(weekly_pivots):
                w_vwap = anchored_vwap(weekly_df, pivot_date).fillna(method='ffill')
                weekly_total_vwap += w_vwap * weights[i]
            weekly_df['AnchoredVWAP_Weekly'] = weekly_total_vwap

            # 3) Daily
            daily_df = download_data(symbol, interval="1d", period="1y")
            if daily_df.empty or len(daily_df) < 60:
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
            for i, pivot_date in enumerate(daily_pivots):
                d_vwap = anchored_vwap(daily_df, pivot_date).fillna(method='ffill')
                daily_total_vwap += d_vwap * weights[i]
            daily_df['AnchoredVWAP_Daily'] = daily_total_vwap

            # 4) Intraday 4H & 1H
            four_hour_df = download_data(symbol, interval="4h", period="3mo")
            one_hour_df = download_data(symbol, interval="60m", period="1mo")
            if four_hour_df.empty or one_hour_df.empty:
                reasons.append("Missing intraday data (4h or 1h).")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            anchor_date_1h = daily_pivots[-1]
            if anchor_date_1h not in one_hour_df.index:
                valid_intra = one_hour_df.index[one_hour_df.index >= anchor_date_1h]
                anchor_date_1h = valid_intra[0] if not valid_intra.empty else one_hour_df.index[0]

            one_hour_df['AnchoredVWAP'] = anchored_vwap(one_hour_df, anchor_date_1h)
            one_hour_df['RSI'] = compute_rsi(one_hour_df['Close'], 14)

            # Evaluate multi-timeframe
            monthly_close = monthly_df['Close'].iloc[-1]
            monthly_vwap_val = monthly_df['AnchoredVWAP_Monthly'].iloc[-1]
            if monthly_close <= monthly_vwap_val:
                reasons.append("Price below Weighted Monthly Anchored VWAP.")

            weekly_close = weekly_df['Close'].iloc[-1]
            weekly_vwap_val = weekly_df['AnchoredVWAP_Weekly'].iloc[-1]
            if weekly_close <= weekly_vwap_val:
                reasons.append("Price below Weighted Weekly Anchored VWAP.")

            daily_close = daily_df['Close'].iloc[-1]
            daily_vwap_val = daily_df['AnchoredVWAP_Daily'].iloc[-1]
            if daily_close <= daily_vwap_val:
                reasons.append("Price below Weighted Daily Anchored VWAP.")

            # Final intraday 1H check
            one_hour_latest = one_hour_df.iloc[-1]
            anchored_vwap_latest = one_hour_latest['AnchoredVWAP']
            rsi_latest = one_hour_latest['RSI']
            if one_hour_latest['Close'] <= anchored_vwap_latest:
                reasons.append("1H Price below anchored VWAP.")
            if not (40 < rsi_latest < 75):
                reasons.append(f"RSI not in [40, 75] range; RSI={rsi_latest:.2f}")

            # Candlestick confirmation on daily
            candlestick_ok = any_bullish_pattern(daily_df.tail(2))
            if not candlestick_ok:
                reasons.append("No bullish candlestick signal on daily chart.")

            buy_signal = (len(reasons) == 0)

            if buy_signal:
                current_price = one_hour_latest['Close']
                print(f"\n{Colors.MAGENTA}{symbol}{Colors.RESET} - Multi-Timeframe Anchored VWAP Buy Signal")

                daily_df['ATR'] = compute_atr(daily_df, 14)
                daily_atr_latest = daily_df['ATR'].iloc[-1] if not daily_df['ATR'].isna().all() else 0
                stop_loss = current_price - 1.5 * daily_atr_latest
                rps = current_price - stop_loss
                profit_target = current_price + (2 * rps)
                exp_gain_pct = (profit_target - current_price) / current_price * 100

                print(f"{Colors.GREEN}Buy Recommendation:{Colors.RESET}")
                print(f"  Current Price:    ${current_price:.2f}")
                print(f"  Stop Loss:        ${stop_loss:.2f} (1.5 x Daily ATR)")
                print(f"  Profit Target:    ${profit_target:.2f} (2:1 R:R)")
                print(f"  Expected Gain:    {exp_gain_pct:.2f}%")
                buy_signals.append(symbol)

                if not only_buy_signals:
                    try:
                        add_holding = input(f"Do you want to add {symbol} to your holdings? (yes/no): ").lower()
                        if add_holding == 'yes':
                            holdings = load_holdings()
                            price_paid_input = input(f"Enter price paid for {symbol} (default = ${current_price:.2f}): ")
                            price_paid = float(price_paid_input) if price_paid_input else current_price
                            shares = int(input("Enter number of shares: "))
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
                            print(
                                f"{Colors.GREEN}Added {symbol} to holdings with stop_loss={stop_loss:.2f}, "
                                f"profit_target={profit_target:.2f}.{Colors.RESET}"
                            )
                    except ValueError:
                        print(f"{Colors.RED}Invalid input. Skipping holding addition.{Colors.RESET}")
            else:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}{symbol}: No buy signal - {e}{Colors.RESET}")
            continue

    # Optional Probability Check
    if buy_signals and not only_buy_signals:
        choice = input("Would you like to run 'Predict Probability' on these buy signal tickers? (yes/no): ").lower()
        if choice == 'yes':
            for ticker in buy_signals:
                try:
                    hist_data = stock_forecast.get_stock_data(ticker)
                    prob = stock_forecast.predict_probability(hist_data, ticker)
                    if prob is not None:
                        print(f"\n--- Predict Probability for {ticker} ---")
                        # Typical details from your model...
                        print(f"Probability of Success: {prob * 100:.2f}%")
                    else:
                        print(f"Not enough data for Probability on {ticker}.")
                except Exception as ex:
                    print(f"Error predicting probability for {ticker}: {ex}")
        input(f"{Colors.YELLOW}\nPress Enter to return to the main menu...{Colors.RESET}")


###################################################
# RUN ALL STRATEGIES FUNCTION FOR "All" Option
###################################################
def run_all_strategies():
    """
    Runs all strategies for the selected watchlist.
    (Does NOT include 100-bagger strategy by design)
    """
    watchlist_name = select_watchlist()
    stock_list = load_watchlist(watchlist_name)
    if not stock_list:
        print(f"{Colors.RED}Your {watchlist_name} watchlist is empty. Please add stocks to your watchlist first.{Colors.RESET}")
        return

    print("\nRunning All Strategies...")
    print("\nSwing Trading Strategy Buy Signals:")
    run_swing_strategy_console(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nPullback Strategy Buy Signals:")
    run_pullback_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nBreakout Strategy Buy Signals:")
    run_breakout_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nVWAP Swing Trade Strategy Buy Signals:")
    run_vwap_swing_trade_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nInverse Head and Shoulders Strategy Buy Signals:")
    run_inverse_head_and_shoulders_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nRectangle Pattern Strategy Buy Signals:")
    run_rectangle_pattern_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nAscending Channel Strategy Buy Signals:")
    run_ascending_channel_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

    print("\nAnchored VWAP Strategy Buy Signals:")
    run_anchored_vwap_strategy(only_buy_signals=False, watchlist_name=watchlist_name)

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
    # Optional: Configure logging
    logging.basicConfig(
        filename='rtscli.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

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
            start_date = end_date - relativedelta(months=52)  # Exactly 52 months (~4 years and 4 months)

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
            
            # Debugging: Print number of months fetched
            num_months = len(monthly_df)
            print(f"{symbol}: Fetched {num_months} months of data.")
            logging.info(f"{symbol}: Fetched {num_months} months of monthly data.")

            if monthly_df.empty:
                reasons.append("Monthly data is empty.")
                logging.warning(f"{symbol}: No buy signal - {', '.join(reasons)}")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue
            elif num_months < 50:
                reasons.append(f"Only {num_months} months of data available; 50 required.")
                logging.warning(f"{symbol}: No buy signal - {', '.join(reasons)}")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Calculate EMAs
            monthly_df['EMA_20'] = monthly_df['Close'].ewm(span=20, adjust=False).mean()
            monthly_df['EMA_50'] = monthly_df['Close'].ewm(span=50, adjust=False).mean()
            latest_monthly = monthly_df.iloc[-1]

            if not (latest_monthly['Close'] > latest_monthly['EMA_20'] > latest_monthly['EMA_50']):
                reasons.append("No monthly uptrend (Close > EMA_20 > EMA_50).")
                logging.info(f"{symbol}: No buy signal - {', '.join(reasons)}")

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
                logging.warning(f"{symbol}: No buy signal - {', '.join(reasons)}")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            weekly_df['EMA_20'] = weekly_df['Close'].ewm(span=20, adjust=False).mean()
            weekly_df['EMA_50'] = weekly_df['Close'].ewm(span=50, adjust=False).mean()
            latest_weekly = weekly_df.iloc[-1]

            if not (latest_weekly['Close'] > latest_weekly['EMA_20'] > latest_weekly['EMA_50']):
                reasons.append("No weekly uptrend (Close > EMA_20 > EMA_50).")
                logging.info(f"{symbol}: No buy signal - {', '.join(reasons)}")

            if reasons:
                # Monthly/Weekly uptrend not confirmed
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Download Daily Data
            daily_df = yf.download(
                symbol,
                start=(end_date - relativedelta(years=1)).strftime('%Y-%m-%d'),  # 1 year for daily data
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d",
                progress=False
            )
            if isinstance(daily_df.columns, pd.MultiIndex):
                daily_df.columns = daily_df.columns.get_level_values(0)
            
            if daily_df.empty or len(daily_df) < 60:
                reasons.append("Insufficient daily data.")
                logging.warning(f"{symbol}: No buy signal - {', '.join(reasons)}")
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                continue

            # Anchored VWAP from last ~60-day pivot low
            pivot_lookback = min(len(daily_df), 60)
            pivot_low_idx = daily_df['Low'].iloc[-pivot_lookback:].idxmin()
            if pd.isna(pivot_low_idx):
                reasons.append("No pivot low found for anchored VWAP.")
                logging.warning(f"{symbol}: No buy signal - {', '.join(reasons)}")
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

            # RSI, MACD, Bollinger Bands, ATR
            daily_df['RSI_14'] = compute_rsi(daily_df['Close'], 14)
            fast_ema = daily_df['Close'].ewm(span=12, adjust=False).mean()
            slow_ema = daily_df['Close'].ewm(span=26, adjust=False).mean()
            daily_df['MACD'] = fast_ema - slow_ema
            daily_df['MACD_Signal'] = daily_df['MACD'].ewm(span=9, adjust=False).mean()
            daily_df['ATR_14'] = compute_atr(daily_df, 14)
            bb_mid = daily_df['Close'].rolling(window=20).mean()
            bb_std = daily_df['Close'].rolling(window=20).std()
            daily_df['BB_Upper'] = bb_mid + 2 * bb_std
            daily_df['BB_Lower'] = bb_mid - 2 * bb_std

            # Heikin Ashi Candles
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

            # Check if price is above a short-term EMA
            daily_df['EMA_10'] = daily_df['Close'].ewm(span=10, adjust=False).mean()
            if latest_row['Close'] < latest_row['EMA_10']:
                reasons.append("Price < EMA_10.")

            if reasons:
                print(f"{Colors.RED}{symbol}: No buy signal - {', '.join(reasons)}{Colors.RESET}")
                logging.info(f"{symbol}: No buy signal - {', '.join(reasons)}")
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

            logging.info(f"{symbol}: Buy signal generated.")

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
                        print(
                            f"{Colors.GREEN}Added {symbol} to holdings with stop_loss=${stop_loss:.2f}, "
                            f"profit_target=${profit_target:.2f}.{Colors.RESET}"
                        )
                        logging.info(f"{symbol}: Added to holdings.")
                except ValueError:
                    print(f"{Colors.RED}Invalid input. Skipping addition of this holding.{Colors.RESET}")
                    logging.warning(f"{symbol}: Invalid input during holding addition.")
        except Exception as exc:
            print(f"{Colors.RED}{symbol}: No buy signal - {exc}{Colors.RESET}")
            logging.error(f"{symbol}: No buy signal - {exc}")
            continue

###################################################
# MAIN EXECUTION AND FUNCTIONALITY
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
