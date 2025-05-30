#!/usr/bin/env python3
"""
Swing Trading Strategy Script with Feature Refinement and Backtesting

Usage examples:
  1) Normal signal generation (for current data):
     python my_trading_script.py AAPL MSFT

  2) Backtest a single ticker over a date range:
     python my_trading_script.py --backtest --start 2022-01-01 --end 2022-12-31 AAPL

This script can:
- Generate swing trade signals (1-5 day hold) for given stock tickers in "live" mode
- Perform a basic historical backtest over a given date range
- Log the signals/trades to a CSV file if desired
"""

import os
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# --- We removed the old indicator code and now import from our separate indicators file ---
from indicators import (
    compute_indicators,
    compute_anchored_vwap,
    to_daily
)

from watchlist_utils import load_watchlist, save_watchlist, manage_watchlist

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

# For color in outputs
from colorama import init, Fore, Style
init(autoreset=True)

from fredapi import Fred 
import configparser

def load_config():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.ini")
    config.read(config_path)
    return config['FRED'].get('api_key', None)


def fetch_data(ticker, start=None, end=None, intervals=None):
    """
    Fetch multiple intervals of data for the given ticker from Yahoo Finance.
    ...
    (Same as before)
    """
    if intervals is None:
        intervals = {
            '5m':  ('14d','5m'),
            '30m': ('60d','30m'),
            '1h':  ('120d','1h'),
            '90m': ('60d','90m'),
            '1d':  ('380d','1d')
        }

    df_5m = yf.download(ticker, period=intervals['5m'][0], interval=intervals['5m'][1],
                        auto_adjust=True, progress=False)
    df_30m = yf.download(ticker, period=intervals['30m'][0], interval=intervals['30m'][1],
                         auto_adjust=True, progress=False)
    df_1h = yf.download(ticker, period=intervals['1h'][0], interval=intervals['1h'][1],
                        auto_adjust=True, progress=False)
    df_90m = yf.download(ticker, period=intervals['90m'][0], interval=intervals['90m'][1],
                         auto_adjust=True, progress=False)

    if start and end:
        df_1d = yf.download(ticker, start=start, end=end, interval='1d', auto_adjust=True)
    else:
        df_1d = yf.download(ticker, period=intervals['1d'][0], interval=intervals['1d'][1],
                            auto_adjust=True)

    for label, df in zip(['5m','30m','1h','90m','1d'], [df_5m, df_30m, df_1h, df_90m, df_1d]):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        df.dropna(how='all', inplace=True)

    return df_5m, df_30m, df_1h, df_90m, df_1d


def prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, macro_df):
    """
    Prepare a combined feature DataFrame from 5m, 30m, 1h, and 1-day data,
    along with macro data. Returns a DataFrame with features + 'future_class'.
    Uses triple-barrier labeling for 'future_class'.
    """
    # We now call compute_indicators() and compute_anchored_vwap() from indicators.py
    if df_1d.empty or 'Close' not in df_1d.columns:
        return pd.DataFrame()

    ind_5m  = compute_indicators(df_5m.copy(),  timeframe='5m')
    ind_30m = compute_indicators(df_30m.copy(), timeframe='30m')
    ind_1h  = compute_indicators(df_1h.copy(),  timeframe='hourly')
    ind_90m = compute_indicators(df_90m.copy(), timeframe='90m')
    ind_1d  = compute_indicators(df_1d.copy(),  timeframe='daily')

    ind_1d['AnchoredVWAP'] = compute_anchored_vwap(ind_1d)

    daily_5m  = to_daily(ind_5m,  "5m")
    daily_30m = to_daily(ind_30m, "30m")
    daily_1h  = to_daily(ind_1h,  "1h")
    daily_90m = to_daily(ind_90m, "90m")

    ind_1d.index.name = 'Date'
    features_df = ind_1d
    features_df = features_df.join(daily_5m,  on='Date', rsuffix='_5m')
    features_df = features_df.join(daily_30m, on='Date', rsuffix='_30m')
    features_df = features_df.join(daily_1h,  on='Date', rsuffix='_1h')
    features_df = features_df.join(daily_90m, on='Date', rsuffix='_90m')

    if macro_df is not None and not macro_df.empty:
        features_df = features_df.join(macro_df, on='Date')

    if 'Close' not in features_df.columns:
        return pd.DataFrame()
    features_df.dropna(subset=['Close'], inplace=True)
    if features_df.empty:
        return features_df

    # If ATR_daily missing, label = 1 (neutral)
    if 'ATR_daily' not in features_df.columns:
        features_df['future_class'] = 1
        return features_df

    # Triple-barrier logic
    horizon = 10
    multiplier = 2.0
    features_df['ATR_daily'] = features_df['ATR_daily'].fillna(0.0)

    up_barrier = features_df['Close'] + multiplier * features_df['ATR_daily']
    down_barrier = features_df['Close'] - multiplier * features_df['ATR_daily']

    future_class = np.ones(len(features_df), dtype=int)
    closes = features_df['Close'].values

    for i in range(len(features_df) - horizon):
        upper_lvl = up_barrier.iloc[i]
        lower_lvl = down_barrier.iloc[i]
        window_prices = closes[i+1 : i+1+horizon]

        above_idx = np.where(window_prices >= upper_lvl)[0]
        below_idx = np.where(window_prices <= lower_lvl)[0]

        if len(above_idx) == 0 and len(below_idx) == 0:
            continue
        elif len(above_idx) > 0 and len(below_idx) > 0:
            if above_idx[0] < below_idx[0]:
                future_class[i] = 2
            else:
                future_class[i] = 0
        elif len(above_idx) > 0:
            future_class[i] = 2
        elif len(below_idx) > 0:
            future_class[i] = 0

    features_df['future_class'] = future_class
    features_df = features_df.iloc[:-horizon]
    return features_df


def refine_features(features_df, importance_cutoff=0.0001, corr_threshold=0.9):
    """
    ...
    (Same as your existing refine_features)
    """
    if features_df.empty or 'future_class' not in features_df.columns:
        return features_df

    y = features_df['future_class']
    X = features_df.drop(columns=['future_class']).copy()

    X.ffill(inplace=True)
    X.bfill(inplace=True)

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',
        device='cuda'
    )

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(np.unique(y_train)) < 2:
        return features_df

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feat_importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    drop_by_importance = feat_importance_series[feat_importance_series < importance_cutoff].index.tolist()
    if drop_by_importance:
        X.drop(columns=drop_by_importance, inplace=True, errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numeric_cols].corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_by_corr = [col for col in upper.columns if any(upper[col].abs() > corr_threshold)]
    if drop_by_corr:
        X.drop(columns=drop_by_corr, inplace=True, errors='ignore')

    refined_df = X.join(y)
    return refined_df


def tune_threshold_and_train(features_df):
    """
    ...
    (Same as your existing tune_threshold_and_train)
    """
    if features_df.empty or 'future_class' not in features_df.columns:
        return None, None

    feature_cols = [c for c in features_df.columns if c != 'future_class']
    X_full = features_df[feature_cols].copy()
    X_full.ffill(inplace=True)
    X_full.bfill(inplace=True)

    y_full = features_df['future_class']
    split_idx = int(len(features_df) * 0.8)
    X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    train_data = pd.concat([X_train, y_train], axis=1)
    class_0 = train_data[train_data['future_class'] == 0]
    class_1 = train_data[train_data['future_class'] == 1]
    class_2 = train_data[train_data['future_class'] == 2]

    max_count = max(len(class_0), len(class_1), len(class_2))
    class_0_up = resample(class_0, replace=True, n_samples=max_count, random_state=42)
    class_1_up = resample(class_1, replace=True, n_samples=max_count, random_state=42)
    class_2_up = resample(class_2, replace=True, n_samples=max_count, random_state=42)

    train_oversampled = pd.concat([class_0_up, class_1_up, class_2_up], axis=0)
    X_train = train_oversampled.drop(columns=['future_class'])
    y_train = train_oversampled['future_class']

    best_thr = None
    best_score = -np.inf

    for thr in [0.01, 0.02, 0.03, 0.04, 0.05]:
        if 'Close' not in features_df.columns:
            continue
        future_ret_train = (features_df['Close'][:split_idx].shift(-5)
                            / features_df['Close'][:split_idx] - 1.0)
        future_ret_test  = (features_df['Close'][split_idx:].shift(-5)
                            / features_df['Close'][split_idx:] - 1.0)

        y_train_temp = y_train.copy()
        y_test_temp  = y_test.copy()
        y_train_temp[:] = 1
        y_train_temp[future_ret_train > thr] = 2
        y_train_temp[future_ret_train < -thr] = 0
        y_test_temp[:] = 1
        y_test_temp[future_ret_test > thr] = 2
        y_test_temp[future_ret_test < -thr] = 0

        unique_classes = np.unique(y_train_temp)
        if len(unique_classes) < 3:
            continue

        model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            tree_method='hist',
            device='cuda',
            max_depth=5,
            min_child_weight=10,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            n_estimators=500
        )
        model.fit(X_train, y_train_temp)
        y_pred = model.predict(X_test)
        report = classification_report(y_test_temp, y_pred, output_dict=True, zero_division=0)
        f1_long = report.get('2', {}).get('f1-score', 0.0)
        f1_short= report.get('0', {}).get('f1-score', 0.0)
        avg_f1  = (f1_long + f1_short)/2.0

        if avg_f1 > best_score:
            best_score = avg_f1
            best_thr   = thr

    if best_thr is None:
        return None, None

    future_ret_all = (features_df['Close'].shift(-5) / features_df['Close'] - 1.0)
    final_y = y_full.copy()
    final_y[:] = 1
    final_y[future_ret_all > best_thr] = 2
    final_y[future_ret_all < -best_thr] = 0

    if len(np.unique(final_y)) < 3:
        return None, best_thr

    final_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',
        device='cuda'
    )
    final_model.fit(X_full, final_y)
    return final_model, best_thr


def generate_signal_output(ticker, latest_row, model, thr, macro_latest):
    """
    ...
    (Same as before, no changes)
    """
    latest_df = latest_row.to_frame().T
    probs = model.predict_proba(latest_df.values)[0]

    class_idx = np.argmax(probs)
    prob_score = probs[class_idx]
    if prob_score < 0.60:
        return f"{ticker}: No clear signal (confidence only {prob_score:.2f})."

    if class_idx == 2:
        direction = "LONG"
    elif class_idx == 0:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    prob_score = probs[class_idx]
    price = latest_row.get('Close', np.nan)
    atr = latest_row.get('ATR_daily', 0.0)

    if direction == "LONG":
        stop_loss = price - atr
        target    = price + 2*atr
    elif direction == "SHORT":
        stop_loss = price + atr
        target    = price - 2*atr
    else:
        return f"{ticker}: No clear signal (model favors staying out w/ {prob_score:.2f} confidence)."

    rationale_parts = []
    # [Your existing logic for building rationale_parts ...]
    # ...
    rationale = ", ".join(rationale_parts)

    return (
        f"{ticker}: {direction} at ${price:.2f}, Stop ${stop_loss:.2f}, Target ${target:.2f}, "
        f"Prob {prob_score:.2f} - Rationale: {rationale}"
    )


def get_macro_data(start, end, fred_api_key=None):
    """
    ...
    (Same as your existing get_macro_data)
    """
    macro_symbols = ['^VIX', '^IRX', 'TIP']
    macro_df = yf.download(macro_symbols, start=start, end=end,
                           interval='1d', auto_adjust=True)['Close']
    if isinstance(macro_df.columns, pd.MultiIndex):
        macro_df.columns = macro_df.columns.droplevel(1)
    macro_df = macro_df.rename(columns={
        '^VIX': 'VIX',
        '^IRX': 'IRX',
        'TIP': 'TIP'
    })
    macro_df.ffill(inplace=True)

    if fred_api_key:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)
        unrate = fred.get_series('UNRATE', observation_start=start, observation_end=end)
        unrate = unrate.resample('D').ffill()
        unrate.name = 'UNRATE'

        spread = fred.get_series('T10Y2Y', observation_start=start, observation_end=end)
        spread = spread.resample('D').ffill()
        spread.name = 'YIELD_SPREAD'

        macro_df = macro_df.join(unrate, how='outer')
        macro_df = macro_df.join(spread, how='outer')
        macro_df.ffill(inplace=True)
    else:
        print("WARNING: No FRED API key provided; skipping UNRATE/T10Y2Y.")

    return macro_df


# ======== Color Helper Functions ========
def colorize_pnl(pnl):
    """Return a color-coded string for the PnL percentage."""
    if pnl >= 0:
        return f"{Fore.GREEN}{pnl*100:.2f}%{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{pnl*100:.2f}%{Style.RESET_ALL}"

def colorize_direction(direction):
    """Color code the direction."""
    if direction == "LONG":
        return f"{Fore.GREEN}{direction}{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{direction}{Style.RESET_ALL}"


# ---------------------------- Backtesting Logic ----------------------------
def backtest_strategy(ticker, start_date, end_date, macro_df, log_file=None):
    """
    Example backtest that simulates checking every 30 minutes, avoiding new trades
    if one is open, and logs exact timestamps for trade entries. Colorful output included.
    """
    df_5m, df_30m, df_1h, df_90m, df_1d = fetch_data(ticker, start=start_date, end=end_date)

    features_30m = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, macro_df)
    if features_30m.empty:
        print(f"No data to backtest for {ticker} in range {start_date} to {end_date}.")
        return

    features_30m = refine_features(features_30m)
    if features_30m.empty:
        print(f"No features left after refinement for {ticker}.")
        return

    model, best_thr = tune_threshold_and_train(features_30m)
    if model is None:
        print(f"Model training failed for {ticker} in backtest.")
        return

    all_feature_cols = [c for c in features_30m.columns if c != 'future_class']
    X_all = features_30m[all_feature_cols].copy()
    X_all.ffill(inplace=True)
    X_all.bfill(inplace=True)

    df_preds = features_30m.copy()
    df_preds['prediction'] = model.predict(X_all)
    probas = model.predict_proba(X_all)

    trades = []
    in_trade = False
    trade_direction = None
    entry_price = None
    entry_time = None
    stop_price = None
    target_price = None

    for i in range(len(df_preds)):
        row_time = df_preds.index[i]
        row_close = df_preds.iloc[i]['Close']
        pred_class = df_preds.iloc[i]['prediction']
        row_probas = probas[i]

        if in_trade:
            # Check exit conditions ...
            if trade_direction == "LONG":
                if row_close >= target_price:
                    pnl_pct = (row_close - entry_price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time,
                        'exit_timestamp': row_time,
                        'direction': "LONG",
                        'entry_price': round(entry_price,2),
                        'exit_price': round(row_close,2),
                        'pnl_pct': round(pnl_pct,4),
                        'stop_price': round(stop_price,2),
                        'target_price': round(target_price,2)
                    })
                    in_trade = False
                elif row_close <= stop_price:
                    pnl_pct = (row_close - entry_price) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time,
                        'exit_timestamp': row_time,
                        'direction': "LONG",
                        'entry_price': round(entry_price,2),
                        'exit_price': round(row_close,2),
                        'pnl_pct': round(pnl_pct,4),
                        'stop_price': round(stop_price,2),
                        'target_price': round(target_price,2)
                    })
                    in_trade = False

            elif trade_direction == "SHORT":
                if row_close <= target_price:
                    pnl_pct = (entry_price - row_close) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time,
                        'exit_timestamp': row_time,
                        'direction': "SHORT",
                        'entry_price': round(entry_price,2),
                        'exit_price': round(row_close,2),
                        'pnl_pct': round(pnl_pct,4),
                        'stop_price': round(stop_price,2),
                        'target_price': round(target_price,2)
                    })
                    in_trade = False
                elif row_close >= stop_price:
                    pnl_pct = (entry_price - row_close) / entry_price
                    trades.append({
                        'entry_timestamp': entry_time,
                        'exit_timestamp': row_time,
                        'direction': "SHORT",
                        'entry_price': round(entry_price,2),
                        'exit_price': round(row_close,2),
                        'pnl_pct': round(pnl_pct,4),
                        'stop_price': round(stop_price,2),
                        'target_price': round(target_price,2)
                    })
                    in_trade = False

            if in_trade:
                continue

        # If no open trade, check new signal
        if not in_trade:
            class_idx = int(pred_class)
            prob_score = row_probas[class_idx]
            if prob_score < 0.60:
                continue

            if class_idx == 2:  # LONG
                in_trade = True
                trade_direction = "LONG"
                entry_price = row_close
                entry_time = row_time
                atr = df_preds.iloc[i].get('ATR_daily', 0.0)
                stop_price = entry_price - atr
                target_price = entry_price + 2 * atr
            elif class_idx == 0:  # SHORT
                in_trade = True
                trade_direction = "SHORT"
                entry_price = row_close
                entry_time = row_time
                atr = df_preds.iloc[i].get('ATR_daily', 0.0)
                stop_price = entry_price + atr
                target_price = entry_price - 2 * atr
            else:
                continue

    if not trades:
        print(f"No trades triggered for {ticker} in backtest.")
        return

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    avg_pnl = trades_df['pnl_pct'].mean()
    win_rate = (trades_df['pnl_pct'] > 0).mean()

    # Construct color-coded summary
    avg_pnl_str = colorize_pnl(avg_pnl)
    win_rate_str = f"{Fore.BLUE}{round(win_rate*100,2)}%{Style.RESET_ALL}"

    print(f"{Fore.CYAN}\nBacktest Results for {ticker} ({start_date} to {end_date}):{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Total trades:{Style.RESET_ALL} {total_trades}")
    print(f"  {Fore.YELLOW}Win rate:{Style.RESET_ALL} {win_rate_str}")
    print(f"  {Fore.YELLOW}Avg P/L per trade:{Style.RESET_ALL} {avg_pnl_str}")

    print(f"\n{Fore.MAGENTA}Sample trades:{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}     EntryTime           ExitTime           Direction   EntryPrice   ExitPrice    PnL%{Style.RESET_ALL}")
    for i, row in trades_df.head(5).iterrows():
        dir_colored = colorize_direction(row['direction'])
        pnl_colored = colorize_pnl(row['pnl_pct'])
        print(f"  {row['entry_timestamp']} -> {row['exit_timestamp']}  {dir_colored:8s}"
              f"   {row['entry_price']:10.2f}  {row['exit_price']:10.2f}  {pnl_colored}")

    if log_file is not None:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if log_file.tell() == 0:
            log_file.write("LogTimestamp,Ticker,EntryTime,ExitTime,Direction,EntryPrice,ExitPrice,PnLPct,StopPrice,TargetPrice\n")
        for _, row in trades_df.iterrows():
            log_file.write(
                f"{now_str},{ticker},{row['entry_timestamp']},{row['exit_timestamp']},"
                f"{row['direction']},{row['entry_price']},{row['exit_price']},{row['pnl_pct']},"
                f"{row['stop_price']},{row['target_price']}\n"
            )
        log_file.flush()


def display_main_menu():
    print("\nMain Menu:")
    print("1. Manage Watchlist")
    print("2. Run Signals on Watchlist (Live Mode)")
    print("3. Backtest All Watchlist Tickers")
    print("4. Run Single Ticker (Live or Backtest)")
    print("0. Exit")

def run_signals_on_watchlist():
    """
    ...
    (Same as existing code for live signals)
    """
    fred_api_key = load_config()
    tickers = load_watchlist()
    if not tickers:
        print("Watchlist empty. Please add tickers first.")
        return
    
    today = datetime.datetime.today()
    macro_start = today - datetime.timedelta(days=380)
    macro_df = get_macro_data(macro_start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'),
                              today.strftime('%Y-%m-%d'), fred_api_key=fred_api_key)

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live signal) ===")
        ...
        # (unchanged)
        

def backtest_watchlist():
    """
    ...
    (Calls backtest_strategy for each watchlist ticker)
    """
    tickers = load_watchlist()
    if not tickers:
        print("Watchlist empty. Please add tickers first.")
        return

    start_arg = input("Enter start date (YYYY-MM-DD): ").strip()
    end_arg   = input("Enter end date (YYYY-MM-DD): ").strip()
    if not start_arg or not end_arg:
        print("Invalid date range.")
        return
    
    macro_df = get_macro_data(start_arg, end_arg)

    for ticker in tickers:
        print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
        backtest_strategy(ticker, start_arg, end_arg, macro_df, log_file=None)


def interactive_menu():
    """
    ...
    (Your existing interactive logic)
    """
    while True:
        display_main_menu()
        choice = input("Select an option: ").strip()
        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            manage_watchlist()
        elif choice == '2':
            run_signals_on_watchlist()
        elif choice == '3':
            backtest_watchlist()
        elif choice == '4':
            ticker = input("Enter the ticker symbol: ").strip().upper()
            mode = input("Enter 'live' or 'backtest': ").strip().lower()
            if mode == "live":
                ...
            elif mode == "backtest":
                ...
            else:
                print("Invalid mode.")
        else:
            print("Invalid option. Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Swing trading signal generator/backtester")
    parser.add_argument('tickers', nargs='*', help="List of stock ticker symbols to analyze")
    parser.add_argument('--log', action='store_true', help="Optionally log selected trades to a file")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode instead of live signal mode")
    parser.add_argument('--start', default=None, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date for backtest (YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.tickers and not any(vars(args).values()):
        interactive_menu()
        return

    fred_api_key = load_config()
    tickers = args.tickers
    log_trades = args.log
    run_backtest = args.backtest
    start_arg = args.start
    end_arg = args.end

    log_file = None
    if log_trades:
        log_file = open("trades_log.csv", "a")

    if run_backtest and start_arg and end_arg:
        macro_df = get_macro_data(start_arg, end_arg, fred_api_key=fred_api_key)
        for ticker in tickers:
            log_filename = f"{ticker}_{start_arg}_{end_arg}.csv"
            with open(log_filename, "a") as log_file:
                print(f"\n=== Backtesting {ticker} from {start_arg} to {end_arg} ===")
                backtest_strategy(ticker, start_arg, end_arg, macro_df, log_file=log_file)
                return
    else:
        today = datetime.datetime.today()
        macro_start = today - datetime.timedelta(days=380)
        macro_df = get_macro_data(macro_start.strftime('%Y-%m-%d'),
                                  today.strftime('%Y-%m-%d'),
                                  fred_api_key=fred_api_key)
        if log_file:
            log_file.close()

    today = datetime.datetime.today()
    macro_start = today - datetime.timedelta(days=380)
    macro_df = get_macro_data(macro_start.strftime('%Y-%m-%d'),
                              today.strftime('%Y-%m-%d'),
                              fred_api_key=fred_api_key)

    for ticker in tickers:
        print(f"\n=== Processing {ticker} (live signal mode) ===")
        try:
            df_5m, df_30m, df_1h, df_90m, df_1d = fetch_data(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        if df_1d.empty or 'Close' not in df_1d.columns:
            print(f"No usable daily data for {ticker}, skipping.")
            continue

        features_df = prepare_features(df_5m, df_30m, df_1h, df_90m, df_1d, macro_df)
        if features_df.empty:
            print(f"Insufficient data for {ticker}, skipping.")
            continue

        features_df = refine_features(features_df)
        if features_df.empty or 'future_class' not in features_df.columns:
            continue

        model, best_thr = tune_threshold_and_train(features_df)
        if model is None:
            print(f"Model training failed or no valid data for {ticker}.")
            continue

        latest_features = features_df.iloc[-1]
        X_latest = latest_features.drop(labels='future_class', errors='ignore').to_frame().T
        X_latest.ffill(inplace=True)
        X_latest.bfill(inplace=True)
        latest_series = X_latest.iloc[0]

        latest_date = features_df.index[-1]
        if latest_date in macro_df.index:
            macro_latest = macro_df.loc[latest_date].to_dict()
        else:
            macro_latest = macro_df.iloc[-1].to_dict()

        signal_output = generate_signal_output(ticker, latest_series, model, best_thr, macro_latest)
        print(signal_output)

        if log_trades and ("LONG" in signal_output or "SHORT" in signal_output):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                direction_part, details_part = signal_output.split(" at $", 1)
                _, direction = direction_part.split(": ")
                entry_str, rest = details_part.split(", Stop ")
                stop_str, rest2 = rest.split(", Target ")
                target_str, rest3 = rest2.split(", Prob ")
                prob_str, rationale_part = rest3.split(" - Rationale: ")

                entry   = entry_str.replace(",", "")
                stop    = stop_str.replace("$", "")
                target  = target_str.replace("$", "")
                prob    = prob_str
                rationale = rationale_part
            except Exception:
                rationale = signal_output
                direction= "N/A"
                entry=stop=target=prob=""

            log_file.write(
                f"{now_str},{ticker},{direction},{entry},{stop},{target},{prob},{rationale}\n"
            )
            log_file.flush()

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
