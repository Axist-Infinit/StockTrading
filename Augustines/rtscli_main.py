# rtscli_main.py

import os
import warnings
warnings.filterwarnings('ignore')
import urwid
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import utility functions
from rtscli_utils import (
    manage_holdings,
    manage_watchlist,
    load_holdings,
    load_watchlist,
    view_trade_history
)

# Import strategy functions (now includes run_100bagger_complex_strategy)
from rtscli_strategies import (
    run_pullback_strategy,
    run_breakout_strategy,
    run_swing_strategy_console,
    run_vwap_swing_trade_strategy,
    run_anchored_vwap_strategy,
    run_100bagger_complex_strategy,  # Newly added
    run_all_strategies
)

# Pattern-based strategies
from rtscli_strategies_patterns import (
    run_inverse_head_and_shoulders_strategy,
    run_rectangle_pattern_strategy,
    run_ascending_channel_strategy
)

# Projected price strategies
from rtscli_strategies_projected import (
    run_projected_swing_strategy,
    run_projected_vwap_swing_trade_strategy
)

# Portfolio management
from rtscli_portfolio import portfolio_cli

# VIX Trading
from VixTrader import run_vix_trader

# Stock forecasting
import stock_forecast

# Backtesting logic
from rtscli_backtesting import (
    run_backtest_menu,
    analyze_all_backtests
)

# ANSI escape sequences for colors (optional, for better UI)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'


##############################################
# Display Menus
##############################################

def display_main_menu():
    print("\nMain Menu:")
    print("1. Holdings Management")
    print("2. Watchlist Management")
    print("3. View Portfolio")
    print("4. Trading Strategies")
    print("5. Run Stock Forecasting")
    print("6. Strategy Backtester")
    print("7. Run VIX Trading Strategy")
    print("8. Backtester History")
    print("9. View Trade History")
    print("0. Exit")

def display_trading_strategies_submenu():
    print("\nTrading Strategies:")
    print("1. Current Price Strategies")
    print("2. Projected Price Strategies")
    print("3. Run All Strategies")
    print("4. Run 100-Bagger Complex Strategy")  # Newly added
    print("0. Return to Main Menu")

def display_current_price_strategies():
    print("\nCurrent Price Strategies:")
    print("1. Run Swing Trading Strategy")
    print("2. Run Pullback Strategy")
    print("3. Run Breakout Strategy")
    print("4. Run VWAP Swing Trade Strategy")
    print("5. Run Pattern-Based Strategies")
    print("6. Run Anchored VWAP Strategy")
    print("0. Return to Trading Strategies Menu")

def display_pattern_based_strategies():
    print("\nPattern-Based Strategies:")
    print("1. Run Inverse Head and Shoulders Strategy")
    print("2. Run Rectangle Pattern Strategy")
    print("3. Run Ascending Channel Strategy")
    print("0. Return to Current Price Strategies Menu")

def display_projected_price_strategies():
    print("\nProjected Price Strategies:")
    print("1. Run Projected Swing Trading Strategy")
    print("2. Run Projected VWAP Swing Trade Strategy")
    print("0. Return to Trading Strategies Menu")

##############################################
# Strategy Backtester Menu
##############################################

def display_strategy_backtester_menu():
    """
    Handles the Strategy Backtester menu.
    Prompts the user to select a strategy, ticker, and timeframe for backtesting.
    """
    print("\nStrategy Backtester:")
    print("1. Swing Strategy")
    print("2. Pullback Strategy")
    print("3. Breakout Strategy")
    print("4. VWAP Swing Trade Strategy")
    print("5. Anchored VWAP Strategy")
    print("6. 100-Bagger Complex Strategy")  # Newly added
    print("7. Backtest All Strategies")
    print("0. Return to Main Menu")

    choice = input("Select a strategy to backtest: ").strip()
    if choice == '0':
        return  # Return to main menu

    ticker = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
    months_str = input("Enter the number of months to backtest (e.g., 3, 6, 12): ").strip()

    try:
        months = int(months_str)
        if months <= 0:
            raise ValueError
    except ValueError:
        print("Invalid number of months. Please enter a positive integer.")
        return

    if choice == '1':
        run_backtest_menu(strategy="Swing", ticker=ticker, months=months)
    elif choice == '2':
        run_backtest_menu(strategy="Pullback", ticker=ticker, months=months)
    elif choice == '3':
        run_backtest_menu(strategy="Breakout", ticker=ticker, months=months)
    elif choice == '4':
        run_backtest_menu(strategy="VWAP", ticker=ticker, months=months)
    elif choice == '5':
        run_backtest_menu(strategy="AnchoredVWAP", ticker=ticker, months=months)
    elif choice == '6':  # Newly added
        run_backtest_menu(strategy="100Bagger", ticker=ticker, months=months)
    elif choice == '7':
        run_backtest_menu(strategy="All", ticker=ticker, months=months)
    else:
        print("Invalid option. Returning to Strategy Backtester Menu.")

##############################################
# Main Loop
##############################################

def main():
    while True:
        display_main_menu()
        choice = input("Select an option: ").strip()

        if choice == '1':
            manage_holdings()
        elif choice == '2':
            manage_watchlist()
        elif choice == '3':
            portfolio_cli()

        elif choice == '4':
            # Trading Strategies Submenu
            while True:
                display_trading_strategies_submenu()
                ts_choice = input("Select an option: ").strip()

                if ts_choice == '0':
                    break  # Return to main menu

                elif ts_choice == '1':
                    # Current Price Strategies Submenu
                    while True:
                        display_current_price_strategies()
                        current_choice = input("Select a strategy to run: ").strip()

                        if current_choice == '0':
                            break  # Return to Trading Strategies Menu
                        elif current_choice == '1':
                            run_swing_strategy_console()
                        elif current_choice == '2':
                            run_pullback_strategy()
                        elif current_choice == '3':
                            run_breakout_strategy()
                        elif current_choice == '4':
                            run_vwap_swing_trade_strategy()
                        elif current_choice == '5':
                            # Pattern-Based Strategies Submenu
                            while True:
                                display_pattern_based_strategies()
                                pattern_choice = input("Select a pattern-based strategy to run: ").strip()

                                if pattern_choice == '0':
                                    break  # Return to Current Price Strategies Menu
                                elif pattern_choice == '1':
                                    run_inverse_head_and_shoulders_strategy()
                                elif pattern_choice == '2':
                                    run_rectangle_pattern_strategy()
                                elif pattern_choice == '3':
                                    run_ascending_channel_strategy()
                                else:
                                    print("Invalid option. Please try again.")
                        elif current_choice == '6':
                            run_anchored_vwap_strategy()
                        else:
                            print("Invalid option. Please try again.")

                elif ts_choice == '2':
                    # Projected Price Strategies Submenu
                    while True:
                        display_projected_price_strategies()
                        projected_choice = input("Select a strategy to run: ").strip()

                        if projected_choice == '0':
                            break  # Return to Trading Strategies Menu
                        elif projected_choice == '1':
                            run_projected_swing_strategy()
                        elif projected_choice == '2':
                            run_projected_vwap_swing_trade_strategy()
                        else:
                            print("Invalid option. Please try again.")

                elif ts_choice == '3':
                    # Run All Strategies
                    run_all_strategies()

                elif ts_choice == '4':  # Newly added
                    run_100bagger_complex_strategy()

                else:
                    print("Invalid option. Please try again.")

        elif choice == '5':
            # Run Stock Forecasting menu
            stock_forecast.main_menu()

        elif choice == '6':
            # Strategy Backtester
            display_strategy_backtester_menu()

        elif choice == '7':
            # VIX Trading Strategy
            run_vix_trader()

        elif choice == '8':
            # Backtester History => Show analytics from all backtests
            analyze_all_backtests()

        elif choice == '9':
            view_trade_history()

        elif choice == '0':
            print("Goodbye!")
            break

        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()
