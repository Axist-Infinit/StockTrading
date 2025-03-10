# rtscli_utils.py

import os
import yfinance as yf
from datetime import datetime
from rtscli_portfolio import portfolio_cli
from rtscli_data import (
    load_holdings,
    save_holdings,
    load_watchlist,
    save_watchlist,
    load_trade_history,
    save_trade_history
)

# ANSI escape sequences for colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'

# Holdings Management Functions
def manage_holdings():
    while True:
        print("\nHoldings Management:")
        print("1. View Holdings")
        print("2. Add Holdings")
        print("3. Edit Holdings")
        print("4. Delete Holdings")
        print("5. Back to Main Menu")
        choice = input("Select an option: ")
        if choice == '1':
            portfolio_cli()
        elif choice == '2':
            add_new_holdings()
        elif choice == '3':
            edit_holdings()
        elif choice == '4':
            delete_holdings()
        elif choice == '5':
            break
        else:
            print("Invalid selection. Please try again.")

def add_new_holdings():
    holdings = load_holdings()
    while True:
        stock_name = input("Enter stock symbol (or type 'done' to finish): ").upper()
        if stock_name.lower() == 'done':
            break
        try:
            price_paid = float(input("Price paid per share: "))
            number_of_shares = int(input("Number of shares: "))
            stop_loss = float(input("Stop loss: "))
            profit_target = input("Profit Target (optional, press Enter to skip): ")
            profit_target = float(profit_target) if profit_target else None
        except ValueError:
            print("Invalid input. Please enter numeric values for price, shares, stop loss, and profit target.")
            continue

        holding = {
            'symbol': stock_name,
            'price_paid': price_paid,
            'shares': number_of_shares,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'strategy': 'Manual'  # Assuming 'Manual' since strategy isn't specified here
        }
        holdings.append(holding)
        print(f"Added {stock_name} to holdings.")
    save_holdings(holdings)

def edit_holdings():
    holdings = load_holdings()
    if not holdings:
        print("\nNo holdings available to edit.")
        return
    for idx, holding in enumerate(holdings, start=1):
        print(f"{idx}. {holding['symbol']} - {holding['shares']} shares at ${holding['price_paid']:.2f}")
    try:
        idx = int(input("Enter the number of the holding to edit: ")) - 1
        if 0 <= idx < len(holdings):
            holding = holdings[idx]
            holding['price_paid'] = float(input(f"New Price Paid (current: {holding['price_paid']}): "))
            holding['shares'] = int(input(f"New Number of Shares (current: {holding['shares']}): "))
            holding['stop_loss'] = float(input(f"New Stop Loss (current: {holding['stop_loss']}): "))
            profit_target = input(f"New Profit Target (current: {holding.get('profit_target', 'N/A')}): ")
            holding['profit_target'] = float(profit_target) if profit_target else None
            save_holdings(holdings)
            print("Holding updated successfully.")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

def delete_holdings():
    holdings = load_holdings()
    if not holdings:
        print("\nNo holdings available to delete.")
        return
    for idx, holding in enumerate(holdings, start=1):
        print(f"{idx}. {holding['symbol']} - {holding['shares']} shares at ${holding['price_paid']:.2f}")
    try:
        idx = int(input("Enter the number of the holding to delete: ")) - 1
        if 0 <= idx < len(holdings):
            removed = holdings.pop(idx)
            save_holdings(holdings)
            print(f"Deleted holding: {removed['symbol']}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

# Watchlist Management Functions
def manage_watchlist():
    while True:
        print("\nSelect Watchlist:")
        print("1. Main Watchlist")
        print("2. Speculative Watchlist")
        print("3. Back to Main Menu")
        choice = input("Select an option: ")
        if choice == '1':
            watchlist_name = 'main'
            watchlist_management_menu(watchlist_name)
        elif choice == '2':
            watchlist_name = 'speculative'
            watchlist_management_menu(watchlist_name)
        elif choice == '3':
            break
        else:
            print("Invalid selection. Please try again.")

def watchlist_management_menu(watchlist_name):
    while True:
        print(f"\n{watchlist_name.capitalize()} Watchlist Management:")
        print("1. View Watchlist")
        print("2. Add Stocks")
        print("3. Delete Stocks")
        print("4. Clear Watchlist")
        print("5. Back to Previous Menu")
        choice = input("Select an option: ")
        if choice == '1':
            view_watchlist(watchlist_name)
        elif choice == '2':
            add_to_watchlist(watchlist_name)
        elif choice == '3':
            delete_from_watchlist(watchlist_name)
        elif choice == '4':
            clear_watchlist(watchlist_name)
        elif choice == '5':
            break
        else:
            print("Invalid selection. Please try again.")

def view_watchlist(watchlist_name):
    stocks = load_watchlist(watchlist_name)
    if stocks:
        print(f"\nCurrent {watchlist_name.capitalize()} Watchlist:")
        for stock in stocks:
            print(f"- {stock}")
    else:
        print(f"\nYour {watchlist_name} watchlist is empty.")

def add_to_watchlist(watchlist_name):
    stocks = load_watchlist(watchlist_name)
    while True:
        new_stock = input(f"Enter stock symbol to add to {watchlist_name} watchlist (or 'done' to finish): ").upper()
        if new_stock.lower() == 'done':
            break
        if new_stock and new_stock not in stocks:
            stocks.append(new_stock)
            print(f"Added {new_stock} to {watchlist_name} watchlist.")
        else:
            print("Stock is already in the watchlist or invalid input.")
    save_watchlist(stocks, watchlist_name)

def delete_from_watchlist(watchlist_name):
    stocks = load_watchlist(watchlist_name)
    if not stocks:
        print(f"\n{watchlist_name.capitalize()} Watchlist is empty.")
        return
    while True:
        del_stock = input(f"Enter stock symbol to delete from {watchlist_name} watchlist (or 'done' to finish): ").upper()
        if del_stock.lower() == 'done':
            break
        if del_stock in stocks:
            stocks.remove(del_stock)
            print(f"Removed {del_stock} from {watchlist_name} watchlist.")
        else:
            print("Stock not found in watchlist.")
    save_watchlist(stocks, watchlist_name)

def clear_watchlist(watchlist_name):
    confirm = input(f"Are you sure you want to clear the {watchlist_name} watchlist? (yes/no): ").lower()
    if confirm == 'yes':
        save_watchlist([], watchlist_name)
        print(f"{watchlist_name.capitalize()} watchlist cleared.")
    else:
        print("Clear action canceled.")

def select_watchlist():
    while True:
        print("\nSelect Watchlist:")
        print("1. Main Watchlist")
        print("2. Speculative Watchlist")
        choice = input("Select an option: ")
        if choice == '1':
            return 'main'
        elif choice == '2':
            return 'speculative'
        else:
            print("Invalid selection. Please try again.")

# Trade History Functions
def view_trade_history():
    trade_history = load_trade_history()
    if not trade_history:
        print("\nNo trade history available.")
        input("\nPress Enter to return to the main menu...")
        return

    # Define column widths and order
    columns = [
        ('symbol', 'Symbol', 8),
        ('entry_date', 'Entry Date', 12),
        ('exit_date', 'Exit Date', 12),
        ('entry_price', 'Entry Price', 12),
        ('exit_price', 'Exit Price', 12),
        ('shares', 'Shares', 8),
        ('strategy', 'Strategy', 12),
        ('result', 'Result', 18)
    ]

    header_line = ''.join(header.ljust(width) for key, header, width in columns)
    print(header_line)
    print('-' * len(header_line))

    for trade in trade_history:
        line_data = {
            'symbol': trade.get('symbol', 'N/A'),
            'entry_date': trade.get('entry_date', 'N/A'),
            'exit_date': trade.get('exit_date', 'N/A'),
            'entry_price': f"${trade.get('price_paid', 0):.2f}",
            'exit_price': f"${trade.get('exit_price', 0):.2f}",
            'shares': str(trade.get('shares', 0)),
            'strategy': trade.get('strategy', 'Manual'),
            'result': trade.get('result', 'N/A')
        }

        line = ''.join(line_data[key].ljust(width) for key, header, width in columns)
        print(line)

    input("\nPress Enter to return to the main menu...")

# Additional utility functions can be added here if needed

