import urwid
import yfinance as yf
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# File paths
WATCHLIST_FILE = 'watchlist.json'
HOLDINGS_FILE_JSON = 'holdings.json'
HOLDINGS_FILE_TXT = 'tickers.txt'

# Helper functions for file operations
def load_json_file(file_path, default_data):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError:
            print(f"Error decoding {file_path}. Using default data.")
            return default_data
    else:
        return default_data

def save_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Holdings Management
def load_holdings():
    if os.path.exists(HOLDINGS_FILE_JSON):
        data = load_json_file(HOLDINGS_FILE_JSON, {'holdings': []})
        return data.get('holdings', [])
    elif os.path.exists(HOLDINGS_FILE_TXT):
        # Migrate holdings from tickers.txt to holdings.json
        holdings = []
        with open(HOLDINGS_FILE_TXT, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    symbol, price_paid, stop_loss, shares = parts
                    holding = {
                        'symbol': symbol,
                        'price_paid': float(price_paid),
                        'stop_loss': float(stop_loss),
                        'shares': int(shares),
                        'profit_target': None  # Assuming no profit target in old data
                    }
                    holdings.append(holding)
        save_holdings(holdings)
        os.remove(HOLDINGS_FILE_TXT)  # Optionally delete the old file
        return holdings
    else:
        return []

def save_holdings(holdings):
    save_json_file(HOLDINGS_FILE_JSON, {'holdings': holdings})

# Watchlist Management
def load_watchlist():
    data = load_json_file(WATCHLIST_FILE, {'stocks': []})
    return data.get('stocks', [])

def save_watchlist(stocks):
    save_json_file(WATCHLIST_FILE, {'stocks': stocks})

def view_watchlist():
    stocks = load_watchlist()
    if stocks:
        print("\nYour current watchlist:")
        for stock in stocks:
            print(f"- {stock}")
    else:
        print("\nYour watchlist is empty.")

def add_to_watchlist():
    stocks = load_watchlist()
    while True:
        new_stock = input("Enter the stock symbol to add (or 'done' to finish): ").upper()
        if new_stock.lower() == 'done':
            break
        elif not new_stock:
            print("Stock symbol cannot be empty.")
        elif new_stock not in stocks:
            stocks.append(new_stock)
            print(f"Added {new_stock} to your watchlist.")
        else:
            print("Stock already in watchlist.")
    save_watchlist(stocks)

def delete_from_watchlist():
    stocks = load_watchlist()
    if not stocks:
        print("\nYour watchlist is empty.")
        return
    while True:
        del_stock = input("Enter the stock symbol to delete (or 'done' to finish): ").upper()
        if del_stock.lower() == 'done':
            break
        elif del_stock in stocks:
            stocks.remove(del_stock)
            print(f"Removed {del_stock} from your watchlist.")
        else:
            print("Stock not found in your watchlist.")
    save_watchlist(stocks)

def clear_watchlist():
    confirm = input("Are you sure you want to clear your watchlist? (yes/no): ").lower()
    if confirm == 'yes':
        save_watchlist([])
        print("Your watchlist has been cleared.")
    else:
        print("Clear watchlist canceled.")

def manage_watchlist():
    while True:
        print("\nWatchlist Management:")
        print("1. View Watchlist")
        print("2. Add Stocks")
        print("3. Delete Stocks")
        print("4. Clear Watchlist")
        print("5. Back to Main Menu")
        choice = input("Select an option: ")
        if choice == '1':
            view_watchlist()
        elif choice == '2':
            add_to_watchlist()
        elif choice == '3':
            delete_from_watchlist()
        elif choice == '4':
            clear_watchlist()
        elif choice == '5':
            break
        else:
            print("Invalid selection. Please try again.")

# Holdings Management Functions

def add_new_holdings():
    holdings = load_holdings()
    while True:
        add_more = input("Add new holdings? (yes/no): ").lower()
        if add_more not in ['yes', 'no']:
            print("Please answer with 'yes' or 'no'.")
            continue
        if add_more == 'no':
            break

        # Gather stock details from the user
        stock_name = input("Name of stock: ").upper()
        try:
            price_paid = float(input("Price paid: "))
            number_of_shares = int(input("Number of shares: "))
            stop_loss = float(input("Stop loss: "))
            profit_target_input = input("Profit Target (optional): ")
            if profit_target_input.strip() == '':
                profit_target = None
            else:
                profit_target = float(profit_target_input)
        except ValueError:
            print("Invalid input. Please enter numeric values for price, shares, stop loss, and profit target.")
            continue

        # Append new holding
        holding = {
            'symbol': stock_name,
            'price_paid': price_paid,
            'shares': number_of_shares,
            'stop_loss': stop_loss,
            'profit_target': profit_target
        }
        holdings.append(holding)
        print(f"Added {stock_name} to holdings.")
    save_holdings(holdings)

def edit_holdings():
    holdings = load_holdings()
    if not holdings:
        print("\nYou have no holdings to edit.")
        return
    view_holdings_simple()
    try:
        idx = int(input("Enter the number of the holding to edit: ")) - 1
        if idx < 0 or idx >= len(holdings):
            print("Invalid selection.")
            return
        holding = holdings[idx]
        print(f"Editing {holding['symbol']}:")
        try:
            holding['price_paid'] = float(input(f"New Price Paid (current: {holding['price_paid']}): "))
            holding['shares'] = int(input(f"New Number of Shares (current: {holding['shares']}): "))
            holding['stop_loss'] = float(input(f"New Stop Loss (current: {holding['stop_loss']}): "))
            profit_target_input = input(f"New Profit Target (current: {holding.get('profit_target', 'N/A')}): ")
            if profit_target_input.strip() == '':
                holding['profit_target'] = None
            else:
                holding['profit_target'] = float(profit_target_input)
            holdings[idx] = holding
            save_holdings(holdings)
            print("Holding updated successfully.")
        except ValueError:
            print("Invalid input. Edit canceled.")
    except ValueError:
        print("Invalid input.")

def delete_holdings():
    holdings = load_holdings()
    if not holdings:
        print("\nYou have no holdings to delete.")
        return
    view_holdings_simple()
    try:
        idx = int(input("Enter the number of the holding to delete: ")) - 1
        if idx < 0 or idx >= len(holdings):
            print("Invalid selection.")
            return
        holding = holdings.pop(idx)
        save_holdings(holdings)
        print(f"Deleted holding: {holding['symbol']}")
    except ValueError:
        print("Invalid input.")

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
            portfolio_cli()  # Changed to show detailed view
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

def view_holdings_simple():
    holdings = load_holdings()
    if holdings:
        print("\nYour current holdings:")
        for idx, holding in enumerate(holdings, 1):
            print(f"{idx}. {holding['symbol']} - Shares: {holding['shares']}, Price Paid: {holding['price_paid']}")
    else:
        print("\nYou have no holdings.")

# Trading Strategy
def run_infinit():
    # Updated to read from watchlist.json
    stock_list = load_watchlist()

    if not stock_list:
        print("Your watchlist is empty. Please add stocks to your watchlist first.")
        return

    # EMA periods
    ema_periods = [7, 21, 50]

    # List to store stocks with buy signals
    buy_signals = []

    for symbol in stock_list:
        print(f"\nProcessing {symbol}...")
        try:
            df = yf.download(symbol, period="6mo", progress=False)

            if df.empty:
                print(f"No data found for {symbol}.\n")
                continue

            # Calculate EMAs
            for period in ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

            # Calculate RSI
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            window_length = 14
            roll_up = up.rolling(window=window_length).mean()
            roll_down = down.rolling(window=window_length).mean()
            rs = roll_up / roll_down
            df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

            # Calculate ATR
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Calculate Stochastic Oscillator
            df['L14'] = df['Low'].rolling(window=14).min()
            df['H14'] = df['High'].rolling(window=14).max()
            df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
            df['%D'] = df['%K'].rolling(window=3).mean()

            # Calculate Average Volume
            df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

            # Latest data
            latest = df.iloc[-1]
            previous = df.iloc[-2]

            # Current price
            current_price = latest['Close']

            # Strategy conditions
            uptrend = latest['EMA_7'] > latest['EMA_21'] > latest['EMA_50']
            price_above_ema7 = current_price > latest['EMA_7']
            rsi_ok = latest['RSI'] < 70
            volume_ok = latest['Volume'] > latest['Avg_Volume']
            macd_bullish = latest['MACD'] > latest['MACD_Signal']
            stochastic_ok = latest['%K'] < 80

            # Check for upcoming earnings (next 14 days)
            stock = yf.Ticker(symbol)
            earnings_date = None
            try:
                cal = stock.calendar
                if not cal.empty:
                    earnings_date = cal.loc['Earnings Date'][0]
                    if pd.isnull(earnings_date):
                        earnings_date = None
            except Exception:
                pass

            earnings_ok = True
            if earnings_date:
                days_until_earnings = (earnings_date - datetime.now()).days
                earnings_ok = days_until_earnings > 14

            # Multiple time frame analysis (Weekly data)
            df_weekly = df.resample('W').last()
            df_weekly['EMA_7'] = df_weekly['Close'].ewm(span=7, adjust=False).mean()
            df_weekly['EMA_21'] = df_weekly['Close'].ewm(span=21, adjust=False).mean()
            higher_timeframe_trend = df_weekly['EMA_7'].iloc[-1] > df_weekly['EMA_21'].iloc[-1]

            # Final buy signal
            buy_signal = (
                uptrend and price_above_ema7 and rsi_ok and volume_ok and
                macd_bullish and stochastic_ok and earnings_ok and higher_timeframe_trend
            )

            # Prepare output
            print(f"Current Price: ${current_price:.2f}")

            if buy_signal:
                # Calculate stop loss and profit target using ATR
                atr = latest['ATR']
                stop_loss = current_price - atr
                profit_target = current_price + (2 * atr)
                expected_gain = ((profit_target - current_price) / current_price) * 100

                print("Buy Recommendation:")
                print(f"  Buy at: ${current_price:.2f}")
                print(f"  Stop Loss: ${stop_loss:.2f} (1 ATR below entry)")
                print(f"  Profit Target: ${profit_target:.2f} (2:1 Reward-to-Risk)")
                print(f"  Expected Gain: {expected_gain:.2f}%")

                # Store the buy signal details
                buy_signals.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'profit_target': profit_target
                })

            else:
                print("No Buy Signal at this time.")

        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")

        print("-" * 40)

    # Prompt user to add stocks with buy signals to holdings
    if buy_signals:
        for signal in buy_signals:
            add_holding = input(f"Do you want to add {signal['symbol']} to your holdings? (yes/no): ").lower()
            if add_holding == 'yes':
                holdings = load_holdings()
                try:
                    price_paid = float(input(f"Enter price paid for {signal['symbol']} (default is current price ${signal['current_price']:.2f}): ") or signal['current_price'])
                    number_of_shares = int(input("Enter number of shares: "))
                    # Use the calculated stop loss and profit target
                    stop_loss = signal['stop_loss']
                    profit_target = signal['profit_target']
                except ValueError:
                    print("Invalid input. Skipping addition of this holding.")
                    continue

                holding = {
                    'symbol': signal['symbol'],
                    'price_paid': price_paid,
                    'shares': number_of_shares,
                    'stop_loss': stop_loss,
                    'profit_target': profit_target
                }
                holdings.append(holding)
                save_holdings(holdings)
                print(f"Added {signal['symbol']} to holdings with stop loss at ${stop_loss:.2f} and profit target at ${profit_target:.2f}.")

    input("\nAnalysis complete. Press Enter to return to the main menu...")

# Portfolio Viewer with corrected line construction
def portfolio_cli():
    holdings = load_holdings()
    if not holdings:
        print("You have no holdings. Please add holdings first.")
        return

    # Set up color scheme
    palette = [
        ('titlebar', 'white,bold', ''),
        ('refresh button', 'dark green,bold', ''),
        ('quit button', 'dark red', ''),
        ('headers', 'light blue,bold', ''),
        ('positive', 'dark green', ''),
        ('negative', 'dark red', ''),
        ('body', 'white', ''),
        ('footer', 'white,bold', ''),
    ]

    header_text = urwid.Text(u' Portfolio Manager', align='center')
    header = urwid.AttrMap(header_text, 'titlebar')

    # Create the menu
    menu = urwid.Text([
        u'Press (', ('refresh button', u'R'), u') to refresh. ',
        u'Press (', ('quit button', u'Q'), u') to return to main menu.'
    ])
    footer = urwid.AttrMap(menu, 'footer')

    # Create the initial body text
    body_text = urwid.Text(u'Fetching portfolio data...', align='left')
    body_filler = urwid.Filler(body_text, valign='top')
    body_padding = urwid.Padding(body_filler, left=1, right=1)
    body = urwid.AttrMap(body_padding, 'body')

    # Assemble the widgets
    layout = urwid.Frame(header=header, body=body, footer=footer)

    def get_color(value):
        return 'positive' if value >= 0 else 'negative'

    def get_portfolio_data():
        updates = []

        # Define column widths
        col_widths = {
            'symbol': 8,
            'last_price': 12,
            'price_paid': 12,
            'shares': 8,
            'gain_percent': 10,
            'gain': 12,
            'stop_loss': 12,
            'profit_target': 12
        }

        # Create headers
        headers = {
            'symbol': 'Symbol',
            'last_price': 'Last Price',
            'price_paid': 'Price Paid',
            'shares': 'Shares',
            'gain_percent': '% Gain',
            'gain': 'Gain/Loss',
            'stop_loss': 'Stop Loss',
            'profit_target': 'Profit Take'
        }

        header_line = ''.join(headers[col].ljust(col_widths[col]) for col in headers)
        updates.append(('headers', header_line + '\n'))

        total_portfolio_gain = 0.0

        for holding in holdings:
            symbol = holding['symbol']
            price_paid = holding['price_paid']
            shares = holding['shares']
            stop_loss = holding['stop_loss']
            profit_target = holding.get('profit_target')

            try:
                stock = yf.Ticker(symbol)
                stock_data = stock.history(period="1d")

                if stock_data.empty:
                    continue

                last_price = round(stock_data['Close'].iloc[-1], 2)
                gain_per_share = last_price - price_paid
                gain = gain_per_share * shares
                gain_percent = (gain_per_share / price_paid) * 100

                total_portfolio_gain += gain

                profit_target_str = f'${profit_target:.2f}' if profit_target else 'N/A'

                line_data = {
                    'symbol': symbol,
                    'last_price': f'${last_price:.2f}',
                    'price_paid': f'${price_paid:.2f}',
                    'shares': str(shares),
                    'gain_percent': f'{gain_percent:.2f}%',
                    'gain': f'${gain:.2f}',
                    'stop_loss': f'${stop_loss:.2f}',
                    'profit_target': profit_target_str
                }

                # Build the line with specified colors
                line = []
                line.append(('white', line_data['symbol'].ljust(col_widths['symbol'])))
                last_price_color = get_color(last_price - price_paid)
                line.append((last_price_color, line_data['last_price'].ljust(col_widths['last_price'])))
                line.append(('white', line_data['price_paid'].ljust(col_widths['price_paid'])))
                line.append(('white', line_data['shares'].ljust(col_widths['shares'])))
                gain_percent_color = get_color(gain_percent)
                line.append((gain_percent_color, line_data['gain_percent'].ljust(col_widths['gain_percent'])))
                gain_color = get_color(gain)
                line.append((gain_color, line_data['gain'].ljust(col_widths['gain'])))
                line.append(('white', line_data['stop_loss'].ljust(col_widths['stop_loss'])))
                line.append(('white', line_data['profit_target'].ljust(col_widths['profit_target'])))
                line.append(('', '\n'))

                updates.extend(line)

            except Exception as e:
                continue

        # Append total portfolio gain and set color only for the value
        updates.append(('', '\n'))
        total_gain_text = f'Total Portfolio Gain/Loss: '
        total_gain_value = f'${total_portfolio_gain:.2f}'
        total_gain_color = 'positive' if total_portfolio_gain >= 0 else 'negative'
        updates.append(('white', total_gain_text))  # Text part remains white
        updates.append((total_gain_color, total_gain_value))  # Value part changes color

        return updates

    # Handle key presses
    def handle_input(key):
        if key in ('R', 'r'):
            refresh()
        if key in ('Q', 'q'):
            raise urwid.ExitMainLoop()

    def refresh():
        body_text.set_text(get_portfolio_data())

    refresh()
    main_loop = urwid.MainLoop(layout, palette, unhandled_input=handle_input)
    main_loop.run()

# Main Menu with updated heading colors
def main():
    while True:
        print("\n\033[94mMain Menu:\033[0m")  # Light blue color
        print("1. Holdings Management")
        print("2. Watchlist Management")
        print("3. Run Swing Trading Strategy")
        print("4. Exit")
        choice = input("Select an option: ")
        if choice == '1':
            manage_holdings()
        elif choice == '2':
            manage_watchlist()
        elif choice == '3':
            run_infinit()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()
