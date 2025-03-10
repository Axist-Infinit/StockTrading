import urwid
import yfinance as yf
import json
import os

# File paths
HOLDINGS_FILE_JSON = 'holdings.json'
HOLDINGS_FILE_TXT = 'tickers.txt'

def load_holdings():
    if os.path.exists(HOLDINGS_FILE_JSON):
        with open(HOLDINGS_FILE_JSON, 'r') as f:
            data = json.load(f)
            return data.get('holdings', [])
    elif os.path.exists(HOLDINGS_FILE_TXT):
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
        return holdings
    else:
        return []

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

    # Save holdings to JSON file
    with open(HOLDINGS_FILE_JSON, 'w') as f:
        json.dump({'holdings': holdings}, f, indent=4)

def parse_holdings(holdings):
    for holding in holdings:
        yield holding

# Read holdings
holdings = load_holdings()

# Set up color scheme
palette = [
    ('titlebar', 'dark red', ''),
    ('refresh button', 'dark green,bold', ''),
    ('quit button', 'dark red', ''),
    ('getting quote', 'dark blue', ''),
    ('headers', 'white,bold', ''),
    ('change positive', 'dark green', ''),
    ('change negative', 'dark red', '')]

header_text = urwid.Text(u' Stock Quotes')
header = urwid.AttrMap(header_text, 'titlebar')

# Create the menu
menu = urwid.Text([
    u'Press (', ('refresh button', u'R'), u') to manually refresh. ',
    u'Press (', ('quit button', u'Q'), u') to quit.'
])

# Create the quotes box
quote_text = urwid.Text(u'Press (R) to get your first quote!')
quote_filler = urwid.Filler(quote_text, valign='top', top=1, bottom=1)
v_padding = urwid.Padding(quote_filler, left=1, right=1)
quote_box = urwid.LineBox(v_padding)

# Assemble the widgets
layout = urwid.Frame(header=header, body=quote_box, footer=menu)

def get_color(change):
    return 'change positive' if change >= 0 else 'change negative'

def calculate_gain(price_in, current_price, shares):
    gain_per_share = current_price - price_in
    gain_percent = (gain_per_share / price_in) * 100
    return gain_per_share * shares, gain_percent

def get_update():
    # Define column widths
    col_widths = {
        'ticker': 10,
        'last_price': 15,
        'price_paid': 15,
        'shares': 10,
        'gain_percent': 10,
        'gain': 15,
        'stop_loss': 15,
        'profit_target': 15
    }

    # Create headers with appropriate spacing
    headers = {
        'ticker': 'Stock',
        'last_price': 'Last Price',
        'price_paid': 'Price Paid',
        'shares': 'Shares',
        'gain_percent': '% Gain',
        'gain': 'Gain',
        'stop_loss': 'Stop Loss',
        'profit_target': 'Profit Take'
    }

    updates = [('headers', ''.join(headers[col].ljust(col_widths[col]) for col in headers) + '\n')]

    total_portfolio_gain = 0.0

    for holding in holdings:
        ticker_sym = holding['symbol']
        price_paid = holding['price_paid']
        shares = holding['shares']
        stop_loss = holding['stop_loss']
        profit_target = holding.get('profit_target')

        stock = yf.Ticker(ticker_sym)
        stock_data = stock.history(period="1d")

        if stock_data.empty:
            continue

        last_price = round(stock_data['Close'].iloc[-1], 2)
        gain, gain_percent = calculate_gain(price_paid, last_price, shares)

        gain = round(gain, 2)
        gain_percent = round(gain_percent, 2)

        profit_target_str = f'{profit_target:.2f}' if profit_target else 'N/A'

        # Append each stock information line with fixed column widths
        line_data = {
            'ticker': ticker_sym,
            'last_price': f'{last_price:.2f}',
            'price_paid': f'{price_paid:.2f}',
            'shares': str(shares),
            'gain_percent': f'{gain_percent:.2f}%',
            'gain': f'{gain:.2f}',
            'stop_loss': f'{stop_loss:.2f}',
            'profit_target': profit_target_str
        }

        # Check if last price is above or below the price paid
        last_price_color = 'change positive' if last_price > price_paid else 'change negative'
        gain_percent_color = 'change positive' if gain_percent >= 0 else 'change negative'

        # Append each stock information line with fixed column widths and appropriate colors
        updates.append(('white', line_data['ticker'].ljust(col_widths['ticker'])))
        updates.append((last_price_color, line_data['last_price'].ljust(col_widths['last_price'])))
        updates.append(('white', line_data['price_paid'].ljust(col_widths['price_paid'])))
        updates.append(('white', line_data['shares'].ljust(col_widths['shares'])))
        updates.append((gain_percent_color, line_data['gain_percent'].ljust(col_widths['gain_percent'])))
        updates.append((get_color(gain), line_data['gain'].ljust(col_widths['gain'])))
        updates.append(('white', line_data['stop_loss'].ljust(col_widths['stop_loss'])))
        updates.append(('white', line_data['profit_target'].ljust(col_widths['profit_target']) + '\n'))

        total_portfolio_gain += gain

    # Ensure a space above 'Net Portfolio Gain'
    updates.append(('', '\n'))

    # Append the net portfolio gain with the correct tuple structure
    gain_color = get_color(total_portfolio_gain)
    net_gain_text = f'Net Portfolio Gain: {total_portfolio_gain:.2f}'
    updates.append((gain_color, net_gain_text))

    return updates

# Handle key presses
def handle_input(key):
    if key in ('R', 'r'):
        refresh(main_loop, '')

    if key in ('Q', 'q'):
        raise urwid.ExitMainLoop()

def refresh(_loop, _data):
    main_loop.draw_screen()
    quote_box.base_widget.set_text(get_update())
    main_loop.set_alarm_in(60, refresh)

main_loop = urwid.MainLoop(layout, palette, unhandled_input=handle_input)

def cli():
    main_loop.set_alarm_in(0, refresh)
    main_loop.run()

if __name__ == "__main__":
    add_new_holdings()
    cli()
