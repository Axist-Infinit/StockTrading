import urwid
import yfinance as yf

def add_new_holdings():
    while True:
        add_more = input("Add new holdings? (yes/no): ").lower()
        if add_more not in ['yes', 'no']:
            print("Please answer with 'yes' or 'no'.")
            continue
        if add_more == 'no':
            break

        # Gather stock details from the user
        stock_name = input("Name of stock: ").upper()
        price_paid = input("Price paid: ")
        number_of_shares = input("Number of shares: ")
        stop_loss = input("Stop loss: ")

        # Append new holding to tickers.txt
        with open("tickers.txt", "a") as file:
            file.write(f"{stock_name},{price_paid},{stop_loss},{number_of_shares}\n")
        print(f"Added {stock_name} to holdings.")

def parse_lines(lines):
    for l in lines:
        ticker = l.strip().split(",")
        yield ticker

# Read files and get symbols
with open("tickers.txt") as file:
    tickers = list(parse_lines(file.readlines()))

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

def pos_neg_change(change):
    if not change:
        return "0"
    else:
        return ("+{}".format(change) if change > 0 else "{}".format(change))

def get_color(change):
    return 'change positive' if change >= 0 else 'change negative'

def append_text(l, s, tabsize=10, color='white'):
    l.append((color, s.expandtabs(tabsize)))

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
        'stop_loss': 15
    }
    
    # Create headers with appropriate spacing
    headers = {
        'ticker': 'Stock',
        'last_price': 'Last Price',
        'price_paid': 'Price Paid',
        'shares': 'Shares',
        'gain_percent': '% Gain',
        'gain': 'Gain',
        'stop_loss': 'Stop Loss'
    }
    
    updates = [('headers', ''.join(headers[col].ljust(col_widths[col]) for col in headers) + '\n')]
    
    total_portfolio_gain = 0.0
    
    for ticker_info in tickers:
        ticker_sym = ticker_info[0]
        stock = yf.Ticker(ticker_sym)
        stock_data = stock.history(period="1d")
        
        last_price = round(stock_data['Close'].iloc[-1], 2)
        price_paid = round(float(ticker_info[1]), 2)
        shares = int(ticker_info[3])
        stop_loss = round(float(ticker_info[2]), 2)
        gain, gain_percent = calculate_gain(price_paid, last_price, shares)
        
        gain = round(gain, 2)
        gain_percent = round(gain_percent, 2)
        
        # Append each stock information line with fixed column widths
        line_data = {
            'ticker': ticker_sym,
            'last_price': f'{last_price:.2f}',
            'price_paid': f'{price_paid:.2f}',
            'shares': str(shares),
            'gain_percent': f'{gain_percent:.2f}%',
            'gain': f'{gain:.2f}',
            'stop_loss': f'{stop_loss:.2f}'
        }
        

        stock_line = ''.join(line_data[col].ljust(col_widths[col]) for col in line_data)
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
        updates.append(('white', line_data['stop_loss'].ljust(col_widths['stop_loss']) + '\n'))

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
