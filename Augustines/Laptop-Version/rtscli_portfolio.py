import json
import urwid
import yfinance as yf
import stock_forecast
from datetime import datetime  
from rtscli_data import load_holdings, save_holdings, load_trade_history, save_trade_history


def get_color(value):
    """Returns 'positive' if the value is greater than or equal to 0, else 'negative'."""
    return 'positive' if value >= 0 else 'negative'

def portfolio_cli():
    """Function to display the portfolio in a urwid-based UI."""
    holdings = load_holdings()
    if not holdings:
        print("You have no holdings. Please add holdings first.")
        input("\nPress Enter to return to the main menu...")
        return

    # Set up color scheme
    palette = [
        ('titlebar', 'white,bold', ''),
        ('refresh button', 'dark green,bold', ''),
        ('delete button', 'dark cyan,bold', ''),
        ('quit button', 'dark red', ''),
        ('headers', 'light blue,bold', ''),
        ('positive', 'dark green', ''),
        ('negative', 'dark red', ''),
        ('stop_loss_hit', 'white', 'dark red'),      # Background red for stop loss hit
        ('profit_target_hit', 'white', 'dark green'),# Background green for profit target hit
        ('body', 'white', ''),
        ('footer', 'white,bold', ''),
    ]

    header_text = urwid.Text(u' Portfolio Manager', align='center')
    header = urwid.AttrMap(header_text, 'titlebar')

    # Create the menu
    menu = urwid.Text([
        u'Press (', ('refresh button', u'R'), u') to refresh. ',
        u'Press (', ('delete button', u'D'), u') to delete holdings that have hit Stop Loss or Profit Target. ',
        u'Press (', ('quit button', u'Q'), u') to return to main menu. ',
        u'Press (P) to view Prediction History.'
    ])
    footer = urwid.AttrMap(menu, 'footer')

    # Create the initial body text
    body_text = urwid.Text(u'Fetching portfolio data...', align='left')
    body_filler = urwid.Filler(body_text, valign='top')
    body_padding = urwid.Padding(body_filler, left=1, right=1)
    body = urwid.AttrMap(body_padding, 'body')

    # Assemble the widgets
    layout = urwid.Frame(header=header, body=body, footer=footer)

    def get_portfolio_data():
        """Fetches and formats the portfolio data for display."""
        updates = []

        # Define column widths and order
        columns = [
            ('symbol', 'Symbol', 8),
            ('last_price', 'Last Price', 12),
            ('price_paid', 'Price Paid', 12),
            ('shares', 'Shares', 8),
            ('gain_percent', '% Gain', 10),
            ('gain', 'Gain/Loss', 12),
            ('strategy', 'Strategy', 12),
            ('stop_loss', 'Stop Loss', 12),
            ('profit_target', 'Profit Take', 12),
            ('status', 'Status', 18)
        ]

        # Create headers
        header_line = ''.join(header.ljust(width) for key, header, width in columns)
        updates.append(('headers', header_line + '\n'))

        total_portfolio_gain = 0.0

        for holding in holdings:
            symbol = holding['symbol']
            price_paid = holding['price_paid']
            shares = holding['shares']
            stop_loss = holding['stop_loss']
            profit_target = holding.get('profit_target')
            strategy = holding.get('strategy', 'Manual')  # Get the strategy or default to 'Manual'

            try:
                stock = yf.Ticker(symbol)
                stock_data = stock.history(period="1d")

                if stock_data.empty:
                    continue

                last_price = round(stock_data['Close'].iloc[-1], 2)
                gain_per_share = last_price - price_paid
                gain = gain_per_share * shares
                gain_percent = (gain_per_share / price_paid) * 100 if price_paid != 0 else 0.0

                total_portfolio_gain += gain

                profit_target_str = f'${profit_target:.2f}' if profit_target else 'N/A'

                # Check if stop loss or profit target is hit
                status = 'Holding'
                attr = None  # Attribute for coloring

                if last_price <= stop_loss:
                    status = 'Stop Loss Hit'
                    attr = 'stop_loss_hit'
                elif profit_target and last_price >= profit_target:
                    status = 'Profit Target Hit'
                    attr = 'profit_target_hit'

                line_data = {
                    'symbol': symbol,
                    'last_price': f'${last_price:.2f}',
                    'price_paid': f'${price_paid:.2f}',
                    'shares': str(shares),
                    'gain_percent': f'{gain_percent:.2f}%',
                    'gain': f'${gain:.2f}',
                    'strategy': strategy,
                    'stop_loss': f'${stop_loss:.2f}',
                    'profit_target': profit_target_str,
                    'status': status
                }

                # Build the line with specified colors
                line = []
                if attr:
                    # Apply the attribute (coloring) to the entire line
                    for key, header, width in columns:
                        line.append((attr, line_data[key].ljust(width)))
                else:
                    for key, header, width in columns:
                        value = line_data[key]
                        if key == 'last_price':
                            color = get_color(float(value.strip('$')) - price_paid)
                        elif key == 'gain_percent':
                            color = get_color(float(value.strip('%')))
                        elif key == 'gain':
                            color = get_color(float(value.strip('$')))
                        else:
                            color = 'white'
                        line.append((color, value.ljust(width)))
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
        elif key in ('D', 'd'):
            delete_hit_holdings()
            refresh()
        elif key in ('Q', 'q'):
            raise urwid.ExitMainLoop()
        elif key in ('P', 'p'):
            # View Prediction History from stock_forecast
            stock_forecast.prediction_history_cli()
            input("Press Enter to return to the portfolio view...")
            refresh()

    def refresh():
        body_text.set_text(get_portfolio_data())

    def delete_hit_holdings():
        """Deletes holdings that have hit stop loss or profit target."""
        nonlocal holdings  # Modify the holdings list in the outer scope
        updated_holdings = []
        removed_holdings = []
        for holding in holdings:
            symbol = holding['symbol']
            price_paid = holding['price_paid']
            shares = holding['shares']
            stop_loss = holding['stop_loss']
            profit_target = holding.get('profit_target')
            strategy = holding.get('strategy', 'Manual')
            entry_date = holding.get('entry_date', datetime.now().strftime('%Y-%m-%d'))

            try:
                stock = yf.Ticker(symbol)
                stock_data = stock.history(period="1d")
                if stock_data.empty:
                    updated_holdings.append(holding)
                    continue

                last_price = round(stock_data['Close'].iloc[-1], 2)

                # Check if stop loss or profit target is hit
                result = None
                if last_price <= stop_loss:
                    result = 'Stop Loss Hit'
                elif profit_target and last_price >= profit_target:
                    result = 'Profit Target Hit'

                if result:
                    # Add to removed holdings for trade history
                    holding['exit_date'] = datetime.now().strftime('%Y-%m-%d')
                    holding['exit_price'] = last_price
                    holding['result'] = result
                    removed_holdings.append(holding)
                else:
                    updated_holdings.append(holding)
            except Exception:
                updated_holdings.append(holding)

        if removed_holdings:
            removed_symbols = [h['symbol'] for h in removed_holdings]
            confirm = input(f"Remove holdings that have hit Stop Loss or Profit Target ({', '.join(removed_symbols)})? (yes/no): ").lower()
            if confirm == 'yes':
                holdings = updated_holdings
                save_holdings(holdings)
                print(f"Removed holdings: {', '.join(removed_symbols)}")

                # Load existing trade history
                trade_history = load_trade_history()
                # Append removed holdings to trade history
                trade_history.extend(removed_holdings)
                # Save updated trade history
                save_trade_history(trade_history)
            else:
                print("Deletion canceled.")
        else:
            print("No holdings have hit Stop Loss or Profit Target.")

    refresh()
    main_loop = urwid.MainLoop(layout, palette, unhandled_input=handle_input)
    main_loop.run()

