import yfinance as yf
import pandas as pd
import numpy as np
import urwid
import time
from datetime import datetime, timedelta

# Define the VIX ETF ticker symbol
TICKER = 'VXX'  # iPath Series B S&P 500 VIX Short-Term Futures ETN

# Trading parameters
MOVING_AVERAGE_PERIOD = 20  # Period for moving average
STD_DEV_MULTIPLIER = 2  # Multiplier for Bollinger Bands
RSI_PERIOD = 14  # Period for RSI
TRADE_AMOUNT = 1000  # Amount in dollars for each trade
TAKE_PROFIT_RATIO = 2  # Profit ratio (2:1)
STOP_LOSS_RATIO = 1  # Loss ratio

# Initialize trade history and position
trade_history = []
position = None  # {'type': 'long' or 'short', 'entry_price': float, 'entry_time': datetime}

# Function to fetch real-time data
def fetch_data(ticker, period='1d', interval='5m'):
    data = yf.download(tickers=ticker, period=period, interval=interval)
    return data

# Function to compute technical indicators
def compute_indicators(data):
    data['MA'] = data['Close'].rolling(MOVING_AVERAGE_PERIOD).mean()
    data['STD'] = data['Close'].rolling(MOVING_AVERAGE_PERIOD).std()
    data['Upper Band'] = data['MA'] + (data['STD'] * STD_DEV_MULTIPLIER)
    data['Lower Band'] = data['MA'] - (data['STD'] * STD_DEV_MULTIPLIER)
    delta = data['Close'].diff(1)
    delta = delta.dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    RS_up = up.rolling(RSI_PERIOD).mean()
    RS_down = down.rolling(RSI_PERIOD).mean()
    data['RSI'] = 100 - (100 / (1 + RS_up / RS_down))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to generate trading signals
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0

    # Buy signal conditions
    buy_conditions = (
        (data['Close'] < data['Lower Band']) &  # Price below lower Bollinger Band
        (data['RSI'] < 30) &  # RSI indicates oversold
        (data['MACD'] < data['Signal Line'])  # MACD indicates bullish crossover soon
    )
    signals.loc[buy_conditions, 'Signal'] = 1

    # Sell signal conditions
    sell_conditions = (
        (data['Close'] > data['Upper Band']) &  # Price above upper Bollinger Band
        (data['RSI'] > 70) &  # RSI indicates overbought
        (data['MACD'] > data['Signal Line'])  # MACD indicates bearish crossover soon
    )
    signals.loc[sell_conditions, 'Signal'] = -1

    return signals

# Function to speculate buy levels when no signal is given
def speculate_buy_levels(data):
    last_close = data['Close'][-1]
    lower_band = data['Lower Band'][-1]
    rsi = data['RSI'][-1]
    macd = data['MACD'][-1]
    signal_line = data['Signal Line'][-1]

    speculative_price = lower_band  # Suggest buying at the lower Bollinger Band
    return speculative_price

# Function to check if profit target or stop loss has been hit
def check_exit_conditions(latest_price):
    global position, trade_history
    if position is None:
        return

    entry_price = position['entry_price']
    position_type = position['type']

    take_profit_price = entry_price * (1 + TAKE_PROFIT_RATIO / 100) if position_type == 'long' else entry_price * (1 - TAKE_PROFIT_RATIO / 100)
    stop_loss_price = entry_price * (1 - STOP_LOSS_RATIO / 100) if position_type == 'long' else entry_price * (1 + STOP_LOSS_RATIO / 100)

    if position_type == 'long':
        if latest_price >= take_profit_price:
            profit = (latest_price - entry_price) * (TRADE_AMOUNT / entry_price)
            trade_history.append({
                'Entry Time': position['entry_time'],
                'Exit Time': datetime.now(),
                'Position': 'Long',
                'Entry Price': entry_price,
                'Exit Price': latest_price,
                'Profit': profit
            })
            position = None
            return 'Profit Target Hit'
        elif latest_price <= stop_loss_price:
            loss = (latest_price - entry_price) * (TRADE_AMOUNT / entry_price)
            trade_history.append({
                'Entry Time': position['entry_time'],
                'Exit Time': datetime.now(),
                'Position': 'Long',
                'Entry Price': entry_price,
                'Exit Price': latest_price,
                'Profit': loss
            })
            position = None
            return 'Stop Loss Hit'
    elif position_type == 'short':
        if latest_price <= take_profit_price:
            profit = (entry_price - latest_price) * (TRADE_AMOUNT / entry_price)
            trade_history.append({
                'Entry Time': position['entry_time'],
                'Exit Time': datetime.now(),
                'Position': 'Short',
                'Entry Price': entry_price,
                'Exit Price': latest_price,
                'Profit': profit
            })
            position = None
            return 'Profit Target Hit'
        elif latest_price >= stop_loss_price:
            loss = (entry_price - latest_price) * (TRADE_AMOUNT / entry_price)
            trade_history.append({
                'Entry Time': position['entry_time'],
                'Exit Time': datetime.now(),
                'Position': 'Short',
                'Entry Price': entry_price,
                'Exit Price': latest_price,
                'Profit': loss
            })
            position = None
            return 'Stop Loss Hit'
    return 'Holding'

# Function to run the trading bot logic
def trading_logic():
    global position
    # Fetch data
    data = fetch_data(TICKER)
    data = compute_indicators(data)
    signals = generate_signals(data)

    # Get the latest data point
    latest_time = data.index[-1]
    latest_price = data['Close'][-1]
    latest_signal = signals['Signal'][-1]

    # If no position, check for entry signals
    if position is None:
        if latest_signal == 1:
            # Enter long position
            position = {
                'type': 'long',
                'entry_price': latest_price,
                'entry_time': latest_time
            }
            status = f"Entered Long Position at {latest_price}"
        elif latest_signal == -1:
            # Enter short position
            position = {
                'type': 'short',
                'entry_price': latest_price,
                'entry_time': latest_time
            }
            status = f"Entered Short Position at {latest_price}"
        else:
            speculative_price = speculate_buy_levels(data)
            status = f"No Signal. Speculated Buy Level: {speculative_price:.2f}"
    else:
        # Check exit conditions
        exit_status = check_exit_conditions(latest_price)
        status = f"Position Status: {exit_status}"

    return {
        'Time': latest_time,
        'Price': latest_price,
        'Position': position,
        'Status': status
    }

# Function to display the portfolio in a urwid-based UI
def portfolio_cli_vix():
    palette = [
        ('titlebar', 'white,bold', ''),
        ('refresh button', 'dark green,bold', ''),
        ('quit button', 'dark red', ''),
        ('headers', 'light blue,bold', ''),
        ('positive', 'dark green', ''),
        ('negative', 'dark red', ''),
        ('stop_loss_hit', 'white', 'dark red'),
        ('profit_target_hit', 'white', 'dark green'),
        ('body', 'white', ''),
        ('footer', 'white,bold', ''),
    ]

    header_text = urwid.Text(u' VXX Mean Reversion Strategy Monitor', align='center')
    header = urwid.AttrMap(header_text, 'titlebar')

    # Create the menu
    menu = urwid.Text([
        u'Press (', ('refresh button', u'R'), u') to refresh. ',
        u'Press (', ('quit button', u'Q'), u') to exit.'
    ])
    footer = urwid.AttrMap(menu, 'footer')

    # Create the initial body text
    body_text = urwid.Text(u'Fetching data...', align='left')
    body_filler = urwid.Filler(body_text, valign='top')
    body_padding = urwid.Padding(body_filler, left=1, right=1)
    body = urwid.AttrMap(body_padding, 'body')

    # Assemble the widgets
    layout = urwid.Frame(header=header, body=body, footer=footer)

    def get_display_data():
        updates = []
        # Fetch and process trading data
        trading_data = trading_logic()

        # Prepare display data
        time_str = trading_data['Time'].strftime('%Y-%m-%d %H:%M')
        price_str = f"{trading_data['Price']:.2f}"
        status_str = trading_data['Status']

        updates.append(('white', f"Time: {time_str}\n"))
        updates.append(('white', f"Price: {price_str}\n"))
        updates.append(('white', f"Status: {status_str}\n\n"))

        if position:
            # Display current position
            pos_type = position['type'].capitalize()
            entry_price = position['entry_price']
            entry_time = position['entry_time'].strftime('%Y-%m-%d %H:%M')
            updates.append(('white', f"Current Position: {pos_type}\n"))
            updates.append(('white', f"Entry Price: {entry_price:.2f}\n"))
            updates.append(('white', f"Entry Time: {entry_time}\n"))
            # Calculate unrealized profit/loss
            if pos_type == 'Long':
                unrealized_pnl = (trading_data['Price'] - entry_price) * (TRADE_AMOUNT / entry_price)
            else:
                unrealized_pnl = (entry_price - trading_data['Price']) * (TRADE_AMOUNT / entry_price)
            pnl_color = 'positive' if unrealized_pnl >= 0 else 'negative'
            updates.append((pnl_color, f"Unrealized P&L: {unrealized_pnl:.2f}\n"))
        else:
            updates.append(('white', "No Open Position\n"))

        # Display trade history
        updates.append(('headers', "\nTrade History:\n"))
        if trade_history:
            for trade in trade_history[-5:]:
                entry_time = trade['Entry Time'].strftime('%Y-%m-%d %H:%M')
                exit_time = trade['Exit Time'].strftime('%Y-%m-%d %H:%M')
                pos_type = trade['Position']
                profit = trade['Profit']
                pnl_color = 'positive' if profit >= 0 else 'negative'
                updates.append(('white', f"{pos_type} | Entry: {entry_time} | Exit: {exit_time} | "))
                updates.append((pnl_color, f"P&L: {profit:.2f}\n"))
        else:
            updates.append(('white', "No Trade History\n"))

        return updates

    # Handle key presses
    def handle_input(key):
        if key in ('R', 'r'):
            refresh()
        elif key in ('Q', 'q'):
            raise urwid.ExitMainLoop()

    def refresh():
        body_text.set_text(get_display_data())

    refresh()
    main_loop = urwid.MainLoop(layout, palette, unhandled_input=handle_input)
    main_loop.run()

def run_vix_trader():
    portfolio_cli_vix()

