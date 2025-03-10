# rtscli_data.py

import json
import os

# File paths
HOLDINGS_FILE_JSON = 'holdings.json'
TRADE_HISTORY_FILE = 'trade_history.json'

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

# Holdings Functions
def load_holdings():
    data = load_json_file(HOLDINGS_FILE_JSON, {'holdings': []})
    return data.get('holdings', [])

def save_holdings(holdings):
    save_json_file(HOLDINGS_FILE_JSON, {'holdings': holdings})

# Watchlist Functions
def load_watchlist(watchlist_name='main'):
    watchlist_file = f'{watchlist_name}_watchlist.json'
    data = load_json_file(watchlist_file, {'stocks': []})
    return data.get('stocks', [])

def save_watchlist(stocks, watchlist_name='main'):
    watchlist_file = f'{watchlist_name}_watchlist.json'
    save_json_file(watchlist_file, {'stocks': stocks})

# Trade History Functions
def load_trade_history():
    """Load trade history from a JSON file, returning an empty list if none exists."""
    data = load_json_file(TRADE_HISTORY_FILE, {'trade_history': []})
    return data.get('trade_history', [])

def save_trade_history(trade_history):
    """Save trade history to a JSON file."""
    save_json_file(TRADE_HISTORY_FILE, {'trade_history': trade_history})

