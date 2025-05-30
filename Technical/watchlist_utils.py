# watchlist_utils.py

import json
import os

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    """Load watchlist from JSON. Returns a list of tickers."""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f)

def save_watchlist(tickers):
    """Save watchlist (list of tickers) to JSON."""
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(tickers, f, indent=2)

def manage_watchlist():
    """Interactive sub-menu to view/add/remove tickers (multiple at once)."""
    while True:
        print("\nWatchlist Management:")
        print("1. View watchlist")
        print("2. Add ticker(s)")
        print("3. Remove ticker(s)")
        print("0. Return to main menu")

        choice = input("Select an option: ").strip()
        if choice == '0':
            break
        elif choice == '1':
            tickers = load_watchlist()
            if not tickers:
                print("Your watchlist is empty.")
            else:
                print("Your watchlist:", ", ".join(tickers))

        elif choice == '2':
            # Allow comma-separated tickers (e.g. "AAPL, MSFT, TSLA")
            new_tickers_str = input("Enter tickers to add (comma-separated): ").strip().upper()
            ticker_list = [t.strip() for t in new_tickers_str.split(',') if t.strip()]

            if not ticker_list:
                print("No valid tickers entered.")
                continue

            watchlist = load_watchlist()
            added = []
            for t in ticker_list:
                if t not in watchlist:
                    watchlist.append(t)
                    added.append(t)
            if added:
                save_watchlist(watchlist)
                print(f"Added to watchlist: {', '.join(added)}")
            else:
                print("No new tickers were added (they might already exist).")

        elif choice == '3':
            watchlist = load_watchlist()
            if not watchlist:
                print("Your watchlist is empty.")
                continue

            print("Current watchlist:")
            for i, t in enumerate(watchlist, start=1):
                print(f"  {i}) {t}")

            selection_str = input(
                "Enter the numbers of the tickers you want to remove (comma-separated), or press Enter to cancel: "
            ).strip()
            if not selection_str:
                print("Remove operation canceled.")
                continue

            # Parse comma-separated indices
            try:
                indices_to_remove = [int(x) for x in selection_str.split(',')]
            except ValueError:
                print("Invalid input. Please enter integers separated by commas.")
                continue

            # Build a new list without the selected indices
            updated_list = []
            removed_tickers = []
            for i, t in enumerate(watchlist, start=1):
                if i not in indices_to_remove:
                    updated_list.append(t)
                else:
                    removed_tickers.append(t)

            if removed_tickers:
                save_watchlist(updated_list)
                print(f"Removed: {', '.join(removed_tickers)}")
            else:
                print("No tickers were removed.")

        else:
            print("Invalid option. Please try again.")
