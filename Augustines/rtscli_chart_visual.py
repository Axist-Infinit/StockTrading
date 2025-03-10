# rtscli_chart_visual.py

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# Strategy => Marker shape mapping
STRATEGY_MARKERS = {
    "Swing": "^",       # Upward triangle
    "Pullback": "o",    # Circle
    "Breakout": "s",    # Square
    "VWAP": "D",        # Diamond
    "Unknown": "^"      # Default marker
}

# Exit reason => color mapping
EXIT_REASON_COLORS = {
    "Profit Target": "green",
    "Hit Stop Loss": "red",
    "Still Open": "orange",
    "EndOfData": "blue",
    "Unknown": "gray"
}

def visualize_backtest_candlestick(ticker, df, trades_df, title=None, show=True, save_path=None):
    """
    Creates a candlestick chart using `mplfinance` with:
      - 20 & 50 SMAs
      - Strategy-based markers for entry/exit
      - Colored exit markers by exit reason

    :param ticker:     The stock ticker symbol (string).
    :param df:         A pandas DataFrame of historical price data 
                       with columns at least ['Open','High','Low','Close','Volume'], 
                       indexed by date (DateTimeIndex).
    :param trades_df:  A pandas DataFrame with columns:
                       ['Strategy','EntryDate','EntryPrice','ExitDate','ExitPrice','ExitReason']
    :param title:      Custom title for the plot (optional).
    :param show:       If True, displays the plot via plt.show().
    :param save_path:  If provided, saves the plot to this file path (e.g. 'chart.png').
    """

    # 1) Sanity check: Is df empty or short?
    if df.empty or len(df) < 2:
        print(f"No or insufficient data for {ticker}. Cannot plot candlestick chart.")
        return  # Skip plotting entirely

    # 2) Ensure DataFrame is sorted and indexed by datetime
    df = df.sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # 3) Convert trades_df 'EntryDate'/'ExitDate' to datetime if needed
    if 'EntryDate' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['EntryDate']):
        trades_df['EntryDate'] = pd.to_datetime(trades_df['EntryDate'])
    if 'ExitDate' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['ExitDate']):
        trades_df['ExitDate'] = pd.to_datetime(trades_df['ExitDate'])

    # 4) Add 20 & 50 SMA if missing
    if 'SMA_20' not in df.columns:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 5) Build additional plots list
    apds = []
    apds.append(mpf.make_addplot(df['SMA_20'], color='orange', width=1.0, label='20 SMA'))
    apds.append(mpf.make_addplot(df['SMA_50'], color='purple', width=1.0, label='50 SMA'))

    # 6) Build columns for entry/exit per strategy
    unique_strategies = trades_df['Strategy'].unique() if 'Strategy' in trades_df.columns else ["Unknown"]

    for strat in unique_strategies:
        entry_col = f"{strat}_Entry"
        exit_col  = f"{strat}_Exit"
        # Force float dtype using np.nan
        df[entry_col] = np.nan
        df[exit_col]  = np.nan

        # Populate columns with trade data
        strat_trades = trades_df[trades_df['Strategy'] == strat]
        for _, trade in strat_trades.iterrows():
            entry_date  = trade.get('EntryDate')
            entry_price = trade.get('EntryPrice')
            exit_date   = trade.get('ExitDate')
            exit_price  = trade.get('ExitPrice')

            if pd.notna(entry_date) and entry_date in df.index:
                df.at[entry_date, entry_col] = entry_price
            if pd.notna(exit_date) and exit_date in df.index:
                df.at[exit_date, exit_col] = exit_price

        # Convert to float to avoid NAType issues
        df[entry_col] = df[entry_col].astype(float)
        df[exit_col]  = df[exit_col].astype(float)

        # Add scatter plots for entry/exit
        marker_shape = STRATEGY_MARKERS.get(strat, STRATEGY_MARKERS["Unknown"])
        apds.append(
            mpf.make_addplot(
                df[entry_col],
                type='scatter',
                markersize=100,
                marker=marker_shape,
                color='lime',  # All entries lime
                panel=0,
                secondary_y=False
            )
        )
        apds.append(
            mpf.make_addplot(
                df[exit_col],
                type='scatter',
                markersize=100,
                marker=marker_shape,
                color='magenta',  # All exits magenta (overridden below for exit reason)
                panel=0,
                secondary_y=False
            )
        )

    # 7) Choose style
    try:
        style = mpf.make_mpf_style(base_mpl_style='ggplot', gridstyle='--', gridaxis='both')
    except ValueError:
        # fallback style if 'ggplot' not available
        style = mpf.make_mpf_style(base_mpl_style='classic', gridstyle='--', gridaxis='both')

    # 8) Define figure title
    fig_title = title if title else f"{ticker} Candlestick Chart with Trades"

    # 9) Plot the candlestick
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style=style,
        addplot=apds,
        volume=True,
        title=fig_title,
        figratio=(16, 8),
        figscale=1.2,
        returnfig=True
    )

    # 10) Color exit markers by reason using a post-plot scatter
    ax_main = axlist[0]  # main price axis
    for _, trade in trades_df.iterrows():
        strat       = trade.get('Strategy', 'Unknown')
        exit_date   = trade.get('ExitDate')
        exit_price  = trade.get('ExitPrice')
        exit_reason = trade.get('ExitReason', 'Unknown')
        color       = EXIT_REASON_COLORS.get(exit_reason, 'gray')
        marker      = STRATEGY_MARKERS.get(strat, STRATEGY_MARKERS["Unknown"])

        if pd.notna(exit_date) and exit_date in df.index and pd.notna(exit_price):
            ax_main.scatter(
                exit_date,
                exit_price,
                color=color,
                marker=marker,
                s=100,
                edgecolors='black',
                zorder=5,
                label=f"{strat} Exit - {exit_reason}"
            )

    # 11) Deduplicate legend labels
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys())

    # 12) Layout
    plt.tight_layout()

    # 13) Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved chart to {save_path}")

    # 14) Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)
