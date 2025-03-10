import os
import json
import math
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
import yfinance as yf
from scipy.stats import norm
import pytz

PREDICTIONS_HISTORY_FILE = 'predictions_history.json'
TRADING_DAYS_PER_YEAR = 252  # Used for annualizing volatility


###############################
# 1) LOAD / SAVE PREDICTIONS
###############################
def load_predictions_history():
    if os.path.exists(PREDICTIONS_HISTORY_FILE):
        with open(PREDICTIONS_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_predictions_history(data):
    with open(PREDICTIONS_HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)


###############################
# 2) SHORT-DATED INTEREST RATE FETCH
###############################
def get_short_dated_rate():
    """
    Fetch a short-dated interest rate.
    Example: 13-week T-bill from Yahoo Finance ticker '^IRX'.
    ^IRX quotes are often in percentage points. For example, 1.50 represents 1.50%
    Convert to decimal.
    """
    try:
        irx = yf.Ticker("^IRX")
        info = irx.history(period="1d")
        if info.empty:
            print("Warning: ^IRX data is empty. Using fallback rate of 3%.")
            return 0.03  # Fallback to 3%
        # ^IRX is in percentage points (e.g., 1.50 represents 1.50%)
        annual_yield = float(info['Close'].iloc[-1]) / 100.0
        print(f"Short-dated rate fetched: {annual_yield*100:.2f}%")
        return annual_yield
    except Exception as e:
        print(f"Error fetching short-dated rate: {e}")
        return 0.03  # Fallback


###############################
# 3) MARKET VOLATILITY FORECAST
###############################
def get_market_vol_forecast():
    """
    Fetches the VIX index as a proxy for market volatility and categorizes it as High, Medium, or Low.
    """
    try:
        vix = yf.Ticker("^VIX")
        vix_info = vix.history(period="1d")
        if vix_info.empty:
            print("Warning: ^VIX data is empty. Using fallback forecast of Medium.")
            return "Medium"  # Fallback
        current_vix = float(vix_info['Close'].iloc[-1])
        print(f"Current VIX: {current_vix:.2f}")
        if current_vix > 25:
            return "High"
        elif current_vix > 20:
            return "Medium"
        else:
            return "Low"
    except Exception as e:
        print(f"Error fetching market volatility forecast: {e}")
        return "Medium"  # Fallback


###############################
# 4) DIVIDEND HANDLING
###############################
def get_dividend_info(ticker, expiration):
    """
    Fetch upcoming dividends before the option's expiration date.
    Returns continuous yield and lists of ex-dates and dividend amounts.
    """
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    if dividends.empty:
        return {
            "continuous_yield": 0.0,
            "ex_dates": [],
            "dividend_amounts": []
        }

    # Remove timezone from dividends.index for comparison
    ex_dates_naive = dividends.index.tz_localize(None) if dividends.index.tz else dividends.index

    exp_dt = parser.parse(expiration).replace(tzinfo=None)  # Ensure naive datetime
    upcoming_dividends = dividends[ex_dates_naive < exp_dt]

    ex_dates = upcoming_dividends.index.tolist()
    dividend_amounts = upcoming_dividends.values.tolist()

    # Continuous dividend yield
    info = stock.info
    q = info.get('dividendYield', 0.0)
    if q is None:
        q = 0.0

    return {
        "continuous_yield": float(q),
        "ex_dates": ex_dates,
        "dividend_amounts": dividend_amounts
    }


###############################
# 5) HISTORICAL REALIZED VOLATILITY
###############################
def get_historical_realized_vol(ticker, lookback_days=30):
    """
    Compute the annualized historical realized volatility based on daily log returns.
    """
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 5)  # Extra days to account for weekends
    hist = stock.history(start=start_date, end=end_date)
    if len(hist) < 2:
        print(f"Warning: Not enough historical data for {ticker}.")
        return None
    hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
    vol_daily = hist['log_ret'].std()
    if not math.isfinite(vol_daily) or vol_daily <= 0:
        print(f"Warning: Invalid realized volatility for {ticker}.")
        return None
    vol_annualized = vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
    print(f"Historical realized volatility for {ticker}: {vol_annualized:.2%}")
    return vol_annualized


###############################
# 6) BINOMIAL TREE FOR AMERICAN OPTIONS
###############################
def binomial_probability_itm_american(S, K, T, r, q, sigma, steps=100, is_call=True):
    """
    Approximate Probability ITM for American options using a binomial tree.
    """
    if T <= 0 or sigma <= 0:
        print("Warning: Invalid T or sigma in binomial_probability_itm_american.")
        return 0.0  # Avoid division by zero

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))       # Up factor
    d = 1.0 / u                                # Down factor
    denominator = u - d
    if denominator == 0:
        print(f"Warning: u - d is zero for S={S}, K={K}, T={T}, sigma={sigma}")
        return 0.0  # Avoid division by zero
    pu = (math.exp((r - q) * dt) - d) / denominator
    pd = 1.0 - pu
    discount = math.exp(-r * dt)

    # Initialize asset prices at maturity
    asset_prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]

    # Initialize option values at maturity
    if is_call:
        option_values = [max(price - K, 0.0) for price in asset_prices]
        prob_itm = [1.0 if price > K else 0.0 for price in asset_prices]
    else:
        option_values = [max(K - price, 0.0) for price in asset_prices]
        prob_itm = [1.0 if price < K else 0.0 for price in asset_prices]

    # Backward induction for option pricing and tracking Probability ITM
    for i in reversed(range(steps)):
        for j in range(i + 1):
            # Continuation value
            continuation = discount * (pu * option_values[j + 1] + pd * option_values[j])

            # Intrinsic value
            if is_call:
                intrinsic = max(S * (u ** j) * (d ** (i - j)) - K, 0.0)
            else:
                intrinsic = max(K - S * (u ** j) * (d ** (i - j)), 0.0)

            # Early exercise
            option_values[j] = max(continuation, intrinsic)

            # Update Probability ITM
            prob_itm[j] = discount * (pu * prob_itm[j + 1] + pd * prob_itm[j])

    return prob_itm[0]


###############################
# 7) BARRIER-BASED CHANCE OF TOUCH
###############################
def barrier_chance_of_touch_binomial(S, K, T, r, q, sigma, steps=100, is_call=True):
    """
    Use a binomial tree to approximate the probability of touching the barrier K at any point before expiration.
    """
    if T <= 0 or sigma <= 0:
        print("Warning: Invalid T or sigma in barrier_chance_of_touch_binomial.")
        return 0.0  # Avoid division by zero

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))       
    d = 1.0 / u
    denominator = u - d
    if denominator == 0:
        print(f"Warning: u - d is zero for S={S}, K={K}, T={T}, sigma={sigma}")
        return 0.0  # Avoid division by zero
    pu = (math.exp((r - q) * dt) - d) / denominator
    pd = 1.0 - pu

    # Each node: (price, probability, touched)
    layer = [(S, 1.0, (S >= K if is_call else S <= K))]

    for step_idx in range(1, steps + 1):
        new_layer = []
        for (price, prob, touched) in layer:
            # Up move
            s_up = price * u
            up_touched = touched or (s_up >= K if is_call else s_up <= K)
            new_layer.append((s_up, prob * pu, up_touched))

            # Down move
            s_down = price * d
            down_touched = touched or (s_down >= K if is_call else s_down <= K)
            new_layer.append((s_down, prob * pd, down_touched))

        layer = new_layer

    # Sum probabilities where touched is True
    prob_touched = sum(prob for (price, prob, touched) in layer if touched)
    return prob_touched


###############################
# 8) BLACK–SCHOLES + PROB ITM
###############################
def black_scholes_option_value(S, K, T, r, q, sigma, option_type="call"):
    """
    Returns a dict with:
      - price   : Black–Scholes fair value (European)
      - P_ITM   : Probability of expiring in the money
    """
    if T <= 0 or sigma <= 0:
        print("Warning: Invalid T or sigma in black_scholes_option_value.")
        return {"price": 0.0, "P_ITM": 0.0}

    denominator = sigma * math.sqrt(T)
    if denominator == 0:
        print(f"Warning: Denominator is zero for S={S}, K={K}, T={T}, sigma={sigma}")
        return {"price": 0.0, "P_ITM": 0.0}

    numerator = np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
    d1 = numerator / denominator
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        p_itm = norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        p_itm = norm.cdf(-d2)

    return {"price": float(price), "P_ITM": float(p_itm)}


###############################
# 9) ANALYZE A SINGLE OPTION (CALL or PUT)
###############################
def analyze_best_otm_option(ticker, option_type="call", fetched_data=None):
    """
    Analyze the best OTM option (call or put) for a given ticker.
    Returns a tuple: (best_pick_dict, fetched_data)
    """
    stock = yf.Ticker(ticker)

    # Fetch intraday data only once if not already fetched
    if fetched_data and 'hist_intraday' in fetched_data:
        hist_intraday = fetched_data['hist_intraday']
    else:
        hist_intraday = stock.history(period="1d", interval="1m")
        if hist_intraday.empty:
            raise ValueError(f"No intraday data for {ticker}.")
        if fetched_data is None:
            fetched_data = {}
        fetched_data['hist_intraday'] = hist_intraday

    current_price = float(hist_intraday['Close'].iloc[-1])
    print(f"Current price for {ticker}: {current_price}")

    # Fetch options only once if not already fetched
    if fetched_data and 'options' in fetched_data:
        exps = fetched_data['options']
    else:
        exps = stock.options
        if not exps:
            return None, fetched_data
        if fetched_data is None:
            fetched_data = {}
        fetched_data['options'] = exps

    # Use the nearest expiration
    expiration = exps[0]
    print(f"Nearest expiration for {ticker}: {expiration}")

    # Fetch option chain only once if not already fetched
    if fetched_data and 'option_chain' in fetched_data and expiration in fetched_data['option_chain']:
        chain = fetched_data['option_chain'][expiration]
    else:
        chain = stock.option_chain(expiration)
        if 'option_chain' not in fetched_data:
            fetched_data['option_chain'] = {}
        fetched_data['option_chain'][expiration] = chain

    if option_type.lower() == "call":
        df = chain.calls
        df = df[df['strike'] > current_price].copy()  # OTM calls
        df.sort_values(by="strike", inplace=True)
        is_call = True
    else:
        df = chain.puts
        df = df[df['strike'] < current_price].copy()  # OTM puts
        df.sort_values(by="strike", ascending=False, inplace=True)
        is_call = False

    if df.empty:
        print(f"No suitable OTM {option_type.capitalize()} found for {ticker}.")
        return None, fetched_data

    # Time to expiration in years using total seconds
    now = datetime.now()
    exp_dt = parser.parse(expiration).replace(tzinfo=None)  # Ensure naive datetime
    time_delta = exp_dt - now
    T = max(time_delta.total_seconds() / (TRADING_DAYS_PER_YEAR * 24 * 60 * 60), 0)  # Time in years
    print(f"Time to expiration (T) for {ticker}: {T:.6f} years")

    # Fetch short-dated rate only once
    if fetched_data and 'short_dated_rate' in fetched_data:
        r = fetched_data['short_dated_rate']
    else:
        r = get_short_dated_rate()
        fetched_data['short_dated_rate'] = r

    # Fetch dividend info only once
    if fetched_data and 'dividend_info' in fetched_data and expiration in fetched_data['dividend_info']:
        div_info = fetched_data['dividend_info'][expiration]
    else:
        div_info = get_dividend_info(ticker, expiration)
        if 'dividend_info' not in fetched_data:
            fetched_data['dividend_info'] = {}
        fetched_data['dividend_info'][expiration] = div_info

    q = div_info['continuous_yield']
    print(f"Continuous dividend yield (q) for {ticker}: {q:.4f}")

    # Fetch historical realized volatility only once
    if fetched_data and 'realized_vol' in fetched_data:
        realized_vol = fetched_data['realized_vol']
    else:
        realized_vol = get_historical_realized_vol(ticker, lookback_days=30)
        fetched_data['realized_vol'] = realized_vol

    if realized_vol is None or realized_vol <= 0 or not math.isfinite(realized_vol):
        print(f"Invalid realized volatility for {ticker}. Skipping analysis.")
        return None, fetched_data

    results = []
    for _, row in df.iterrows():
        strike = float(row['strike'])
        last_price = float(row['lastPrice'])
        iv = float(row['impliedVolatility'])

        if last_price <= 0 or iv <= 0:
            continue

        # Additional Validation: Ensure last_price is finite
        if not math.isfinite(last_price):
            print(f"Skipping option with invalid last_price: {last_price}")
            continue

        # Probability ITM via binomial American approach
        try:
            prob_itm_bin = binomial_probability_itm_american(
                S=current_price,
                K=strike,
                T=T,
                r=r,
                q=q,
                sigma=iv,
                steps=100,
                is_call=is_call
            )
        except ZeroDivisionError:
            print(f"ZeroDivisionError in binomial_probability_itm_american for strike={strike}")
            continue

        # Chance of reaching barrier
        try:
            chance_touch_bin = barrier_chance_of_touch_binomial(
                S=current_price,
                K=strike,
                T=T,
                r=r,
                q=q,
                sigma=iv,
                steps=100,
                is_call=is_call
            )
        except ZeroDivisionError:
            print(f"ZeroDivisionError in barrier_chance_of_touch_binomial for strike={strike}")
            continue

        # Validate prob_itm_bin
        if not math.isfinite(prob_itm_bin):
            print(f"Invalid prob_itm_bin: {prob_itm_bin} for strike={strike}")
            continue

        # Score calculation
        try:
            score = prob_itm_bin / last_price if last_price else 0
        except ZeroDivisionError:
            print(f"Attempted division by zero for score with last_price={last_price}")
            continue

        # Compare implied vol to realized vol
        vol_ratio = iv / realized_vol if realized_vol and realized_vol > 0 else None

        # Black-Scholes Fair Price
        try:
            bs_res = black_scholes_option_value(
                S=current_price,
                K=strike,
                T=T,
                r=r,
                q=q,
                sigma=iv,
                option_type=option_type
            )
            bs_price = bs_res["price"]
        except ZeroDivisionError:
            print(f"ZeroDivisionError in black_scholes_option_value for strike={strike}")
            bs_price = 0.0

        pick = {
            "ticker": ticker.upper(),
            "option_type": option_type,
            "expiration": expiration,
            "strike": strike,
            "last_price": last_price,
            "prob_itm": prob_itm_bin,
            "chance_touch": chance_touch_bin,
            "score": score,
            "current_price": current_price,
            "iv": iv,
            "realized_vol": realized_vol,
            "vol_ratio": vol_ratio,
            "bs_price": bs_price
        }
        results.append(pick)

    if not results:
        print(f"No valid OTM {option_type.capitalize()} options found for {ticker}.")
        return None, fetched_data

    # Sort by Score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    best_pick = results[0]
    print(f"Best OTM {option_type.capitalize()} for {ticker}: Strike {best_pick['strike']}, Score {best_pick['score']:.4f}")
    return best_pick, fetched_data  # Return the best pick and the fetched_data for reuse


###############################
# 10) LOG PREDICTIONS
###############################
def log_prediction(pick):
    """
    Log the pick to predictions_history.json with a unique key.
    """
    if not pick:
        return
    hist = load_predictions_history()
    key = f"{pick['ticker']}_{pick['expiration']}_{pick['strike']}_{pick['option_type'].lower()}"
    hist[key] = {
        "date_queried": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": pick["ticker"],
        "option_type": pick["option_type"],
        "expiration": pick["expiration"],
        "strike": pick["strike"],
        "last_price": pick["last_price"],
        "prob_itm": pick["prob_itm"],
        "chance_touch": pick["chance_touch"],
        "score": pick["score"],
        "current_price": pick["current_price"],
        "iv": pick["iv"],
        "realized_vol": pick["realized_vol"],
        "vol_ratio": pick["vol_ratio"],
        "bs_price": pick["bs_price"],
        "did_touch_itm": None,
        "did_expire_itm": None
    }
    save_predictions_history(hist)


###############################
# 11) ANALYZE MULTIPLE TICKERS FOR BOTH CALLS AND PUTS
###############################
def analyze_watchlist(tickers):
    """
    Analyze both the best OTM Call and Put for each ticker.
    Print metrics, log picks, and determine overall best pick based on Score.
    Optimized to reduce yFinance API calls by fetching data once per ticker.
    """
    forecast = get_market_vol_forecast()
    print(f"\nToday's Market Volatility Forecast: {forecast}\n")

    all_picks = []

    for ticker in tickers:
        print(f"Analyzing {ticker}...\n")
        try:
            # Fetch all necessary data once per ticker
            fetched_data = {}
            # Fetch intraday data
            fetched_data['hist_intraday'] = stock_history(ticker)
            if fetched_data['hist_intraday'].empty:
                print(f"[Ticker: {ticker}] No intraday data available.\n")
                continue
            # Fetch options
            fetched_data['options'] = get_ticker_expirations(ticker)
            if not fetched_data['options']:
                print(f"[Ticker: {ticker}] No options available.\n")
                continue
            # Fetch option chains
            fetched_data['option_chain'] = get_option_chains(ticker, fetched_data['options'])
            # Fetch short-dated rate
            fetched_data['short_dated_rate'] = get_short_dated_rate()
            # Fetch historical realized volatility
            fetched_data['realized_vol'] = get_historical_realized_vol(ticker, lookback_days=30)
            if fetched_data['realized_vol'] is None:
                print(f"[Ticker: {ticker}] Invalid realized volatility. Skipping.\n")
                continue

            for option_type in ["call", "put"]:
                best_pick, fetched_data = analyze_best_otm_option(ticker, option_type, fetched_data)
                if best_pick:
                    all_picks.append(best_pick)

                    # Print the pick
                    print(f"[Ticker: {best_pick['ticker']}] Best OTM {best_pick['option_type'].capitalize()}:")
                    print(f"  Expiration:               {best_pick['expiration']}")
                    print(f"  Strike:                   {best_pick['strike']:.2f}")
                    print(f"  LastPrice:                {best_pick['last_price']:.2f}")
                    print(f"  ProbabilityITM:           {best_pick['prob_itm']*100:.2f}%")
                    print(f"  Chance of Reaching ITM:   {best_pick['chance_touch']*100:.2f}%")
                    print(f"  Score:                    {best_pick['score']:.4f}")
                    if best_pick['vol_ratio'] is not None:
                        print(f"  ImpliedVol / RealizedVol: {best_pick['iv']:.2f} / {best_pick['realized_vol']:.2f} = {best_pick['vol_ratio']:.2f}")
                    else:
                        print(f"  ImpliedVol / RealizedVol: N/A")
                    print(f"  BS Fair Price:            {best_pick['bs_price']:.2f}")
                    print(f"  Current Price:            {best_pick['current_price']:.2f}\n")

                    # Log the pick
                    log_prediction(best_pick)
                else:
                    print(f"[Ticker: {ticker.upper()}] No suitable OTM {option_type.capitalize()} found.\n")
        except ZeroDivisionError as zde:
            print(f"ZeroDivisionError analyzing {ticker}: {zde}\n")
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}\n")

    if not all_picks:
        print("No valid OTM options found across the watchlist.\n")
        return

    # Determine the overall best pick based on highest Score
    all_picks.sort(key=lambda x: x["score"], reverse=True)
    top_pick = all_picks[0]

    print("=== Overall Best Pick from Watchlist ===")
    print(f"Ticker:                  {top_pick['ticker']}")
    print(f"Option Type:             {top_pick['option_type'].capitalize()}")
    print(f"Expiration:              {top_pick['expiration']}")
    print(f"Strike:                  {top_pick['strike']:.2f}")
    print(f"LastPrice:               {top_pick['last_price']:.2f}")
    print(f"ProbabilityITM:          {top_pick['prob_itm']*100:.2f}%")
    print(f"Chance of Reaching ITM:  {top_pick['chance_touch']*100:.2f}%")
    print(f"Score:                   {top_pick['score']:.4f}")
    print(f"ImpliedVol:              {top_pick['iv']:.2f}")
    if top_pick['realized_vol']:
        print(f"RealizedVol (30d):       {top_pick['realized_vol']*100:.2f}%")
    if top_pick['vol_ratio'] is not None:
        print(f"IV / RealizedVol:        {top_pick['vol_ratio']:.2f}")
    else:
        print(f"IV / RealizedVol:        N/A")
    print(f"BS Fair Price:           {top_pick['bs_price']:.2f}")
    print(f"Current Price:           {top_pick['current_price']:.2f}\n")


###############################
# 12) VIEW PREDICTIONS HISTORY
###############################
def view_predictions_history():
    hist = load_predictions_history()
    if not hist:
        print("\nNo Predictions History found.\n")
        return
    print("\n--- Predictions History ---")
    for key, val in hist.items():
        print(f"\nKey: {key}")
        print(f"  Queried On:              {val.get('date_queried', 'N/A')}")
        print(f"  Ticker:                  {val.get('ticker', 'N/A')}")
        print(f"  Option Type:             {val.get('option_type', 'N/A').capitalize()}")
        print(f"  Expiration:              {val.get('expiration', 'N/A')}")
        print(f"  Strike:                  {val.get('strike', 'N/A')}")
        print(f"  LastPrice:               {val.get('last_price', 'N/A')}")
        print(f"  ProbabilityITM:          {val.get('prob_itm', 0.0)*100:.2f}%")
        print(f"  Chance of Reaching ITM:  {val.get('chance_touch', 0.0)*100:.2f}%")
        print(f"  Score:                   {val.get('score', 0.0):.4f}")
        print(f"  Current Price:           {val.get('current_price', 'N/A'):.2f}")
        print(f"  ImpliedVol:              {val.get('iv', 'N/A'):.2f}")
        if val.get('realized_vol') is not None:
            print(f"  RealizedVol (30d):       {val.get('realized_vol')*100:.2f}%")
        if val.get('vol_ratio') is not None:
            print(f"  IV / RealizedVol:        {val.get('vol_ratio'):.2f}")
        else:
            print(f"  IV / RealizedVol:        N/A")
        print(f"  BS Fair Price:           {val.get('bs_price', 'N/A'):.2f}")
        print(f"  did_touch_itm:           {val.get('did_touch_itm', 'N/A')}")
        print(f"  did_expire_itm:          {val.get('did_expire_itm', 'N/A')}")
    print("")


###############################
# 13) MAIN MENU
###############################
def main_menu():
    while True:
        print("\n--- OTM Options Probability Tool ---")
        print("1. Analyze a list of stocks for Best OTM Call/Put")
        print("2. View Predictions History")
        print("3. Exit")
        choice = input("Please select an option (1-3): ")

        if choice == '1':
            tickers_str = input("Enter comma-separated stock tickers (e.g., AAPL,MSFT): ")
            tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
            if not tickers:
                print("No valid tickers entered. Please try again.")
                continue  # Optionally, prompt the user again
            analyze_watchlist(tickers)
        elif choice == '2':
            view_predictions_history()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


###############################
# 14) HELPER FUNCTIONS TO REDUCE yFINANCE API CALLS
###############################
def stock_history(ticker):
    """
    Fetch intraday history for the ticker.
    """
    stock = yf.Ticker(ticker)
    hist_intraday = stock.history(period="1d", interval="1m")
    return hist_intraday


def get_ticker_expirations(ticker):
    """
    Fetch available option expiration dates for the ticker.
    """
    stock = yf.Ticker(ticker)
    exps = stock.options
    return exps


def get_option_chains(ticker, expirations):
    """
    Fetch option chains for all expirations.
    Returns a dictionary with expiration dates as keys and option chains as values.
    """
    stock = yf.Ticker(ticker)
    option_chain_dict = {}
    for exp in expirations:
        option_chain = stock.option_chain(exp)
        option_chain_dict[exp] = option_chain
    return option_chain_dict


###############################
# 15) MAIN ENTRY
###############################
if __name__ == "__main__":
    main_menu()
