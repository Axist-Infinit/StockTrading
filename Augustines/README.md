## rtscli - Realtime Stock Ticker CLI
<a target="_blank" href="https://opensource.org/licenses/MIT" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a> <a target="_blank" href="http://makeapullrequest.com" title="PRs Welcome"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>

- A stock ticker that runs in console
- It grabs info from Google Finance Api every 10 seconds, or if you press R

## Screenshot

![Demo](https://github.com/aranair/rtscli/blob/master/rtscli-demo.png?raw=true "Demo")

## Dependencies

Currently this is dependent on the list below but the next step is to build this into an executable so
all that stuff with python and pip can be skipped.

- Python2.7
- pip
- Bunch of other python packages

## Install via Pip

```
pip install rtscli
```

## Running it

```bash
$ cp tickers.txt.sample tickers.txt
$ rtscli
```

## Tickers.txt Sample

Format: Name, Ticker(Alphavantage format), cost price, shares held

```
GLD,GLD,139,1
```

## Downloading and building manually

```
$ git clone git@github.com:aranair/rtscli.git
$ pip install -e .
$ rtscli
```

## Swing Trading Strategy Explained
```
The strategy implemented is designed for swing trading, aiming to profit from short- to medium-term price movements over 2 days to 2 weeks, targeting 6% to 12% gains (or greater). It utilizes technical analysis to identify stocks in an uptrend with potential for continued upward movement.

Enhanced Criteria for Buy Zone
A stock is considered to be in the buy zone and receives a buy recommendation if it meets all of the following enhanced criteria:

Uptrend Confirmation

Exponential Moving Averages (EMAs):

The 7-day EMA is above the 21-day EMA.
The 21-day EMA is above the 50-day EMA.
Multiple Time Frame Analysis:

On the weekly chart, the 7-day EMA is above the 21-day EMA.
This ensures the stock is in an uptrend across both daily and weekly time frames.

Price Positioning

The current price is above the 7-day EMA.
Momentum Indicators

Relative Strength Index (RSI):

RSI is below 70 to avoid overbought conditions.
MACD (Moving Average Convergence Divergence):

MACD line is above the MACD signal line, indicating bullish momentum.
Stochastic Oscillator:

%K line is below 80, further ensuring the stock is not overbought.
Volume Confirmation

Current Volume is above the 20-day average volume, indicating strong buying interest.
Earnings Filter

No earnings announcements scheduled within the next 2 weeks to avoid volatility associated with earnings releases.
Volatility-Based Risk Management

Average True Range (ATR):

Used to set stop loss and profit target levels based on the stock's recent volatility.
Why These Enhancements?

ATR for Dynamic Risk Management:

Adapts stop loss and profit target to current market volatility, aiming for 6% to 12% gains.
Volume Confirmation:

High volume on price increases confirms strong buying pressure, improving trade success probability.
MACD and Stochastic Indicators:

Provide additional confirmation of bullish momentum and help avoid entering trades during overbought conditions.
Earnings Filter:

Reduces the risk of unexpected volatility due to earnings announcements.
Multiple Time Frame Analysis:

Aligns the trend on daily and weekly charts, strengthening the trade setup.
By incorporating these enhancements, the strategy aims to improve the rate of profitable trades while maintaining the desired holding period and profit targets.

Understanding the Output
For each stock, the script outputs:

Current Price

Buy Recommendation (if applicable):

Buy at: Exact price point to enter the trade.
Stop Loss: Price level to set a stop-loss order (1 ATR below entry).
Profit Target: Price level to take profits, maintaining a 2:1 reward-to-risk ratio.
Expected Gain: Percentage gain if the profit target is reached.
No Buy Signal: If the stock doesn't meet the criteria, it states that there's no buy signal at this time.
```
