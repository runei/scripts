#!/usr/bin/env python3
import os
import time
import hashlib
import pandas as pd
import numpy as np
import vectorbt as vbt
import yfinance as yf
from datetime import datetime
import concurrent.futures
import logging
import requests
from random import uniform

# -----------------------------
# Configuration
# -----------------------------
RISK_PER_TRADE = 0.01  # 1% of portfolio per trade
INITIAL_CASH = 10000
EMA_WEEKLY_FAST = 20
EMA_WEEKLY_SLOW = 50
EMA_DAILY = 50

# Set this flag to True to download all tickers at once; set to False to use individual cached files.
DOWNLOAD_NEW_DATA = False
CSV_NAME = "mega.csv"
CACHE_DIR = "cache"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MAX_RETRIES = 3  # Maximum retries for failed downloads
REQUEST_DELAY = 1.5  # Seconds between requests
BATCH_SIZE = 10  # Number of tickers per download batch
CACHE_EXPIRATION_DAYS = 7  # Refresh cached data weekly


# -----------------------------
# Create cache directory if needed
# -----------------------------
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Read tickers from CSV
# -----------------------------
tickers_df = pd.read_csv("input/" + CSV_NAME)
tickers = [
    t.replace("/", "-") for t in tickers_df["Symbol"].to_list()
]  # Fix Yahoo ticker format


# -----------------------------
# Helper: Flatten columns if necessary
# -----------------------------
def flatten_columns(df):
    # Check if columns are a MultiIndex or tuple-based; if so, flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
    elif isinstance(df.columns[0], tuple):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns]
    return df


# -----------------------------
# Modified Download Functions with Throttling and Retry
# -----------------------------
def download_data_with_retry(tickers, interval, session, start, end, retries=0):
    """Download data with retries and throttling"""
    try:
        logger.info(
            f"Downloading {len(tickers)} {interval} tickers (attempt {retries + 1})"
        )
        df = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            session=session,
            group_by="ticker",
            progress=False,
        )
        time.sleep(uniform(REQUEST_DELAY, REQUEST_DELAY * 2))  # Randomized delay
        return df
    except Exception as e:
        if retries < MAX_RETRIES:
            wait_time = REQUEST_DELAY * (2**retries)
            logger.warning(f"Retrying in {wait_time}s... ({str(e)})")
            time.sleep(wait_time)
            return download_data_with_retry(
                tickers, interval, session, start, end, retries + 1
            )
        else:
            logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
            return pd.DataFrame()


def download_data_all(interval, session, tickers, start=START_DATE, end=END_DATE):
    """Download data in batches with rate limiting"""
    # Create a reproducible filename based on the sorted ticker list and interval.

    filename = os.path.join(CACHE_DIR, f"{interval}.csv")

    # Check if cached data is still valid
    if os.path.exists(filename):
        file_age = (
            datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))
        ).days
        if file_age < CACHE_EXPIRATION_DAYS:
            logger.info(f"Using cached {interval} data (age: {file_age} days)")
            df = pd.read_csv(filename, index_col="Date", parse_dates=True)
            return flatten_columns(df)
    # Split tickers into batches
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_data = []
    for batch in batches:
        logger.info(f"Processing batch of {len(batch)} tickers")
        batch_data = download_data_with_retry(batch, interval, session, start, end)
        if not batch_data.empty:
            batch_data = flatten_columns(batch_data)
            all_data.append(batch_data)
        time.sleep(REQUEST_DELAY)
    if all_data:
        combined_data = pd.concat(all_data, axis=1)
        combined_data.to_csv(filename)
        return combined_data
    return pd.DataFrame()


# -----------------------------
# Modified Session Configuration
# -----------------------------
daily_data_dict = {}
weekly_data_dict = {}

if DOWNLOAD_NEW_DATA:
    session = requests.Session()
    # Set custom headers to mimic a browser
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    )
    adapter = requests.adapters.HTTPAdapter(
        max_retries=3, pool_connections=20, pool_maxsize=20
    )
    session.mount("https://", adapter)
    # Download data with throttling
    daily_all = download_data_all("1d", session, tickers)
    weekly_all = download_data_all("1wk", session, tickers)

else:
    logger.info("Loading cached combined data")

    # Load daily data from cache
    daily_all = pd.read_csv(
        os.path.join(CACHE_DIR, f"1d.csv"), index_col="Date", parse_dates=True
    )

    # Load weekly data from cache
    weekly_all = pd.read_csv(
        os.path.join(CACHE_DIR, f"1wk.csv"), index_col="Date", parse_dates=True
    )

for ticker in tickers:
    daily_cols = [col for col in daily_all.columns if col.startswith(f"{ticker}_")]
    if daily_cols:
        daily_data_dict[ticker] = daily_all[daily_cols].rename(
            columns=lambda x: x.replace(f"{ticker}_", "")
        )

    weekly_cols = [col for col in weekly_all.columns if col.startswith(f"{ticker}_")]
    if weekly_cols:
        weekly_data_dict[ticker] = weekly_all[weekly_cols].rename(
            columns=lambda x: x.replace(f"{ticker}_", "")
        )

# Validate data existence
missing_tickers = [t for t in tickers if t not in daily_data_dict]
if missing_tickers:
    logger.warning(f"Missing data for tickers: {missing_tickers}")

# -----------------------------
# Prepare Global DataFrames for Signals
# -----------------------------
date_index = pd.date_range(start=START_DATE, end=END_DATE)
entries = pd.DataFrame(False, index=date_index, columns=tickers)
exits = pd.DataFrame(False, index=date_index, columns=tickers)
sizes = pd.DataFrame(0.0, index=date_index, columns=tickers)


# -----------------------------
# Process a Single Ticker
# -----------------------------
def process_ticker(ticker):
    logger.info(f"Processing {ticker}...")
    try:
        if ticker not in daily_data_dict or ticker not in weekly_data_dict:
            logger.warning(f"No data available for {ticker}")
            return None, None, None

        daily = daily_data_dict.get(ticker, pd.DataFrame())
        weekly = weekly_data_dict.get(ticker, pd.DataFrame())

        # Calculate EMAs using vectorbt's MA indicator with ewm=True
        weekly["ema20"] = vbt.MA.run(
            weekly["Close"], window=EMA_WEEKLY_FAST, ewm=True
        ).ma
        weekly["ema50"] = vbt.MA.run(
            weekly["Close"], window=EMA_WEEKLY_SLOW, ewm=True
        ).ma
        daily["ema50"] = vbt.MA.run(daily["Close"], window=EMA_DAILY, ewm=True).ma

        # Resample weekly EMA data to daily frequency and add a "_weekly" suffix
        weekly_daily = (
            weekly.resample("D")
            .ffill()[["ema20", "ema50"]]
            .rename(columns=lambda col: col + "_weekly")
        )

        # Join daily data with the resampled weekly EMAs
        aligned_data = daily.merge(
            weekly_daily, left_index=True, right_index=True, how="left"
        )
        aligned_data["stop_loss"] = (
            aligned_data["Low"].rolling(window=3, min_periods=1).min()
        )

        ticker_entries = pd.Series(False, index=aligned_data.index)
        ticker_exits = pd.Series(False, index=aligned_data.index)
        ticker_sizes = pd.Series(0.0, index=aligned_data.index)

        in_position = False
        entry_price = 0
        stop_loss_val = 0

        # Loop over the aligned data starting from the third row
        for i in range(2, len(aligned_data)):
            ema20 = aligned_data["ema20_weekly"].iloc[i]
            ema50 = aligned_data["ema50_weekly"].iloc[i]
            close = aligned_data["Close"].iloc[i]
            low = aligned_data["Low"].iloc[i]
            high = aligned_data["High"].iloc[i]
            prev_low = aligned_data["Low"].iloc[i - 1]
            prev_high = aligned_data["High"].iloc[i - 1]

            weekly_uptrend = (ema20 > ema50) and (close > ema20)
            weekly_downtrend = (ema20 < ema50) and (close < ema20)
            bullish_reversal = (low < prev_low) and (close > prev_high)
            above_ema50_daily = close > aligned_data["ema50"].iloc[i]

            if (
                weekly_uptrend
                and (not in_position)
                and bullish_reversal
                and above_ema50_daily
            ):
                stop_loss_val = aligned_data["stop_loss"].iloc[i]
                risk_per_share = close - stop_loss_val
                if risk_per_share == 0:
                    continue
                position_size = (INITIAL_CASH * RISK_PER_TRADE) / risk_per_share

                ticker_entries.iloc[i] = True
                ticker_sizes.iloc[i] = position_size
                entry_price = close
                in_position = True

            if in_position:
                take_profit = entry_price + 3 * (entry_price - stop_loss_val)
                if (
                    (close <= stop_loss_val)
                    or (close >= take_profit)
                    or weekly_downtrend
                ):
                    ticker_exits.iloc[i] = True
                    in_position = False

        ticker_entries = ticker_entries.reindex(date_index, fill_value=False)
        ticker_exits = ticker_exits.reindex(date_index, fill_value=False)
        ticker_sizes = ticker_sizes.reindex(date_index, fill_value=0)

        return ticker_entries, ticker_exits, ticker_sizes

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None, None, None


# -----------------------------
# Process All Tickers in Parallel
# -----------------------------
results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_ticker = {
        executor.submit(process_ticker, ticker): ticker for ticker in tickers
    }
    for future in concurrent.futures.as_completed(future_to_ticker):
        ticker = future_to_ticker[future]
        try:
            t_entries, t_exits, t_sizes = future.result()
            if t_entries is not None:
                results[ticker] = (t_entries, t_exits, t_sizes)
        except Exception as exc:
            logger.error(f"{ticker} generated an exception: {exc}")

for ticker, (t_entries, t_exits, t_sizes) in results.items():
    entries[ticker] = t_entries
    exits[ticker] = t_exits
    sizes[ticker] = t_sizes

close_prices = pd.DataFrame(index=date_index, columns=tickers)
for ticker in tickers:
    if ticker in daily_data_dict:
        close_prices[ticker] = daily_data_dict[ticker]["Close"]
    else:
        # Optional: Fill missing data (or handle it as you prefer)
        close_prices[ticker] = np.nan

# Make sure the data type is float64 (or another precise numeric type)
close_prices = close_prices.astype(np.float64)

# -----------------------------
# Create Portfolio with vectorbt
# -----------------------------
pf = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=entries,
    exits=exits,
    size=sizes,
    init_cash=INITIAL_CASH,
    fees=0.000,
    slippage=0.000,
    upon_opposite_entry="close",
)

# -----------------------------
# Output Performance Results
# -----------------------------
total_return = float(pf.total_return().mean() * 100)
sharpe_ratio = float(pf.sharpe_ratio().mean())
max_drawdown = float(pf.max_drawdown().mean() * 100)
logger.info(f"Total Return: {total_return:.2f}%")
logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
# pf.plot().show()

# -----------------------------
# Save Trades to CSV
# -----------------------------
trades = pf.trades.records_readable
trades.to_csv("strategy_trades.csv", index=False)
