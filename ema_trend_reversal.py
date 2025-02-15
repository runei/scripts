#!/venv/bin/python
"""
This script downloads historical stock data, processes trading signals based on technical analysis,
and evaluates a trading strategy using the vectorbt library.
"""

import concurrent.futures
import logging
import os
import time
from datetime import datetime
from random import uniform
from typing import Optional

import numpy as np
import pandas as pd
import requests
import vectorbt as vbt
import yfinance as yf
from requests.exceptions import RequestException

# -----------------------------
# Configuration
# -----------------------------
RISK_PER_TRADE = 0.01  # 1% of portfolio per trade
INITIAL_CASH = 10000
EMA_WEEKLY_FAST = 20
EMA_WEEKLY_SLOW = 50
EMA_DAILY = 50

# Set this flag to True to download all tickers; set to False to use individual cached files.
DOWNLOAD_NEW_DATA = False
CSV_NAME = (
    "mega.csv"  # Downloaded from https://www.nasdaq.com/market-activity/stocks/screener
)
CACHE_DIR = "cache"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MAX_RETRIES = 3
REQUEST_DELAY = 1.5
BATCH_SIZE = 10
CACHE_EXPIRATION_DAYS = 7


def flatten_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten the columns of a DataFrame if they are a MultiIndex or tuple-based.

    Parameters:
        df (Optional[pd.DataFrame]): The DataFrame whose columns may be a MultiIndex or
            contain tuples.

    Returns:
        pd.DataFrame: The DataFrame with flattened column names.
        Returns an empty DataFrame if df is None.
    """
    if df is None:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
    elif isinstance(df.columns[0], tuple):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns]
    return df


def download_data_with_retry(ticker_list, interval, session, start, end, retries=0):
    """
    Download data for a list of tickers with retries and throttling.

    Parameters:
        ticker_list (list): List of ticker symbols.
        interval (str): Data interval (e.g., '1d', '1wk').
        session (requests.Session): The session object to use for downloading.
        start (str): Start date for data.
        end (str): End date for data.
        retries (int): Current number of retries.

    Returns:
        pd.DataFrame: DataFrame containing the downloaded data.
    """
    try:
        logger.info(
            "Downloading %s %s tickers (attempt %d)",
            len(ticker_list),
            interval,
            retries + 1,
        )
        df = yf.download(
            ticker_list,
            start=start,
            end=end,
            interval=interval,
            session=session,
            group_by="ticker",
            progress=False,
        )
        time.sleep(uniform(REQUEST_DELAY, REQUEST_DELAY * 2))  # Randomized delay
        return df
    except (RequestException, ValueError) as error:
        if retries < MAX_RETRIES:
            wait_time = REQUEST_DELAY * (2**retries)
            logger.warning("Retrying in %s s... (%s)", wait_time, error)
            time.sleep(wait_time)
            return download_data_with_retry(
                ticker_list, interval, session, start, end, retries + 1
            )
        else:
            logger.error("Failed after %d attempts: %s", MAX_RETRIES, error)
            return pd.DataFrame()


def download_data_all(interval, session, ticker_list, start=START_DATE, end=END_DATE):
    """
    Download data for all tickers in batches with rate limiting, using caching.

    Parameters:
        interval (str): Data interval (e.g., '1d', '1wk').
        session (requests.Session): The session object to use for downloading.
        ticker_list (list): List of ticker symbols.
        start (str): Start date for data.
        end (str): End date for data.

    Returns:
        pd.DataFrame: Combined DataFrame of all downloaded data.
    """
    filename = os.path.join(CACHE_DIR, f"{interval}.csv")
    if os.path.exists(filename):
        file_age = (
            datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))
        ).days
        if file_age < CACHE_EXPIRATION_DAYS:
            logger.info("Using cached %s data (age: %s days)", interval, file_age)
            df = pd.read_csv(filename, index_col="Date", parse_dates=True)
            return flatten_columns(df)
    batches = [
        ticker_list[i : i + BATCH_SIZE] for i in range(0, len(ticker_list), BATCH_SIZE)
    ]
    all_data = []
    for batch in batches:
        logger.info("Processing batch of %s tickers", len(batch))
        batch_data = download_data_with_retry(batch, interval, session, start, end)
        if batch_data is not None and not batch_data.empty:
            batch_data = flatten_columns(batch_data)
            all_data.append(batch_data)
        time.sleep(REQUEST_DELAY)
    if all_data:
        combined_data = pd.concat(all_data, axis=1)
        combined_data.to_csv(filename)
        return combined_data
    return pd.DataFrame()


def process_ticker(ticker):
    """
    Process a single ticker to generate trading signals and position sizes.

    Parameters:
        ticker (str): The ticker symbol to process.

    Returns:
        tuple: A tuple of three pandas Series (ticker_entries, ticker_exits, ticker_sizes) for
        trading entries, exits, and position sizes, respectively.
        Returns (None, None, None) on failure.
    """
    logger.info("Processing %s...", ticker)
    try:
        if ticker not in daily_data_dict or ticker not in weekly_data_dict:
            logger.warning("No data available for %s", ticker)
            return None, None, None

        daily = daily_data_dict.get(ticker, pd.DataFrame())
        weekly = weekly_data_dict.get(ticker, pd.DataFrame())

        # Calculate EMAs using vectorbt's MA indicator with exponential weighting
        weekly["ema20"] = vbt.MA.run(
            weekly["Close"], window=EMA_WEEKLY_FAST, ewm=True
        ).ma
        weekly["ema50"] = vbt.MA.run(
            weekly["Close"], window=EMA_WEEKLY_SLOW, ewm=True
        ).ma
        daily["ema50"] = vbt.MA.run(daily["Close"], window=EMA_DAILY, ewm=True).ma

        # Resample weekly EMA data to daily frequency and rename columns
        weekly_daily = (
            weekly.resample("D")
            .ffill()[["ema20", "ema50"]]
            .rename(columns=lambda col: f"{col}_weekly")
        )

        # Join daily data with the resampled weekly EMAs
        aligned_data = daily.merge(
            weekly_daily, left_index=True, right_index=True, how="left"
        )
        aligned_data["stop_loss"] = (
            aligned_data["Low"].rolling(window=3, min_periods=1).min()
        )

        ticker_entries = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_exits = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_sizes = pd.Series(0.0, index=aligned_data.index, dtype=float)

        in_position = False
        entry_price = 0.0
        stop_loss_val = 0.0

        # Loop over the aligned data starting from the third row
        for i in range(2, len(aligned_data)):
            ema20 = aligned_data["ema20_weekly"].iloc[i]
            ema50 = aligned_data["ema50_weekly"].iloc[i]
            close = aligned_data["Close"].iloc[i]
            low = aligned_data["Low"].iloc[i]
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

    except (KeyError, IndexError, ValueError) as error:
        logger.error("Error processing %s: %s", ticker, error)
        return None, None, None


def evaluate_strategy(portfolio, benchmark_ticker="SPY"):
    """
    Evaluate the performance of a trading strategy using portfolio statistics and benchmark comparison.

    Parameters:
        portfolio (vbt.Portfolio): The portfolio object containing trading signals and performance data.
        benchmark_ticker (str): The ticker symbol for the benchmark (default is 'SPY').

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    trade_stats = portfolio.trades.stats()
    benchmark = yf.download(benchmark_ticker, start=START_DATE, end=END_DATE)["Close"]
    benchmark_rets = benchmark.pct_change().dropna()

    total_return = portfolio.total_return().mean() * 100
    annualized_return = (
        portfolio.annualized_return().mean() * 100
        if not portfolio.trades.records_readable.empty
        else 0
    )
    max_dd = portfolio.max_drawdown().mean() * 100

    bench_return = (
        (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100 if len(benchmark) > 0 else 0
    )
    bench_vol = (
        benchmark_rets.std() * (252**0.5) * 100 if len(benchmark_rets) > 0 else 0
    )

    trades = portfolio.trades.records_readable
    win_rate = (trades["Status"] == "Win").mean() * 100 if not trades.empty else 0

    evaluation = {
        "Total Return (%)": round(total_return, 2),
        "Annualized Return (%)": round(annualized_return, 2),
        "Benchmark Return (%)": round(bench_return, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Benchmark Volatility (%)": round(bench_vol, 2),
        "Total Trades": len(trades),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": (
            round(trade_stats.get("Profit Factor", np.nan), 2)
            if not trade_stats.empty
            else np.nan
        ),
        "Avg Win Trade (%)": (
            round(trade_stats.get("Avg Winning Trade [%]", np.nan), 2)
            if not trade_stats.empty
            else np.nan
        ),
        "Avg Loss Trade (%)": (
            round(trade_stats.get("Avg Losing Trade [%]", np.nan), 2)
            if not trade_stats.empty
            else np.nan
        ),
    }

    logger.info("\nStrategy Evaluation:")
    for metric, value in evaluation.items():
        logger.info("%s %s", metric.ljust(30, "."), value)

    if not portfolio.trades.records_readable.empty:
        fig = portfolio.plot(
            subplots=[
                ("cum_returns", {"title": "Cumulative Returns"}),
                ("drawdowns", {"title": "Drawdowns"}),
            ]
        )
        fig.show()

    return evaluation


def main():
    # Create cache directory if needed
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    global logger, tickers, daily_data_dict, weekly_data_dict, date_index
    logger = logging.getLogger(__name__)

    # Read tickers from CSV
    tickers_df = pd.read_csv(os.path.join("input", CSV_NAME))
    tickers = [t.replace("/", "-") for t in tickers_df["Symbol"].to_list()]

    # Prepare data dictionaries
    daily_data_dict = {}
    weekly_data_dict = {}

    if DOWNLOAD_NEW_DATA:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3, pool_connections=20, pool_maxsize=20
        )
        session.mount("https://", adapter)
        daily_all = download_data_all("1d", session, tickers)
        weekly_all = download_data_all("1wk", session, tickers)
    else:
        logger.info("Loading cached combined data")
        daily_all = pd.read_csv(
            os.path.join(CACHE_DIR, "1d.csv"), index_col="Date", parse_dates=True
        )
        weekly_all = pd.read_csv(
            os.path.join(CACHE_DIR, "1wk.csv"), index_col="Date", parse_dates=True
        )

    # Populate the data dictionaries
    for ticker in tickers:
        daily_cols = [col for col in daily_all.columns if col.startswith(f"{ticker}_")]
        if daily_cols:
            daily_data_dict[ticker] = daily_all[daily_cols].rename(
                columns=lambda x: x.replace(f"{ticker}_", "")
            )
        weekly_cols = [
            col for col in weekly_all.columns if col.startswith(f"{ticker}_")
        ]
        if weekly_cols:
            weekly_data_dict[ticker] = weekly_all[weekly_cols].rename(
                columns=lambda x: x.replace(f"{ticker}_", "")
            )

    missing_tickers = [t for t in tickers if t not in daily_data_dict]
    if missing_tickers:
        logger.warning("Missing data for tickers: %s", missing_tickers)

    # Prepare global dataframes for signals
    date_index = pd.date_range(start=START_DATE, end=END_DATE)
    entries = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    exits = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    sizes = pd.DataFrame(0.0, index=date_index, columns=tickers, dtype=float)

    # Process tickers in parallel
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker): ticker for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_item = future_to_ticker[future]
            try:
                t_entries, t_exits, t_sizes = future.result()
                if t_entries is not None:
                    results[ticker_item] = (t_entries, t_exits, t_sizes)
            except (KeyError, IndexError, ValueError, RequestException) as exc:
                logger.error("%s generated an exception: %s", ticker_item, exc)

    for ticker_item, (t_entries, t_exits, t_sizes) in results.items():
        entries[ticker_item] = t_entries
        exits[ticker_item] = t_exits
        sizes[ticker_item] = t_sizes

    close_prices = pd.DataFrame(index=date_index, columns=tickers)
    for ticker_item in tickers:
        if ticker_item in daily_data_dict:
            close_prices[ticker_item] = daily_data_dict[ticker_item]["Close"]
        else:
            close_prices[ticker_item] = np.nan
    close_prices = close_prices.astype(np.float64)

    # Create portfolio using vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        size=sizes,
        init_cash=INITIAL_CASH,
        fees=0.000,
        slippage=0.000,
        upon_opposite_entry="close",
    )

    # Evaluate strategy and save trades
    evaluate_strategy(portfolio)
    trades_df = portfolio.trades.records_readable
    trades_df.to_csv("strategy_trades.csv", index=False)


if __name__ == "__main__":
    main()
