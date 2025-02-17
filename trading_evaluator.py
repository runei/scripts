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
from typing import Optional, Tuple

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
PROFIT_FACTOR = 5
EMA_WEEKLY_FAST = 20
EMA_WEEKLY_SLOW = 50
EMA_DAILY = 50

DOWNLOAD_NEW_DATA = False
CSV_NAME = (
    "all"  # Downloaded from https://www.nasdaq.com/market-activity/stocks/screener
)
CACHE_DIR = "cache"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MAX_RETRIES = 3
REQUEST_DELAY = 1.5
BATCH_SIZE = 10
CACHE_EXPIRATION_DAYS = 7

# Global variables to be populated in main()
daily_data_dict = {}
weekly_data_dict = {}
date_index = None
tickers = []
logger = logging.getLogger(__name__)


# -----------------------------
# Data Download and Utility Functions
# -----------------------------
def flatten_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten the columns of a DataFrame if they are a MultiIndex or tuple-based.
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
    """
    filename = os.path.join(CACHE_DIR, CSV_NAME + f"{interval}.csv")
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
    total_batches = len(batches)
    for idx, batch in enumerate(batches, start=1):
        logger.info(
            "Processing batch %d of %d: %d tickers", idx, total_batches, len(batch)
        )
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


# -----------------------------
# Strategy Classes
# -----------------------------
class BaseStrategy:
    """
    Base strategy interface. New strategies should inherit from this class and implement
    the generate_signals method.
    """

    def generate_signals(
        self, daily: pd.DataFrame, weekly: pd.DataFrame, date_index: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        raise NotImplementedError("Subclasses should implement this method.")


class DefaultStrategy(BaseStrategy):
    """
    Default trading strategy based on a combination of weekly and daily EMAs.
    """

    def __init__(
        self,
        ema_weekly_fast: int,
        ema_weekly_slow: int,
        ema_daily: int,
        risk_per_trade: float,
        initial_cash: float,
    ):
        self.ema_weekly_fast = ema_weekly_fast
        self.ema_weekly_slow = ema_weekly_slow
        self.ema_daily = ema_daily
        self.risk_per_trade = risk_per_trade
        self.initial_cash = initial_cash

    def generate_signals(
        self, daily: pd.DataFrame, weekly: pd.DataFrame, date_index: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        weekly["ema20"] = vbt.MA.run(
            weekly["Close"], window=self.ema_weekly_fast, ewm=True
        ).ma
        weekly["ema50"] = vbt.MA.run(
            weekly["Close"], window=self.ema_weekly_slow, ewm=True
        ).ma
        daily["ema50"] = vbt.MA.run(daily["Close"], window=self.ema_daily, ewm=True).ma

        weekly_daily = (
            weekly.resample("D")
            .ffill()[["ema20", "ema50"]]
            .rename(columns=lambda col: f"{col}_weekly")
        )

        aligned_data = daily.merge(
            weekly_daily, left_index=True, right_index=True, how="left"
        )
        aligned_data["stop_loss"] = (
            aligned_data["Low"].rolling(window=3, min_periods=1).min()
        )

        ticker_entries = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_exits = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_sizes = pd.Series(0.0, index=aligned_data.index, dtype=float)
        exit_prices = pd.Series(np.nan, index=aligned_data.index, dtype=float)

        in_position = False
        entry_price = 0.0
        stop_loss_val = 0.0

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
                position_size = (
                    self.initial_cash * self.risk_per_trade
                ) / risk_per_share
                ticker_entries.iloc[i] = True
                ticker_sizes.iloc[i] = position_size
                entry_price = close
                in_position = True

            if in_position:
                take_profit = entry_price + PROFIT_FACTOR * (
                    entry_price - stop_loss_val
                )
                if close <= stop_loss_val:
                    ticker_exits.iloc[i] = True
                    exit_prices.iloc[i] = stop_loss_val
                    in_position = False
                elif close >= take_profit:
                    ticker_exits.iloc[i] = True
                    exit_prices.iloc[i] = take_profit
                    in_position = False
                elif weekly_downtrend:
                    ticker_exits.iloc[i] = True
                    exit_prices.iloc[i] = close
                    in_position = False

        ticker_entries = ticker_entries.reindex(date_index, fill_value=False)
        ticker_exits = ticker_exits.reindex(date_index, fill_value=False)
        ticker_sizes = ticker_sizes.reindex(date_index, fill_value=0)
        exit_prices = exit_prices.reindex(date_index)
        return ticker_entries, ticker_exits, ticker_sizes, exit_prices


# -----------------------------
# Ticker Processing
# -----------------------------
def process_ticker(
    ticker: str, strategy: BaseStrategy
) -> Tuple[
    Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]
]:
    """
    Process a single ticker to generate trading signals using the provided strategy.
    """
    logger.info("Processing %s...", ticker)
    try:
        if ticker not in daily_data_dict or ticker not in weekly_data_dict:
            logger.warning("No data available for %s", ticker)
            return None, None, None, None

        daily = daily_data_dict.get(ticker, pd.DataFrame())
        weekly = weekly_data_dict.get(ticker, pd.DataFrame())

        return strategy.generate_signals(daily, weekly, date_index)
    except Exception as error:
        logger.error("Error processing %s: %s", ticker, error)
        return None, None, None, None


# -----------------------------
# Strategy Evaluation
# -----------------------------
def evaluate_strategy(portfolio):
    """
    Evaluate the performance of a trading strategy.
    """
    total_profit = portfolio.total_profit().sum()
    total_return = (total_profit / INITIAL_CASH) * 100

    trades = portfolio.trades.records_readable

    profitable_trades = trades["PnL"] > 0
    num_profitable_trades = profitable_trades.sum()
    total_closed_trades = trades["PnL"].count()
    win_rate = (
        (num_profitable_trades / total_closed_trades) * 100
        if total_closed_trades > 0
        else 0
    )

    total_gains = trades.loc[profitable_trades, "PnL"].sum()
    total_losses = trades.loc[~profitable_trades, "PnL"].sum()
    profit_factor = (
        total_gains / abs(total_losses) if total_losses != 0 else float("inf")
    )

    average_win = (
        trades.loc[profitable_trades, "PnL"].mean() if num_profitable_trades > 0 else 0
    )
    average_loss = (
        trades.loc[~profitable_trades, "PnL"].mean()
        if (total_closed_trades - num_profitable_trades) > 0
        else 0
    )
    risk_to_reward_ratio = (
        average_win / abs(average_loss) if average_loss != 0 else float("inf")
    )

    largest_win = trades["PnL"].max() if not trades.empty else 0
    largest_win_trade_id = trades["PnL"].idxmax() if not trades.empty else None
    largest_loss = trades["PnL"].min() if not trades.empty else 0
    largest_loss_trade_id = trades["PnL"].idxmin() if not trades.empty else None

    loss_rate = 1 - (win_rate / 100)
    expectancy = (win_rate / 100 * average_win) + (loss_rate * average_loss)
    max_drawdown = portfolio.drawdowns.max_drawdown().min()

    evaluation = [
        f"Total Profit/Loss (PnL): ${total_profit:.2f}",
        f"Total Return: {total_return:.2f}%",
        f"Profitable Trades: {num_profitable_trades}",
        f"Total Closed Trades: {total_closed_trades}",
        f"Win Rate: {win_rate:.2f}%",
        f"Total Gains: ${total_gains:.2f}",
        f"Total Losses: ${total_losses:.2f}",
        f"Profit Factor: {profit_factor:.2f}",
        f"Average Win: ${average_win:.2f}",
        f"Average Loss: ${average_loss:.2f}",
        f"Risk-to-Reward Ratio: {risk_to_reward_ratio:.2f}",
        f"Largest Single Win: ${largest_win:.2f} (Trade ID {largest_win_trade_id})",
        f"Largest Single Loss: ${largest_loss:.2f} (Trade ID {largest_loss_trade_id})",
        f"Expectancy: ${expectancy:.2f} per trade",
        f"Max Drawdown: {max_drawdown:.2f}%",
    ]

    return evaluation


# -----------------------------
# Main Execution
# -----------------------------
def main():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    logging.basicConfig(level=logging.INFO)
    global tickers, daily_data_dict, weekly_data_dict, date_index

    tickers_df = pd.read_csv(os.path.join("input", CSV_NAME + ".csv"))
    tickers = [t.replace("/", "-") for t in tickers_df["Symbol"].to_list()]

    daily_data_dict.clear()
    weekly_data_dict.clear()

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
            os.path.join(CACHE_DIR, CSV_NAME + "1d.csv"),
            index_col="Date",
            parse_dates=True,
        )
        weekly_all = pd.read_csv(
            os.path.join(CACHE_DIR, CSV_NAME + "1wk.csv"),
            index_col="Date",
            parse_dates=True,
        )

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

    date_index = pd.date_range(start=START_DATE, end=END_DATE)

    default_strategy = DefaultStrategy(
        ema_weekly_fast=EMA_WEEKLY_FAST,
        ema_weekly_slow=EMA_WEEKLY_SLOW,
        ema_daily=EMA_DAILY,
        risk_per_trade=RISK_PER_TRADE,
        initial_cash=INITIAL_CASH,
    )

    entries = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    exits = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    sizes = pd.DataFrame(0.0, index=date_index, columns=tickers, dtype=float)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker, default_strategy): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_item = future_to_ticker[future]
            try:
                t_entries, t_exits, t_sizes, t_exit_prices = future.result()
                if t_entries is not None:
                    results[ticker_item] = (t_entries, t_exits, t_sizes, t_exit_prices)
            except Exception as exc:
                logger.error("%s generated an exception: %s", ticker_item, exc)

    for ticker_item, (t_entries, t_exits, t_sizes, t_exit_prices) in results.items():
        entries[ticker_item] = t_entries
        exits[ticker_item] = t_exits
        sizes[ticker_item] = t_sizes

    # Build close_prices from daily data
    close_prices = pd.DataFrame(index=date_index, columns=tickers)
    for ticker_item in tickers:
        if ticker_item in daily_data_dict:
            close_prices[ticker_item] = daily_data_dict[ticker_item]["Close"]
        else:
            close_prices[ticker_item] = np.nan
    close_prices = close_prices.astype(np.float64)

    # Override exit prices in close_prices with custom exit prices from our strategy.
    # This way, when a trade is exited, the exit price will be the stop loss or profit target.
    for ticker_item, (_, t_exits, _, t_exit_prices) in results.items():
        exit_dates = t_exit_prices[t_exits].index
        for date in exit_dates:
            if not pd.isna(t_exit_prices.loc[date]):
                close_prices.loc[date, ticker_item] = t_exit_prices.loc[date]

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

    evaluation = evaluate_strategy(portfolio)
    for entry in evaluation:
        print(entry)

    trades_df = portfolio.trades.records_readable
    trades_df.to_csv("strategy_trades.csv", index=False)


if __name__ == "__main__":
    main()
