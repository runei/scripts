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
from typing import Dict, List, Optional, Tuple

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
PROFIT_FACTOR = 3

DOWNLOAD_NEW_DATA = True
CSV_NAME = "mega"  # e.g., downloaded from https://www.nasdaq.com/market-activity/stocks/screener
CACHE_DIR = "cache"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MAX_RETRIES = 3
REQUEST_DELAY = 1.5
BATCH_SIZE = 10

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Utility Functions
# -----------------------------
def get_session() -> requests.Session:
    """Returns a configured requests Session with retry and header settings."""
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
    return session


def flatten_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten the columns of a DataFrame if they are a MultiIndex or tuple-based.
    """
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex) or (
        len(df.columns) > 0 and isinstance(df.columns[0], tuple)
    ):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    return df


def download_data_with_retry(
    ticker_list: List[str],
    interval: str,
    session: requests.Session,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download data for a list of tickers with retries and throttling.
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"Downloading {len(ticker_list)} {interval} tickers (attempt {attempt + 1})"
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
            time.sleep(uniform(REQUEST_DELAY, REQUEST_DELAY * 2))
            return df
        except (RequestException, ValueError) as error:
            wait_time = REQUEST_DELAY * (2**attempt)
            logger.warning(f"Error: {error}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    logger.error(
        f"Failed to download data for tickers: {ticker_list} after {MAX_RETRIES} attempts."
    )
    return pd.DataFrame()


def download_data_all(
    interval: str,
    session: requests.Session,
    ticker_list: List[str],
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Download data for all tickers in batches with rate limiting, using a caching mechanism.
    """
    filename = os.path.join(CACHE_DIR, f"{CSV_NAME}_{interval}.csv")

    batches = [
        ticker_list[i : i + BATCH_SIZE] for i in range(0, len(ticker_list), BATCH_SIZE)
    ]
    all_data = []
    total_batches = len(batches)
    for idx, batch in enumerate(batches, start=1):
        logger.info(f"Processing batch {idx} of {total_batches} ({len(batch)} tickers)")
        batch_data = download_data_with_retry(batch, interval, session, start, end)
        if not batch_data.empty:
            batch_data = flatten_columns(batch_data)
            all_data.append(batch_data)
        time.sleep(REQUEST_DELAY)
    if all_data:
        combined_data = pd.concat(all_data, axis=1)
        os.makedirs(CACHE_DIR, exist_ok=True)
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
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        raise NotImplementedError("Subclasses should implement this method.")


class TrendMomentumVolumeATRStrategy(BaseStrategy):
    """
    Enhanced trading strategy incorporating RSI, ATR for dynamic stops, and volume filters.
    """

    def __init__(
        self,
        ema_weekly_fast: int = 20,
        ema_weekly_slow: int = 50,
        ema_daily: int = 50,
        risk_per_trade: float = RISK_PER_TRADE,
        initial_cash: float = INITIAL_CASH,
        rsi_window: int = 14,
        atr_window: int = 14,
        volume_ma_window: int = 20,
    ):
        self.ema_weekly_fast = ema_weekly_fast
        self.ema_weekly_slow = ema_weekly_slow
        self.ema_daily = ema_daily
        self.risk_per_trade = risk_per_trade
        self.initial_cash = initial_cash
        self.rsi_window = rsi_window
        self.atr_window = atr_window
        self.volume_ma_window = volume_ma_window

    def generate_signals(
        self, daily: pd.DataFrame, weekly: pd.DataFrame, date_index: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        # Calculate weekly EMAs
        weekly["ema20"] = vbt.MA.run(
            weekly["Close"], window=self.ema_weekly_fast, ewm=True
        ).ma
        weekly["ema50"] = vbt.MA.run(
            weekly["Close"], window=self.ema_weekly_slow, ewm=True
        ).ma

        # Calculate daily indicators
        daily["ema50"] = vbt.MA.run(daily["Close"], window=self.ema_daily, ewm=True).ma
        daily["rsi"] = vbt.RSI.run(daily["Close"], window=self.rsi_window).rsi
        daily["atr"] = vbt.ATR.run(
            daily["High"], daily["Low"], daily["Close"], window=self.atr_window
        ).atr
        daily["volume_ma"] = (
            daily["Volume"].rolling(window=self.volume_ma_window).mean()
        )

        # Align weekly data to daily frequency
        weekly_daily = weekly.resample("D").ffill()[["ema20", "ema50"]]
        weekly_daily = weekly_daily.rename(columns=lambda col: f"{col}_weekly")
        aligned_data = daily.merge(
            weekly_daily, left_index=True, right_index=True, how="left"
        )

        # Initialize signal series
        ticker_entries = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_exits = pd.Series(False, index=aligned_data.index, dtype=bool)
        ticker_sizes = pd.Series(0.0, index=aligned_data.index, dtype=float)
        exit_prices = pd.Series(np.nan, index=aligned_data.index, dtype=float)

        in_position = False
        entry_price = 0.0
        initial_atr = 0.0
        highest_high = 0.0
        take_profit = 0.0

        for i in range(2, len(aligned_data)):
            current = aligned_data.iloc[i]
            prev = aligned_data.iloc[i - 1]
            prev_prev = aligned_data.iloc[i - 2]

            # Trend conditions
            weekly_uptrend = (current["ema20_weekly"] > current["ema50_weekly"]) and (
                current["Close"] > current["ema20_weekly"]
            )
            weekly_downtrend = (current["ema20_weekly"] < current["ema50_weekly"]) and (
                current["Close"] < current["ema20_weekly"]
            )

            # Reversal pattern: bullish reversal defined as lower low followed by breakout
            bullish_reversal = (prev_prev["Low"] > prev["Low"]) and (
                current["Close"] > prev["High"]
            )

            # Indicator conditions
            above_ema50 = current["Close"] > current["ema50"]
            rsi_ok = 50 < current["rsi"] < 70  # avoid overbought conditions
            volume_ok = current["Volume"] > current["volume_ma"]

            # Entry logic
            if (
                not in_position
                and weekly_uptrend
                and bullish_reversal
                and above_ema50
                and rsi_ok
                and volume_ok
            ):
                risk_per_share = 2 * current["atr"]
                if risk_per_share == 0:
                    continue
                position_size = (
                    self.initial_cash * self.risk_per_trade
                ) / risk_per_share
                entry_price = current["Close"]
                initial_atr = current["atr"]
                highest_high = current["High"]
                take_profit = (
                    entry_price + 3 * risk_per_share
                )  # 3:1 reward-to-risk ratio

                ticker_entries.iloc[i] = True
                ticker_sizes.iloc[i] = position_size
                in_position = True

            # Exit logic
            if in_position:
                highest_high = max(highest_high, current["High"])
                trailing_stop = highest_high - 1.5 * initial_atr
                if (
                    current["Close"] <= trailing_stop
                    or current["Close"] >= take_profit
                    or weekly_downtrend
                ):
                    exit_price = (
                        trailing_stop
                        if current["Close"] <= trailing_stop
                        else take_profit
                    )
                    ticker_exits.iloc[i] = True
                    exit_prices.iloc[i] = exit_price
                    in_position = False

        # Reindex signals to match the full date index
        ticker_entries = ticker_entries.reindex(date_index, fill_value=False)
        ticker_exits = ticker_exits.reindex(date_index, fill_value=False)
        ticker_sizes = ticker_sizes.reindex(date_index, fill_value=0)
        exit_prices = exit_prices.reindex(date_index)

        return ticker_entries, ticker_exits, ticker_sizes, exit_prices

    def filter_weekly(self) -> List[str]:
        """
        Downloads weekly data for tickers in CSV_NAME, filters tickers satisfying weekly conditions,
        and outputs a CSV with qualifying tickers in the 'out/' folder. Only downloads data from the past 10 years.
        """
        from datetime import datetime, timedelta

        start_date = (datetime.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs("out", exist_ok=True)

        input_csv = os.path.join("input", f"{CSV_NAME}.csv")
        tickers = load_tickers(input_csv)
        logger.info(f"Loaded {len(tickers)} tickers from {input_csv}")

        session = get_session()
        weekly_all = download_data_all("1wk", session, tickers, start=start_date)
        data = prepare_data_dicts(weekly_all, tickers)

        def ticker_qualifies(ticker: str) -> Optional[str]:
            if ticker not in data:
                logger.warning(f"No weekly data available for {ticker}")
                return None
            df = data[ticker]
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker}")
                return None

            try:
                ema_fast = vbt.MA.run(
                    df["Close"], window=self.ema_weekly_fast, ewm=True
                ).ma
                ema_slow = vbt.MA.run(
                    df["Close"], window=self.ema_weekly_slow, ewm=True
                ).ma
            except Exception as e:
                logger.error(f"Error calculating EMAs for {ticker}: {e}")
                return None

            if len(df) < 1 or len(ema_fast) < 1 or len(ema_slow) < 1:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            latest_close = df["Close"].iloc[-1]
            latest_ema_fast = ema_fast.iloc[-1]
            latest_ema_slow = ema_slow.iloc[-1]

            if (
                not pd.isna(latest_ema_fast)
                and not pd.isna(latest_ema_slow)
                and (latest_close > latest_ema_fast)
                and (latest_ema_fast > latest_ema_slow)
            ):
                return ticker
            return None

        result_tickers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(ticker_qualifies, ticker): ticker for ticker in tickers
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    result_tickers.append(result)

        output_path = os.path.join("out", "weekly_tickers.csv")
        pd.DataFrame({"Ticker": result_tickers}).to_csv(output_path, index=False)
        logger.info(f"Exported {len(result_tickers)} tickers to {output_path}")
        return result_tickers


# -----------------------------
# Ticker Processing
# -----------------------------
def process_ticker(
    ticker: str,
    strategy: BaseStrategy,
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> Tuple[
    Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]
]:
    """
    Process a single ticker to generate trading signals using the provided strategy.
    """
    logger.info(f"Processing ticker: {ticker}")
    try:
        if ticker not in daily_data or ticker not in weekly_data:
            logger.warning(f"No data available for {ticker}")
            return None, None, None, None

        daily = daily_data[ticker]
        weekly = weekly_data[ticker]
        return strategy.generate_signals(daily, weekly, date_index)
    except Exception as error:
        logger.error(f"Error processing {ticker}: {error}", exc_info=True)
        return None, None, None, None


# -----------------------------
# Strategy Evaluation
# -----------------------------
def evaluate_strategy(portfolio: vbt.Portfolio) -> List[str]:
    """
    Evaluate the performance of a trading strategy and return summary lines.
    """
    total_profit = portfolio.total_profit().sum()
    total_return = (total_profit / INITIAL_CASH) * 100
    trades = portfolio.trades.records_readable

    profitable_trades = trades["PnL"] > 0
    num_profitable_trades = profitable_trades.sum()
    total_closed_trades = trades["PnL"].count()
    win_rate = (
        (num_profitable_trades / total_closed_trades * 100)
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


def get_strategy() -> BaseStrategy:
    """Factory function to select and return a trading strategy."""
    return TrendMomentumVolumeATRStrategy()


# -----------------------------
# Data Loading Helpers
# -----------------------------
def load_tickers(csv_file: str) -> List[str]:
    """Load ticker symbols from a CSV file."""
    tickers_df = pd.read_csv(csv_file)
    return [t.replace("/", "-") for t in tickers_df["Symbol"].to_list()]


def prepare_data_dicts(
    combined_df: pd.DataFrame, tickers: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Extract per-ticker DataFrames from the combined DataFrame.
    """
    data_dict: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_cols = [
            col for col in combined_df.columns if col.startswith(f"{ticker}_")
        ]
        if ticker_cols:
            df = combined_df[ticker_cols].rename(
                columns=lambda x: x.replace(f"{ticker}_", "")
            )
            data_dict[ticker] = df
    return data_dict


# -----------------------------
# Main Execution
# -----------------------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    input_csv = os.path.join("input", f"{CSV_NAME}.csv")
    tickers = load_tickers(input_csv)
    logger.info(f"Loaded {len(tickers)} tickers from {input_csv}")

    session = get_session()
    if DOWNLOAD_NEW_DATA:
        daily_all = download_data_all("1d", session, tickers)
        weekly_all = download_data_all("1wk", session, tickers)
    else:
        logger.info("Loading cached combined data")
        daily_path = os.path.join(CACHE_DIR, f"{CSV_NAME}1d.csv")
        weekly_path = os.path.join(CACHE_DIR, f"{CSV_NAME}1wk.csv")
        daily_all = pd.read_csv(daily_path, index_col="Date", parse_dates=True)
        weekly_all = pd.read_csv(weekly_path, index_col="Date", parse_dates=True)

    daily_data_dict = prepare_data_dicts(daily_all, tickers)
    weekly_data_dict = prepare_data_dicts(weekly_all, tickers)

    missing_tickers = [t for t in tickers if t not in daily_data_dict]
    if missing_tickers:
        logger.warning(f"Missing daily data for tickers: {missing_tickers}")

    date_index = pd.date_range(start=START_DATE, end=END_DATE)
    strategy = get_strategy()

    # Initialize DataFrames for signals
    entries = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    exits = pd.DataFrame(False, index=date_index, columns=tickers, dtype=bool)
    sizes = pd.DataFrame(0.0, index=date_index, columns=tickers, dtype=float)

    results: Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(
                process_ticker,
                ticker,
                strategy,
                daily_data_dict,
                weekly_data_dict,
                date_index,
            ): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_item = future_to_ticker[future]
            t_entries, t_exits, t_sizes, t_exit_prices = future.result()
            if t_entries is not None:
                results[ticker_item] = (t_entries, t_exits, t_sizes, t_exit_prices)

    # Populate signal DataFrames from per-ticker results
    for ticker_item, (t_entries, t_exits, t_sizes, _) in results.items():
        entries[ticker_item] = t_entries
        exits[ticker_item] = t_exits
        sizes[ticker_item] = t_sizes

    # Build close prices from daily data
    close_prices = pd.DataFrame(index=date_index, columns=tickers, dtype=np.float64)
    for ticker_item in tickers:
        if ticker_item in daily_data_dict:
            close_prices[ticker_item] = daily_data_dict[ticker_item]["Close"]
        else:
            close_prices[ticker_item] = np.nan

    # Override close prices on exit dates with custom exit prices from our strategy
    for ticker_item, (_, t_exits, _, t_exit_prices) in results.items():
        exit_dates = t_exit_prices[t_exits].index
        for date in exit_dates:
            if not pd.isna(t_exit_prices.loc[date]):
                close_prices.loc[date, ticker_item] = t_exit_prices.loc[date]

    # Create portfolio from signals
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

    # Evaluate and print the strategy performance
    evaluation = evaluate_strategy(portfolio)
    for line in evaluation:
        print(line)

    # Export trade records to CSV
    trades_df = portfolio.trades.records_readable
    trades_df.to_csv("strategy_trades.csv", index=False)
    logger.info(
        "Strategy evaluation complete. Trades exported to 'strategy_trades.csv'."
    )


if __name__ == "__main__":
    # main()
    # Optionally, run the weekly filter if desired:
    strategy = TrendMomentumVolumeATRStrategy()
    strategy.filter_weekly()
