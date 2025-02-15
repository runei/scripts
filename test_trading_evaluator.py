import time

import numpy as np
import pandas as pd
import yfinance as yf

# Import functions, classes, and globals from your module.
# (Assume your module is named `strategy.py`)
from trading_evaluator import (
    INITIAL_CASH,
    RISK_PER_TRADE,
    DefaultStrategy,
    daily_data_dict,
    download_data_with_retry,
    evaluate_strategy,
    flatten_columns,
    process_ticker,
    weekly_data_dict,
)


# -------------
# flatten_columns tests
# -------------
def test_flatten_columns_multiindex():
    # Create a DataFrame with a MultiIndex on columns.
    arrays = [["A", "A"], ["x", "y"]]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
    df = pd.DataFrame([[1, 2], [3, 4]], columns=index)
    flat_df = flatten_columns(df)
    expected_columns = ["A_x", "A_y"]
    assert list(flat_df.columns) == expected_columns


def test_flatten_columns_tuple():
    # Create a DataFrame with tuple-based column names.
    df = pd.DataFrame([[1, 2], [3, 4]], columns=[("B", "x"), ("B", "y")])
    flat_df = flatten_columns(df)
    expected_columns = ["B_x", "B_y"]
    assert list(flat_df.columns) == expected_columns


def test_flatten_columns_none():
    # Passing None should return an empty DataFrame.
    result = flatten_columns(None)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# -------------
# download_data_with_retry tests
# -------------
def fake_download_success(
    ticker_list, start, end, interval, session, group_by, progress
):
    dates = pd.date_range(start=start, end=end)
    # Create a fake DataFrame with simple columns.
    data = pd.DataFrame(
        {
            "Open": np.linspace(1, 2, len(dates)),
            "High": np.linspace(2, 3, len(dates)),
            "Low": np.linspace(0.5, 1.5, len(dates)),
            "Close": np.linspace(1, 2, len(dates)),
            "Volume": np.linspace(100, 200, len(dates)),
        },
        index=dates,
    )
    # If a single ticker is provided, prepend the ticker symbol to the column names.
    if isinstance(ticker_list, list) and len(ticker_list) == 1:
        data.columns = [f"{ticker_list[0]}_{col}" for col in data.columns]
    return data


def fake_download_failure(*args, **kwargs):
    raise ValueError("Fake download error")


def test_download_data_with_retry_success(monkeypatch):
    monkeypatch.setattr(yf, "download", fake_download_success)
    session = object()  # Dummy session.
    ticker_list = ["FAKE"]
    start = "2020-01-01"
    end = "2020-01-05"
    interval = "1d"
    df = download_data_with_retry(ticker_list, interval, session, start, end)
    # Expect a nonempty DataFrame with ticker-prefixed columns.
    assert not df.empty
    for col in df.columns:
        assert col.startswith("FAKE_")


def test_download_data_with_retry_failure(monkeypatch):
    monkeypatch.setattr(yf, "download", fake_download_failure)
    session = object()
    ticker_list = ["FAKE"]
    start = "2020-01-01"
    end = "2020-01-05"
    interval = "1d"
    start_time = time.time()
    df = download_data_with_retry(ticker_list, interval, session, start, end)
    elapsed = time.time() - start_time
    # Since download fails after retries, an empty DataFrame should be returned.
    assert df.empty
    # Also, check that some delay occurred due to retry waits.
    assert elapsed >= 1.5  # Using the REQUEST_DELAY default


# -------------
# DefaultStrategy.generate_signals tests
# -------------
def test_default_strategy_generate_signals():
    # Create fake daily data.
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    daily = pd.DataFrame(
        {
            "Close": np.linspace(10, 20, 10),
            "Low": np.linspace(9, 19, 10),
            "High": np.linspace(11, 21, 10),
        },
        index=dates,
    )
    # For simplicity, let the weekly data be the same.
    weekly = daily.copy()
    fake_date_index = dates

    strategy = DefaultStrategy(
        ema_weekly_fast=5,
        ema_weekly_slow=8,
        ema_daily=5,
        risk_per_trade=RISK_PER_TRADE,
        initial_cash=INITIAL_CASH,
    )
    entries, exits, sizes = strategy.generate_signals(
        daily.copy(), weekly.copy(), fake_date_index
    )

    # Ensure that the returned objects are pandas Series and have the same length as date_index.
    for series in (entries, exits, sizes):
        assert isinstance(series, pd.Series)
        assert len(series) == len(fake_date_index)


# -------------
# process_ticker tests
# -------------
def test_process_ticker(monkeypatch):
    # Create fake data for a ticker.
    fake_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    fake_daily = pd.DataFrame(
        {
            "Close": np.linspace(10, 20, 10),
            "Low": np.linspace(9, 19, 10),
            "High": np.linspace(11, 21, 10),
        },
        index=fake_dates,
    )
    fake_weekly = fake_daily.copy()

    # Set the module globals for data.
    daily_data_dict.clear()
    weekly_data_dict.clear()
    global date_index
    date_index = fake_dates

    ticker = "FAKE"
    daily_data_dict[ticker] = fake_daily
    weekly_data_dict[ticker] = fake_weekly

    strategy = DefaultStrategy(
        ema_weekly_fast=5,
        ema_weekly_slow=8,
        ema_daily=5,
        risk_per_trade=RISK_PER_TRADE,
        initial_cash=INITIAL_CASH,
    )
    entries, exits, sizes = process_ticker(ticker, strategy)
    # Check that signals were returned.
    assert entries is not None
    assert exits is not None
    assert sizes is not None
    # Verify that the signal Series have the same length as the fake date index.
    for series in (entries, exits, sizes):
        assert len(series) == len(fake_dates)


# -------------
# evaluate_strategy tests
# -------------
# Create dummy classes to simulate a portfolio for evaluation.
class FakeTrades:
    def stats(self):
        return {
            "Profit Factor": 1.5,
            "Avg Winning Trade [%]": 2.0,
            "Avg Losing Trade [%]": -1.0,
        }

    @property
    def records_readable(self):
        # Return a simple DataFrame to simulate trade records.
        return pd.DataFrame({"Status": ["Win", "Loss", "Win"]})


class FakePortfolio:
    def __init__(self):
        self.trades = FakeTrades()

    def total_return(self):
        return pd.Series([0.1, 0.2, 0.15])

    def annualized_return(self):
        return pd.Series([0.12, 0.18, 0.15])

    def max_drawdown(self):
        return pd.Series([0.05, 0.07, 0.06])

    def plot(self, subplots):
        class FakeFigure:
            def show(self):
                pass

        return FakeFigure()


def fake_yf_download(ticker, start, end):
    dates = pd.date_range(start=start, end=end)
    # Create a fake benchmark DataFrame.
    return pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates)


def test_evaluate_strategy(monkeypatch):
    # Monkey-patch yf.download so that evaluate_strategy does not perform a real network call.
    monkeypatch.setattr(yf, "download", fake_yf_download)
    fake_portfolio = FakePortfolio()
    evaluation = evaluate_strategy(fake_portfolio, benchmark_ticker="FAKE")
    expected_keys = {
        "Total Return (%)",
        "Annualized Return (%)",
        "Benchmark Return (%)",
        "Max Drawdown (%)",
        "Benchmark Volatility (%)",
        "Total Trades",
        "Win Rate (%)",
        "Profit Factor",
        "Avg Win Trade (%)",
        "Avg Loss Trade (%)",
    }
    assert isinstance(evaluation, dict)
    assert expected_keys.issubset(evaluation.keys())
