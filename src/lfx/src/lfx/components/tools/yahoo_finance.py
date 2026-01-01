import ast
import pprint
import time
import hashlib
import json
import os
import threading
import random
from enum import Enum

from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import DropdownInput, IntInput, MessageTextInput
from lfx.log.logger import logger
from lfx.schema.data import Data

# Configuration (can be adjusted)
# RATE_PER_MIN: number of allowed yfinance calls per minute (per-process)
RATE_PER_MIN = int(os.environ.get("YFINANCE_RATE_PER_MIN", "10"))
TOKEN_BUCKET_CAPACITY = float(os.environ.get("YFINANCE_TOKEN_CAPACITY", "2"))
CACHE_TTL_SECONDS = int(os.environ.get("YFINANCE_CACHE_TTL_SECONDS", str(60 * 60)))  # default 1 hour
CACHE_MAX_ENTRIES = int(os.environ.get("YFINANCE_CACHE_MAX_ENTRIES", "512"))

# Simple in-memory TTL cache for history results
# Structure: _history_cache[cache_key] = (timestamp_seconds, records)
# Use setdefault in helpers to ensure the dict exists even if something odd happens at import time.
_history_cache: dict[str, tuple[float, object]] = {}


def _make_cache_key_for_history(symbol: str, start_date: str | None, end_date: str | None, period: str, interval: str) -> str:
    raw = f"{symbol}|{start_date or ''}|{end_date or ''}|{period}|{interval}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cached_history(cache_key: str):
    # Ensure the cache dict exists (defensive)
    cache = globals().setdefault("_history_cache", {})
    entry = cache.get(cache_key)
    if not entry:
        return None
    ts, records = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        # expired
        try:
            del cache[cache_key]
        except KeyError:
            pass
        return None
    return records


def _set_cached_history(cache_key: str, records: object):
    # Ensure the cache dict exists (defensive)
    cache = globals().setdefault("_history_cache", {})
    # Simple eviction if cache grows too big (remove oldest)
    if len(cache) >= CACHE_MAX_ENTRIES:
        # remove oldest entry
        oldest_key = min(cache.items(), key=lambda kv: kv[1][0])[0]
        try:
            del cache[oldest_key]
        except KeyError:
            pass
    cache[cache_key] = (time.time(), records)


# Token bucket rate limiter (per-process)
class TokenBucket:
    def __init__(self, rate_per_min: float, capacity: float):
        # rate_per_min: tokens added per minute
        self.rate_per_sec = float(rate_per_min) / 60.0
        self.capacity = float(capacity)
        self._tokens = self.capacity
        self._last = time.time()
        self._lock = threading.Lock()

    def _add_tokens(self):
        now = time.time()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        added = elapsed * self.rate_per_sec
        self._tokens = min(self.capacity, self._tokens + added)
        self._last = now

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._add_tokens()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_for_token(self, tokens: float = 1.0, poll_sleep: float = 0.1):
        # Wait until a token is available
        while True:
            if self.consume(tokens):
                return
            # small jitter to avoid thundering herd
            time.sleep(poll_sleep + random.random() * 0.05)


# Initialize a module-level token bucket
_token_bucket = TokenBucket(rate_per_min=RATE_PER_MIN, capacity=TOKEN_BUCKET_CAPACITY)

# Retry wrapper: try to use tenacity if available, otherwise a simple fallback
# Ensure the symbol is always defined to avoid NameError if import fails earlier
TENACITY_AVAILABLE = False
try:
    import tenacity  # type: ignore

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


def _call_with_rate_and_retry(func, *args, **kwargs):
    """
    Consume a token, then call func with retries/backoff.
    Uses tenacity if available; otherwise uses simple exponential backoff with jitter.
    """
    # Wait for token (rate limiting)
    logger.debug("YahooFinance: waiting for rate token")
    _token_bucket.wait_for_token()

    if TENACITY_AVAILABLE:
        # Use tenacity.Retrying to call the function with retry/backoff
        retrying = tenacity.Retrying(
            wait=tenacity.wait_exponential_jitter(initial=1, max=60),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )
        return retrying.call(lambda: func(*args, **kwargs))
    else:
        # Fallback: simple retry loop
        attempts = 5
        delay = 1.0
        for i in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception:
                if i == attempts - 1:
                    raise
                jitter = random.random() * 0.5
                sleep_time = delay + jitter
                logger.debug(f"YahooFinance: call failed, retrying in {sleep_time:.2f}s (attempt {i+1}/{attempts})")
                time.sleep(sleep_time)
                delay = min(delay * 2, 60.0)


class YahooFinanceMethod(Enum):
    GET_INFO = "get_info"
    GET_NEWS = "get_news"
    GET_ACTIONS = "get_actions"
    GET_ANALYSIS = "get_analysis"
    GET_BALANCE_SHEET = "get_balance_sheet"
    GET_CALENDAR = "get_calendar"
    GET_CASHFLOW = "get_cashflow"
    GET_INSTITUTIONAL_HOLDERS = "get_institutional_holders"
    GET_RECOMMENDATIONS = "get_recommendations"
    GET_SUSTAINABILITY = "get_sustainability"
    GET_MAJOR_HOLDERS = "get_major_holders"
    GET_MUTUALFUND_HOLDERS = "get_mutualfund_holders"
    GET_INSIDER_PURCHASES = "get_insider_purchases"
    GET_INSIDER_TRANSACTIONS = "get_insider_transactions"
    GET_INSIDER_ROSTER_HOLDERS = "get_insider_roster_holders"
    GET_DIVIDENDS = "get_dividends"
    GET_CAPITAL_GAINS = "get_capital_gains"
    GET_SPLITS = "get_splits"
    GET_SHARES = "get_shares"
    GET_FAST_INFO = "get_fast_info"
    GET_SEC_FILINGS = "get_sec_filings"
    GET_RECOMMENDATIONS_SUMMARY = "get_recommendations_summary"
    GET_UPGRADES_DOWNGRADES = "get_upgrades_downgrades"
    GET_EARNINGS = "get_earnings"
    GET_INCOME_STMT = "get_income_stmt"
    GET_HISTORY = "get_history"


class YahooFinanceSchema(BaseModel):
    symbol: str = Field(..., description="The stock symbol to retrieve data for.")
    method: YahooFinanceMethod = Field(YahooFinanceMethod.GET_INFO, description="The type of data to retrieve.")
    num_news: int | None = Field(5, description="The number of news articles to retrieve.")
    start_date: str | None = Field(None, description="Custom start date for historical data (YYYY-MM-DD).")
    end_date: str | None = Field(None, description="Custom end date for historical data (YYYY-MM-DD).")
    period: str = Field("1y", description="Default time period if no specific dates provided (e.g., 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max).")
    interval: str = Field("1d", description="Data granularity (e.g., 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo).")


class YfinanceToolComponent(LCToolComponent):
    display_name = "Yahoo! Finance"
    description = """Uses [yfinance](https://pypi.org/project/yfinance/) (unofficial package) \
to access financial data and market information from Yahoo! Finance."""
    icon = "trending-up"
    name = "YahooFinanceTool"
    legacy = True
    replacement = ["yahoosearch.YfinanceComponent"]

    inputs = [
        MessageTextInput(
            name="symbol",
            display_name="Stock Symbol",
            info="The stock symbol to retrieve data for (e.g., AAPL, GOOG).",
        ),
        DropdownInput(
            name="method",
            display_name="Data Method",
            info="The type of data to retrieve.",
            options=list(YahooFinanceMethod),
            value="get_news",
        ),
        IntInput(
            name="num_news",
            display_name="Number of News",
            info="The number of news articles to retrieve (only applicable for get_news).",
            value=5,
        ),
        MessageTextInput(
            name="start_date",
            display_name="Start Date",
            info="Custom start date for historical data (YYYY-MM-DD). Optional.",
        ),
        MessageTextInput(
            name="end_date",
            display_name="End Date",
            info="Custom end date for historical data (YYYY-MM-DD). Optional.",
        ),
        DropdownInput(
            name="period",
            display_name="Period",
            info="Default time period if no specific dates provided.",
            options=[
                "1d",
                "5d",
                "1mo",
                "3mo",
                "6mo",
                "1y",
                "2y",
                "5y",
                "10y",
                "ytd",
                "max",
            ],
            value="1y",
        ),
        DropdownInput(
            name="interval",
            display_name="Interval",
            info="Data granularity (e.g., 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo).",
            options=[
                "1m",
                "2m",
                "5m",
                "15m",
                "30m",
                "60m",
                "90m",
                "1h",
                "1d",
                "5d",
                "1wk",
                "1mo",
                "3mo",
            ],
            value="1d",
        ),
    ]

    def run_model(self) -> list[Data]:
        return self._yahoo_finance_tool(
            self.symbol,
            self.method,
            self.num_news,
            self.start_date,
            self.end_date,
            self.period,
            self.interval,
        )

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="yahoo_finance",
            description="Access financial data and market information from Yahoo! Finance.",
            func=self._yahoo_finance_tool,
            args_schema=YahooFinanceSchema,
        )

    def _yahoo_finance_tool(
        self,
        symbol: str,
        method: YahooFinanceMethod | str,
        num_news: int | None = 5,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[Data]:
        # Normalize method to enum if needed (handles case where tool passes a string)
        if isinstance(method, YahooFinanceMethod):
            method_enum = method
        else:
            # try by enum value first (e.g., "get_history"), then by name (e.g., "GET_HISTORY")
            try:
                method_enum = YahooFinanceMethod(method)  # by value
            except Exception:
                try:
                    method_enum = YahooFinanceMethod[method.upper()]  # by name
                except Exception:
                    raise ToolException(f"Unknown yahoo finance method: {method!r}")

        try:
            import yfinance as yf
        except ImportError as e:
            msg = "yfinance is required for YahooFinanceTool"
            raise ImportError(msg) from e

        ticker = yf.Ticker(symbol)

        try:
            # For history, check cache first (cache misses go through rate limiter and retry)
            if method_enum == YahooFinanceMethod.GET_HISTORY:
                cache_key = _make_cache_key_for_history(symbol, start_date, end_date, period, interval)
                cached = _get_cached_history(cache_key)
                if cached is not None:
                    logger.debug(f"YahooFinance: returning cached history for {symbol}")
                    return [Data(data={"historical": cached})]

                # Define a callable to fetch history (so we can wrap it)
                def _fetch_history():
                    if start_date and end_date:
                        return ticker.history(start=start_date, end=end_date, interval=interval)
                    return ticker.history(period=period, interval=interval)

                historical = _call_with_rate_and_retry(_fetch_history)

                # Convert DataFrame to list of records (with index reset to include datetime)
                try:
                    records = historical.reset_index().to_dict(orient="records")
                except Exception:
                    records = pprint.pformat(historical)

                # store in cache
                try:
                    _set_cached_history(cache_key, records)
                except Exception:
                    logger.debug("YahooFinance: failed to set cache (non-fatal)")

                return [Data(data={"historical": records})]

            # For other methods, we call through the rate-limited wrapper
            if method_enum == YahooFinanceMethod.GET_INFO:
                result = _call_with_rate_and_retry(lambda: ticker.info)
                result = pprint.pformat(result)
                return [Data(data={"result": result})]

            if method_enum == YahooFinanceMethod.GET_NEWS:
                # news may be a list; fetch via rate-limited wrapper
                result = _call_with_rate_and_retry(lambda: ticker.news)
                # slice if num_news provided
                if isinstance(result, list) and num_news is not None:
                    result = result[:num_news]
                # preserve previous behavior of formatting then literal-eval if needed
                # but if result is already list/dict we can return directly
                if isinstance(result, (list, dict)):
                    return [Data(data=article) for article in result] if isinstance(result, list) else [Data(data=result)]
                result = pprint.pformat(result)
                try:
                    return [Data(data=article) for article in ast.literal_eval(result)]
                except Exception:
                    return [Data(data={"result": result})]

            # Generic getattr handlers (rate-limited)
            def _call_method():
                return getattr(ticker, method_enum.value)()

            result = _call_with_rate_and_retry(_call_method)
            result = pprint.pformat(result)
            return [Data(data={"result": result})]

        except Exception as e:
            error_message = f"Error retrieving data: {e}"
            logger.debug(error_message)
            self.status = error_message
            raise ToolException(error_message) from e
