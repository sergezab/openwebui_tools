"""
title: Stock Market Helper
description: A comprehensive stock analysis tool that gathers data from Finnhub API and compiles a detailed report.
author: Sergii Zabigailo
author_url: https://github.com/sergezab/
github: https://github.com/sergezab/openwebui_tools/
funding_url: https://github.com/open-webui
version: 0.2.1
license: MIT
requirements: finnhub-python,pytz
"""

import finnhub
import yfinance as yf  # For Yahoo Finance API
import requests
import aiohttp
import asyncio
import shelve
import os
import json
import logging
import pytz
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import TypedDict, List
import difflib
import traceback
from functools import wraps
import time
import random
from typing import (
    Dict,
    Any,
    List,
    Union,
    Generator,
    Iterator,
    Tuple,
    Optional,
    Callable,
    Awaitable,
)
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(
    logging.INFO
)  # Changed from WARNING to INFO for better cache visibility

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a FileHandler
log_dir = os.path.expanduser("~")
log_file = os.path.join(log_dir, "stock_reporter.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file, mode="a")  # using append mode here
file_handler.setLevel(logging.INFO)  # Changed from WARNING to INFO
file_handler.setFormatter(formatter)

# Create a StreamHandler (console)
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)  # Changed from WARNING to INFO
# stream_handler.setFormatter(formatter)

# Clear any existing handlers if necessary (be cautious if you know the upper app has its own config)
# if logger.hasHandlers():
#    logger.handlers.clear()

# Add our handlers to our logger
logger.addHandler(file_handler)
# logger.addHandler(stream_handler)


def _format_date(date: datetime) -> str:
    """Helper function to format date for Finnhub API"""
    return date.strftime("%Y-%m-%d")


def is_similar(text1: str, text2: str, threshold: float = 0.9) -> bool:
    """
    Returns True if text1 and text2 are similar above the threshold.
    The texts are first stripped and lowercased for a simple comparison.
    """
    # Remove any extra spaces and lowercase the texts
    text1_clean = text1.strip().lower()
    text2_clean = text2.strip().lower()
    similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()
    return similarity >= threshold


# Caching for expensive operations
@lru_cache(maxsize=128)
def _get_sentiment_model():
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def is_html_response(response):
    """Check if a response is HTML instead of the expected JSON"""
    if isinstance(response, str) and (
        "<html" in response.lower() or "<!doctype" in response.lower()
    ):
        return True
    return False


def retry_with_backoff(max_retries=3, initial_backoff=1, max_backoff=10):
    """
    Retry decorator with exponential backoff, with special handling for API rate limits
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            backoff = initial_backoff

            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)

                    # Check if this is a rate limit error (429)
                    is_rate_limit = (
                        "429" in error_str or "API limit reached" in error_str
                    )

                    # Don't retry for errors that aren't network/timeout/rate limit related
                    if not (
                        is_rate_limit
                        or "502" in error_str
                        or "504" in error_str
                        or "timeout" in error_str.lower()
                        or "connection" in error_str.lower()
                    ):
                        raise

                    retry_count += 1
                    if retry_count >= max_retries:
                        raise  # Max retries reached, re-raise the exception

                    # For rate limit errors, use a longer backoff
                    if is_rate_limit:
                        sleep_time = min(backoff * 3, 30) + random.uniform(
                            0, 1
                        )  # Longer wait for rate limits
                        logger.warning(
                            f"API rate limit hit. Waiting {sleep_time:.2f}s before retry {retry_count}/{max_retries}"
                        )
                    else:
                        sleep_time = backoff + random.uniform(0, 1)
                        logger.warning(
                            f"Retrying in {sleep_time:.2f}s after error: {error_str}"
                        )

                    time.sleep(sleep_time)

                    # Exponential backoff
                    backoff = min(backoff * 2, max_backoff)

            # This should never be reached due to the raise inside the loop
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Wrap all API methods with retry
@retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=15)
def get_company_profile(client, ticker):
    return client.company_profile2(symbol=ticker, timeout=10)


@retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=15)
def get_company_financials(client, ticker, metric_type="all"):
    return client.company_basic_financials(ticker, metric_type)


@retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=15)
def get_company_peers(client, ticker):
    return client.company_peers(ticker)


@retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=15)
def get_stock_quote(client, ticker):
    return client.quote(ticker, timeout=10)


@retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=15)
def get_news(client, ticker, start_date, end_date):
    return client.company_news(ticker, start_date, end_date, timeout=10)


def _get_basic_info(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company information from Finnhub API.
    Handles failures gracefully by providing default values.
    """
    result = {
        "profile": {
            "ticker": ticker,
            "name": ticker,
        },  # Ensure we always have name and ticker
        "basic_financials": {"metric": {}},  # Empty financials
        "peers": [],  # Empty peers list
    }

    try:
        # Try to get profile
        try:
            profile = get_company_profile(client, ticker)
            
            #Enable for debugging
            #logger.info(f"Raw profile data for {ticker}: {json.dumps(profile, indent=2)}")
            
            # Check if the response is valid JSON (not HTML)
            if is_html_response(profile):
                logger.warning(
                    f"Received HTML response instead of JSON for {ticker} profile"
                )
            elif isinstance(profile, dict):
                # Ensure we have both name and ticker in the profile
                if not profile.get("ticker"):
                    profile["ticker"] = ticker
                if not profile.get("name"):
                    profile["name"] = ticker
                result["profile"] = profile
            else:
                logger.warning(f"Invalid profile response format for {ticker}")
        except Exception as e:
            logger.warning(
                f"Could not fetch profile for {ticker}, using minimal profile: {str(e)}"
            )

        # Try to get financials
        try:
            basic_financials = get_company_financials(client, ticker, "all")
            if isinstance(basic_financials, dict) and "metric" in basic_financials:
                result["basic_financials"] = basic_financials
            else:
                logger.warning(f"Invalid financials response format for {ticker}")
        except Exception as e:
            logger.warning(
                f"Could not fetch financials for {ticker}, using empty financials: {str(e)}"
            )

        # Try to get peers
        try:
            peers = get_company_peers(client, ticker)
            if isinstance(peers, list):
                result["peers"] = peers
            else:
                logger.warning(f"Invalid peers response format for {ticker}")
        except Exception as e:
            logger.warning(
                f"Could not fetch peers for {ticker}, using empty peers list: {str(e)}"
            )

        return result

    except Exception as e:
        logger.error(f"Critical error in _get_basic_info for {ticker}: {str(e)}")
        return result  # Return default structure even on critical error

def _get_current_price(client: finnhub.Client, ticker: str) -> Optional[Dict[str, Union[float, str]]]:
    try:
        quote = client.quote(ticker)
        
        #Enable for debugging
        #logger.info(f"Raw quote data for {ticker}: {json.dumps(quote, indent=2)}")
        
        current_price = quote.get("c", 0)
        daily_percent = quote.get("dp", 0) 
        previous_close = quote.get("pc", 0)
        
        # DETECT SUSPICIOUS DATA - likely delisted stock
        is_suspicious = (
            abs(daily_percent) > 1000 or  # Extreme percentage change
            previous_close < 0.10 or      # Previous close under 10 cents
            current_price > previous_close * 50  # Current price 50x+ previous close
        )
        
        if is_suspicious:
            logger.warning(f"Suspicious price data for {ticker} - likely delisted/stale data")
            logger.warning(f"  Current: ${current_price}, Previous: ${previous_close}, Change: {daily_percent}%")
            
            # Return data but mark the change as unreliable
            return {
                "current_price": current_price,
                "change": 0.0,  # Set to 0 instead of the bogus percentage
                "change_amount": 0.0,
                "high": quote.get("h", 0),
                "low": quote.get("l", 0), 
                "open": quote.get("o", 0),
                "previous_close": previous_close,
                "data_warning": "Potentially stale/delisted"
            }
        
        return {
            "current_price": current_price,
            "change": daily_percent,
            "change_amount": quote.get("d", 0),
            "high": quote.get("h", 0),
            "low": quote.get("l", 0),
            "open": quote.get("o", 0),
            "previous_close": previous_close,
        }
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {str(e)}")
        return None

class NewsItem(TypedDict):
    url: str
    title: str
    summary: str


def _get_company_news(client: finnhub.Client, ticker: str) -> List[NewsItem]:
    """
    Fetch recent news articles about the company from Finnhub API.
    Returns an empty list if API call fails.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        news = client.company_news(
            ticker, _format_date(start_date), _format_date(end_date)
        )
        news_items = news[:10]  # Get the first 10 news items
        return [
            {"url": item["url"], "title": item["headline"], "summary": item["summary"]}
            for item in news_items
        ]
    except Exception as e:
        logger.warning(
            f"Could not fetch news for {ticker}, using empty news list: {str(e)}"
        )
        return []  # Return empty list instead of raising exception


async def _async_web_scrape(session: aiohttp.ClientSession, url: str) -> str:
    """
    Scrape and process a web page using r.jina.ai

    :param session: The aiohttp ClientSession to use for the request.
    :param url: The URL of the web page to scrape.
    :return: The scraped and processed content without the Links/Buttons section, or an error message.
    """
    jina_url = f"https://r.jina.ai/{url}"

    headers = {
        "X-No-Cache": "true",
        "X-With-Images-Summary": "true",
        "X-With-Links-Summary": "true",
    }

    try:
        async with session.get(jina_url, headers=headers) as response:
            response.raise_for_status()
            content = await response.text()

        # Extract content and remove Links/Buttons section as its too many tokens
        links_section_start = content.rfind("Images:")
        if links_section_start != -1:
            content = content[:links_section_start].strip()

        return content

    except aiohttp.ClientError as e:
        logger.error(f"Error scraping web page {url}: {str(e)}")
        return f"Error scraping web page: {str(e)}"


# Asynchronous sentiment analysis
async def _async_sentiment_analysis(
    summary: str, title: str
) -> Dict[str, Union[str, float, List[List[float]]]]:
    """
    Perform sentiment analysis on text content with improved title-based validation.
    Returns neutral sentiment with 0 confidence if analysis fails.
    """
    try:
        tokenizer, model = _get_sentiment_model()

        # Comprehensive sentiment keyword lists
        negative_keywords = [
            "weak",
            "concern",
            "risk",
            "overvalued",
            "bearish",
            "sell",
            "short",
            "elevated risk",
            "unreasonable price",
            "dip",
            "stubborn",
            "valuation concerns",
            "downgrade",
            "warning",
            "caution",
            "negative",
            "decline",
            "drop",
            "fall",
            "loses",
            "plunge",
            "crash",
            "bubble",
            "expensive",
            "correction",
            "worrying",
            "deteriorating",
            "miss",
            "missed",
            "missing",
            "fails",
            "failed",
            "disappointing",
        ]
        positive_keywords = [
            "strong",
            "buy",
            "bullish",
            "undervalued",
            "opportunity",
            "upgrade",
            "outperform",
            "beat",
            "beats",
            "beating",
            "exceeded",
            "exceeds",
            "growth",
            "growing",
            "positive",
            "advantage",
            "potential",
            "momentum",
            "successful",
            "success",
            "gain",
            "gains",
            "winning",
            "rally",
            "surge",
            "breakthrough",
            "innovative",
            "leading",
            "leader",
            "dominant",
        ]

        # Combine title and summary to create a fuller context for sentiment evaluation.
        combined_text = f"Headline: {title}\nSummary: {summary}"
        combined_text_lower = combined_text.lower()

        # Keyword-based sentiment scoring using the combined text
        negative_count = sum(
            1 for word in negative_keywords if word.lower() in combined_text_lower
        )
        positive_count = sum(
            1 for word in positive_keywords if word.lower() in combined_text_lower
        )

        # Calculate base sentiment from keywords
        if negative_count > positive_count:
            base_sentiment = "Negative"
            base_confidence = min(
                0.9, negative_count / (negative_count + positive_count + 1)
            )
        elif positive_count > negative_count:
            base_sentiment = "Positive"
            base_confidence = min(
                0.9, positive_count / (negative_count + positive_count + 1)
            )
        else:
            base_sentiment = "Neutral"
            base_confidence = 0.5

        #logger.info(f"Keyword-based sentiment: {base_sentiment} (confidence: {base_confidence})")

        # Perform model-based sentiment analysis on the combined text
        inputs = tokenizer(
            combined_text_lower, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        temperature = 1  # adjust this value
        scaled_logits = outputs.logits / temperature
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)

        model_sentiment = "<none>"
        sentiment_scores = probabilities.tolist()[0]
        model_confidence = max(sentiment_scores)

        # Use model.config.id2label if available, otherwise fall back to the list
        if hasattr(model.config, "id2label"):
            id2label = (
                model.config.id2label
            )  # e.g., {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
            predicted_index = torch.argmax(probabilities, dim=-1).item()
            model_sentiment = id2label[predicted_index]
        else:
            # Fallback if no config is available
            # The model's labels assumed here are: 0: Neutral, 1: Positive, 2: Negative.
            sentiments = ["Neutral", "Positive", "Negative"]
            model_sentiment = sentiments[sentiment_scores.index(max(sentiment_scores))]

        #logger.info(f"Combined text: {combined_text}")
        # logger.info(f"Logits: {outputs.logits.tolist()}")
        # logger.info(f"Scaled logits: {scaled_logits.tolist()}")
        # logger.info(f"Probabilities: {sentiment_scores}")
        logger.info(f"Model sentiment: {model_sentiment} (confidence: {model_confidence})")

        # Combine the two sentiment evaluations
        if base_sentiment == model_sentiment:
            sentiment = base_sentiment
            confidence = (base_confidence + model_confidence) / 2
        else:
            # If they disagree, choose the one with higher confidence (with a slight penalty)
            if base_confidence > model_confidence:
                sentiment = base_sentiment
                confidence = base_confidence * 0.8
            else:
                sentiment = model_sentiment
                confidence = model_confidence * 0.8

        formatted_probs = formatted_probs = [
            [round(p, 5) for p in sample] for sample in probabilities.tolist()
        ]

        return {
            "sentiment": sentiment.lower(),
            "confidence": confidence,
            "probabilities": formatted_probs,
        }

    except Exception as e:
        logger.warning(
            f"Error in sentiment analysis, using neutral sentiment: {str(e)}"
        )
        return {"sentiment": "Neutral", "confidence": 0.0}


# Asynchronous data gathering


def _is_cache_valid(cached_data: Dict[str, Any], data_type: str) -> bool:
    """Check if cached data is from today"""

    if not cached_data or data_type not in cached_data:
        return False
    try:
        est = pytz.timezone("US/Eastern")
        cache_time = datetime.fromisoformat(cached_data[data_type]["timestamp"])
        if cache_time.tzinfo is None:
            cache_time = est.localize(cache_time)
        current_time = datetime.now(est)
        return cache_time.date() == current_time.date()
    except Exception as e:
        logger.error(f"Error validating cache for {data_type}: {str(e)}")
        return False


def get_last_trading_day(current_time: datetime) -> datetime:
    """
    Get the most recent trading day (excluding weekends and current day before market opens)
    """
    est = pytz.timezone("US/Eastern")
    if not isinstance(current_time, datetime):
        current_time = datetime.now(est)
    elif current_time.tzinfo is None:
        current_time = est.localize(current_time)

    # Start with current day
    last_trading_day = current_time

    # If it's before market open (9:30 AM), look at previous day
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    if current_time < market_open:
        last_trading_day = current_time - timedelta(days=1)

    # Keep going back until we find a weekday
    while last_trading_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        last_trading_day = last_trading_day - timedelta(days=1)

    return last_trading_day


def is_cache_stale(cache_time: datetime) -> bool:
    """
    Check if cached data is stale by comparing with last valid trading day and market hours
    """
    est = pytz.timezone("US/Eastern")
    if cache_time.tzinfo is None:
        cache_time = est.localize(cache_time)

    current_time = datetime.now(est)
    last_trading = get_last_trading_day(current_time)

    # Convert to dates for comparison
    cache_date = cache_time.date()
    current_date = current_time.date()
    last_trading_date = last_trading.date()

    # If cache is from before the last trading day, it's stale
    if cache_date < last_trading_date:
        return True

    # During market hours (9:30 AM - 4:00 PM EST)
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_open = (
        market_open <= current_time <= market_close and current_time.weekday() < 5
    )

    # If market is open, cache should be no older than 60 minutes
    if is_market_open:
        cache_age = (current_time - cache_time).total_seconds() / 60
        return cache_age > 60

    # If cache is from today but outside market hours, it's not stale
    if cache_date == current_date:
        return False

    # If cache is from last trading day, it must be from after market close
    if cache_date == last_trading_date:
        last_market_close = cache_time.replace(
            hour=16, minute=0, second=0, microsecond=0
        )
        return cache_time < last_market_close

    return True


def _is_market_hours() -> Tuple[bool, str]:
    """
    Check if current time is within market hours and return explanation
    Returns: (is_open: bool, reason: str)
    """
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False, "Market is closed (Weekend)"

    # Market hours: 9:30 AM - 4:00 PM EST
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < market_open:
        return (
            False,
            f"Market is not open yet (Opens at 9:30 AM EST, current time: {now.strftime('%I:%M %p')} EST)",
        )
    elif now > market_close:
        return (
            False,
            f"Market is closed for the day (Closed at 4:00 PM EST, current time: {now.strftime('%I:%M %p')} EST)",
        )

    return True, "Market is open"


def is_etf_or_fund(profile):
    """
    Determine if a security is an ETF or money market fund based on its profile.

    Args:
        profile (dict): The profile data from Finnhub API

    Returns:
        tuple: (is_etf_or_fund, type_str) where type_str is 'ETF', 'Money Market Fund', etc.
    """
    # Check profile data for indicators
    if not profile:
        return False, "Unknown"

    # Check name indicators for ETFs and funds
    name = profile.get("name", "").lower()
    ticker = profile.get("ticker", "").lower()
    security_type = profile.get("type", "").lower()

    # Common ETF indicators in name
    etf_keywords = [
        "etf",
        "exchange traded fund",
        "index fund",
        "index trust",
        "shares",
    ]

    # Common money market fund indicators
    mmf_keywords = [
        "money market",
        "cash reserves",
        "liquidity fund",
        "treasury",
        "govt",
    ]

    # Check for ETF patterns in name or ticker
    if any(keyword in name for keyword in etf_keywords) or ticker.endswith("etf"):
        return True, "ETF"

    # Check for money market fund patterns
    if any(keyword in name for keyword in mmf_keywords) or ticker.endswith("mm"):
        return True, "Money Market Fund"

    # Check security type if available
    if security_type:
        if "etf" in security_type or "fund" in security_type:
            return True, "ETF"

    # Additional specific checks for common ETF tickers
    common_etfs = {
        "spy",
        "qqq",
        "voo",
        "dia",
        "iwm",
        "vti",
        "gld",
        "xlf",
        "xle",
        "xlu",
        "vnq",
        "kweb",
        "ihi",
        "ewz",
        "bnd",
        "vusxx",
        "vwo",
        "botz",
        "snsxx",
    }
    if ticker.lower() in common_etfs:
        if "snsxx" in ticker.lower():
            return True, "Money Market Fund"
        return True, "ETF"

    return False, "Stock"


# Add this function to get ETF-specific data when available
async def _get_etf_data(client, ticker):
    """
    Retrieve ETF-specific data that would be more relevant than regular stock metrics.

    Args:
        client: Finnhub client
        ticker (str): The ETF ticker

    Returns:
        dict: ETF-specific data
    """
    etf_data = {
        "expense_ratio": None,
        "aum": None,  # Assets Under Management
        "nav": None,  # Net Asset Value
        "category": None,
        "top_holdings": [],
        "asset_allocation": {},
        "inception_date": None,
    }

    try:
        # Try to get ETF profile if available
        # Note: This would ideally be a specialized Finnhub endpoint for ETFs
        # Since it's not directly available, we can supplement with custom logic
        profile = get_company_profile(client, ticker)

        # Parse any available ETF data from profile
        # Many of these might not be available directly from Finnhub

        return etf_data
    except Exception as e:
        logger.warning(f"Could not fetch ETF-specific data for {ticker}: {str(e)}")
        return etf_data


async def _async_gather_stock_data(
    client: finnhub.Client, ticker: str, cache: shelve.Shelf
) -> Dict[str, Any]:
    """
    Gather all stock data with improved error handling.
    Enhanced to better handle ETFs and funds.
    """
    # Initialize result with valid default values
    result = {
        "basic_info": {
            "profile": {
                "ticker": ticker,
                "name": ticker,
            },
            "basic_financials": {"metric": {}},
            "peers": [],
        },
        "current_price": {
            "current_price": 0.0,
            "change": 0.0,
            "change_amount": 0.0,
            "high": 0.0,
            "low": 0.0,
            "open": 0.0,
            "previous_close": 0.0,
        },
        "sentiments": [],
        "is_etf": False,
        "fund_type": "Stock",
        "etf_data": {},
    }

    try:
        # Get the ticker's cache data; if not present, start with an empty dict
        ticker_cache = cache.get(ticker, {})

        try:
            # Get basic info (use cache if valid)
            if not _is_cache_valid(ticker_cache, "basic_info"):
                logger.info(f"Fetching fresh basic info for {ticker}")
                basic_info = _get_basic_info(client, ticker)

                # Check if the basic info looks valid (e.g. has a 'name' and 'ticker')
                profile = basic_info.get("profile", {})
                if profile.get("name") and profile.get("ticker"):
                    ticker_cache["basic_info"] = {
                        "data": basic_info,
                        "timestamp": datetime.now(
                            pytz.timezone("US/Eastern")
                        ).isoformat(),
                    }
                else:
                    logger.error(
                        f"Basic info for {ticker} is incomplete; not updating cache."
                    )
                    # If we have previous cached data, use that instead
                    if "basic_info" in ticker_cache:
                        logger.info(f"Using previously cached basic info for {ticker}")
                        basic_info = ticker_cache["basic_info"]["data"]
            else:
                logger.info(f"Using cached basic info for {ticker}")
                basic_info = ticker_cache["basic_info"]["data"]

            # Ensure result has valid basic info
            if not basic_info.get("profile", {}).get("name"):
                logger.warning(
                    f"Name missing in profile for {ticker}, using ticker as name"
                )
                basic_info.setdefault("profile", {})["name"] = ticker

            if not basic_info.get("profile", {}).get("ticker"):
                logger.warning(f"Ticker missing in profile for {ticker}, adding it")
                basic_info.setdefault("profile", {})["ticker"] = ticker

            result["basic_info"] = basic_info

            # Check if the security is an ETF or fund
            is_etf, fund_type = is_etf_or_fund(basic_info.get("profile", {}))
            result["is_etf"] = is_etf
            result["fund_type"] = fund_type

            # If it's an ETF/fund, try to get additional ETF-specific data
            if is_etf:
                logger.info(
                    f"Detected {fund_type} for {ticker}, fetching specialized data"
                )

                # Try to get ETF data from cache first
                if _is_cache_valid(ticker_cache, "etf_data"):
                    logger.info(f"Using cached ETF data for {ticker}")
                    result["etf_data"] = ticker_cache["etf_data"]["data"]
                else:
                    # Try to supplement with data from yfinance if Finnhub lacks ETF data
                    try:
                        # First attempt to get data from Finnhub
                        etf_data = {}

                        # Supplement with yfinance data if available
                        try:
                            logger.info(
                                f"Attempting to get supplemental ETF data from yfinance for {ticker}"
                            )
                            yf_ticker = yf.Ticker(ticker)
                            yf_info = yf_ticker.info

                            # Extract relevant ETF data
                            etf_data.update(
                                {
                                    "expense_ratio": yf_info.get("expenseRatio"),
                                    "category": yf_info.get("category"),
                                    "aum": yf_info.get("totalAssets"),
                                    "nav": yf_info.get("navPrice"),
                                    "inception_date": yf_info.get("fundInceptionDate"),
                                    "yield": yf_info.get("yield"),
                                    "ytd_return": yf_info.get("ytdReturn"),
                                    "three_year_return": yf_info.get(
                                        "threeYearAverageReturn"
                                    ),
                                    "five_year_return": yf_info.get(
                                        "fiveYearAverageReturn"
                                    ),
                                }
                            )

                            # For money market funds, get specific attributes
                            if fund_type == "Money Market Fund":
                                etf_data.update(
                                    {
                                        "seven_day_yield": yf_info.get("sevenDayYield"),
                                        "weighted_avg_maturity": yf_info.get(
                                            "weightedAverageMaturity"
                                        ),
                                    }
                                )

                            # Cache the ETF data
                            ticker_cache["etf_data"] = {
                                "data": etf_data,
                                "timestamp": datetime.now(
                                    pytz.timezone("US/Eastern")
                                ).isoformat(),
                            }
                            result["etf_data"] = etf_data

                        except Exception as e:
                            logger.warning(
                                f"Error getting supplemental ETF data from yfinance: {str(e)}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error getting ETF-specific data for {ticker}: {str(e)}"
                        )

                        # Still cache an empty result to avoid repeated failed attempts
                        ticker_cache["etf_data"] = {
                            "data": {},
                            "timestamp": datetime.now(
                                pytz.timezone("US/Eastern")
                            ).isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error getting basic info for {ticker}: {str(e)}")
            # We already initialized result with valid defaults

        try:
            # Check if we should use cached price
            use_cached_price = False
            if "current_price" in ticker_cache:
                cache_time = datetime.fromisoformat(
                    ticker_cache["current_price"]["timestamp"]
                )
                est = pytz.timezone("US/Eastern")
                if cache_time.tzinfo is None:
                    cache_time = est.localize(cache_time)
                is_market_open, market_status = _is_market_hours()
                # Check if cache is stale
                if not is_cache_stale(cache_time):
                    use_cached_price = True
                    est_time = datetime.now(est)
                    cache_age = (est_time - cache_time).total_seconds() / 60
                    logger.info(
                        f"Using cached price for {ticker} ({market_status}, cache age: {cache_age:.1f} minutes, EST: {est_time.strftime('%I:%M %p')})"
                    )
                else:
                    est_time = datetime.now(est)
                    logger.info(
                        f"Cache stale for {ticker} (EST: {est_time.strftime('%I:%M %p')})"
                    )

            if use_cached_price:
                current_price = ticker_cache["current_price"]["data"]
            else:
                # Get fresh price data - try Finnhub first, then fallback methods
                logger.info(f"Fetching fresh price for {ticker}")
                current_price = None

                # First try Finnhub
                try:
                    current_price = _get_current_price(client, ticker)
                except Exception as e:
                    logger.warning(f"Error getting price from Finnhub: {str(e)}")

                # If Finnhub fails or returns None, try yfinance as fallback for ETFs
                if current_price is None and result["is_etf"]:
                    try:
                        logger.info(
                            f"Attempting to get price data from yfinance for {ticker}"
                        )
                        yf_ticker = yf.Ticker(ticker)
                        hist = yf_ticker.history(period="2d")

                        if not hist.empty and len(hist) >= 1:
                            # Get the most recent data
                            latest = hist.iloc[-1]
                            prev = hist.iloc[-2] if len(hist) >= 2 else latest

                            current_price = {
                                "current_price": float(latest["Close"]),
                                "change": (
                                    float((latest["Close"] / prev["Close"] - 1) * 100)
                                    if prev["Close"] > 0
                                    else 0.0
                                ),
                                "change_amount": float(latest["Close"] - prev["Close"]),
                                "high": float(latest["High"]),
                                "low": float(latest["Low"]),
                                "open": float(latest["Open"]),
                                "previous_close": float(prev["Close"]),
                            }
                            logger.info(
                                f"Successfully retrieved yfinance price data for {ticker}"
                            )
                        else:
                            logger.warning(f"yfinance returned empty data for {ticker}")
                    except Exception as e:
                        logger.warning(f"Error getting yfinance price data: {str(e)}")

                # If still no price data, try previously cached data
                if current_price is None:
                    logger.error(
                        f"Failed to fetch fresh price for {ticker}. Using cached price if available."
                    )
                    if "current_price" in ticker_cache:
                        current_price = ticker_cache["current_price"]["data"]
                        logger.info(f"Using previously cached price for {ticker}.")
                    else:
                        # No cached price available; use a fallback value
                        logger.warning(
                            f"No cached price available for {ticker}, using defaults"
                        )
                        current_price = {
                            "current_price": 0.0,
                            "change": 0.0,
                            "change_amount": 0.0,
                            "high": 0.0,
                            "low": 0.0,
                            "open": 0.0,
                            "previous_close": 0.0,
                        }
                else:
                    # Only update the cache if fresh data is obtained
                    ticker_cache["current_price"] = {
                        "data": current_price,
                        "timestamp": datetime.now(
                            pytz.timezone("US/Eastern")
                        ).isoformat(),
                    }

            # Ensure we have valid price data
            if current_price is None or not isinstance(current_price, dict):
                logger.warning(f"Invalid price data for {ticker}, using defaults")
                current_price = {
                    "current_price": 0.0,
                    "change": 0.0,
                    "change_amount": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "open": 0.0,
                    "previous_close": 0.0,
                }

            result["current_price"] = current_price

        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            # We already initialized result with valid defaults

        try:
            # Get news and process sentiments
            if not _is_cache_valid(ticker_cache, "sentiments"):
                logger.info(f"Processing fresh news and sentiments for {ticker}")
                try:
                    news_items = _get_company_news(client, ticker)
                    if news_items:
                        # Filter out items without titles
                        news_items = [item for item in news_items if item.get("title")]

                        effective_summaries = [
                            (
                                ""
                                if is_similar(
                                    item.get("title", ""), item.get("summary", "")
                                )
                                else item.get("summary", "")
                            )
                            for item in news_items
                        ]

                        # Create sentiment analysis tasks
                        sentiment_tasks = [
                            _async_sentiment_analysis(
                                effective_summary, item.get("title", "")
                            )
                            for effective_summary, item in zip(
                                effective_summaries, news_items
                            )
                        ]
                        sentiments = await asyncio.gather(*sentiment_tasks)

                        sentiment_results = [
                            {
                                "url": item.get("url", ""),
                                "title": item.get("title", "No title"),
                                "summary": effective_summary,
                                "sentiment": sentiment.get("sentiment", "neutral"),
                                "confidence": sentiment.get("confidence", 0.0),
                                "probabilities": sentiment.get("probabilities", []),
                            }
                            for item, sentiment, effective_summary in zip(
                                news_items, sentiments, effective_summaries
                            )
                        ]

                        # Cache sentiment results
                        ticker_cache["sentiments"] = {
                            "data": sentiment_results,
                            "timestamp": datetime.now(
                                pytz.timezone("US/Eastern")
                            ).isoformat(),
                        }
                        result["sentiments"] = sentiment_results
                except Exception as e:
                    logger.error(
                        f"Error processing news/sentiments for {ticker}: {str(e)}"
                    )
                    # If we have cached sentiments, use those
                    if "sentiments" in ticker_cache:
                        result["sentiments"] = ticker_cache["sentiments"]["data"]
            else:
                logger.info(f"Using cached sentiments for {ticker}")
                result["sentiments"] = ticker_cache["sentiments"]["data"]
        except Exception as e:
            logger.error(f"Error in sentiment processing for {ticker}: {str(e)}")
            # We already initialized result with valid defaults

        # Write updated ticker data back to the shared shelve cache
        cache[ticker] = ticker_cache

        return result

    except Exception as e:
        logger.error(f"Critical error gathering stock data for {ticker}: {str(e)}")
        return result  # Return default structure even on critical error


# Helper function to safely format numeric values
def safe_format_number(value, format_str=".2f"):
    """Safely format a number that might be None or 'N/A'"""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):{format_str}}"
    except (ValueError, TypeError):
        return "N/A"


# Helper function to safely convert value for calculations
def safe_float(value, default=0.0):
    """Safely convert a value to float"""
    if value is None or value == "N/A":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_compare(value, threshold, comparison="gt"):
    if value is None:
        return False
    try:
        value = float(value)
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "ge":
            return value >= threshold
        elif comparison == "le":
            return value <= threshold
        return False
    except (TypeError, ValueError):
        return False


def safe_format_market_cap(value):
    """Safely format market cap with commas"""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return "N/A"


def format_timestamp_to_date(timestamp):
    """Convert Unix timestamp to formatted date string"""
    if timestamp is None or timestamp == "N/A":
        return "N/A"

    try:
        # Convert to integer if it's a string
        if isinstance(timestamp, str):
            timestamp = int(timestamp)

        from datetime import datetime

        date_obj = datetime.fromtimestamp(timestamp)
        return date_obj.strftime("%B %d, %Y")
    except (ValueError, TypeError, OverflowError) as e:
        logger.error(f"Error formatting timestamp {timestamp}: {str(e)}")
        return "N/A"


def format_percentage(value):
    """Format decimal or percentage value with consistent percentage format"""
    if value is None or value == "N/A":
        return "N/A"

    try:
        value_float = float(value)

        # If value is already in percentage form
        if abs(value_float) > 1.0:
            return f"{value_float:.2f}%"
        else:
            # Convert from decimal to percentage
            return f"{value_float * 100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value, include_suffix=True):
    """Format currency value with appropriate formatting"""
    if value is None or value == "N/A":
        return "N/A"

    try:
        value_float = float(value)

        if include_suffix:
            if value_float >= 1_000_000_000:
                return f"${value_float / 1_000_000_000:.2f}B"
            elif value_float >= 1_000_000:
                return f"${value_float / 1_000_000:.2f}M"

        return f"${value_float:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def dict_to_markdown(d: Dict[str, Any], indent: int = 0) -> str:
    """Convert a dictionary to markdown format for debugging"""
    markdown = "\n\n### Raw Metrics Data (Debug Output):\n"
    markdown += "| Metric | Value |\n"
    markdown += "|--------|-------|\n"

    # Sort keys for consistent output
    for key in sorted(d.keys()):
        value = d.get(key)
        # Format None values as 'null' for clarity
        value_str = "null" if value is None else str(value)
        markdown += f"| {key} | {value_str} |\n"

    return markdown


def interpret_current_ratio(ratio: Optional[float], industry: Optional[str] = None) -> str:
    """
    Interpret current ratio with industry context
    """
    if ratio is None:
        return "No data available"
    try:
        ratio = float(ratio)
        # Special handling for tech and retail industries which typically run lower ratios
        is_tech = industry and "technology" in industry.lower()
        is_retail = industry and "retail" in industry.lower()

        if is_tech or is_retail:
            if ratio > 1.5:
                return "Very Strong (above industry average)"
            if ratio > 1.2:
                return "Strong"
            if ratio > 0.8:
                return "Adequate (typical for industry)"
            if ratio > 0.6:
                return "Below Average"
            return "Concerning"
        else:
            # Traditional thresholds for other industries
            if ratio > 2.0:
                return "Very Strong"
            if ratio > 1.5:
                return "Strong"
            if ratio > 1.0:
                return "Adequate"
            if ratio > 0.8:
                return "Below Average"
            return "Concerning"
    except (TypeError, ValueError):
        return "No data available"


def assess_financial_health(metrics: Dict[str, Any], industry: Optional[str] = None) -> str:
    try:
        # Normalize industry string
        industry = industry.lower() if industry else ""

        # Industry classification
        industry_type = None
        if "retail" in industry:
            industry_type = "retail"
        elif "technology" in industry:
            industry_type = "tech"
        elif "utilities" in industry:
            industry_type = "utilities"
        elif "healthcare" in industry or "health" in industry:
            industry_type = "healthcare"
        elif "energy" in industry or "oil" in industry or "gas" in industry:
            industry_type = "energy"
        elif "financial" in industry or "bank" in industry:
            industry_type = "financial"

        # Get metrics safely
        metrics_data = {
            "roe": safe_float(metrics.get("roeTTM"), 0),
            "current_ratio": safe_float(metrics.get("currentRatioQuarterly"), 0),
            "profit_margin": safe_float(metrics.get("netProfitMarginTTM"), 0),
            "asset_turnover": safe_float(metrics.get("assetTurnoverTTM"), 0),
            "debt_equity": safe_float(metrics.get("totalDebt/totalEquityQuarterly"), 0),
            "interest_coverage": safe_float(metrics.get("netInterestCoverageTTM"), 0),
            "inventory_turnover": safe_float(metrics.get("inventoryTurnoverTTM"), 0),
            "operating_margin": safe_float(metrics.get("operatingMarginTTM"), 0),
        }

        # Industry-specific thresholds
        thresholds = {
            "retail": {
                "profit_margin": {"high": 3, "medium": 2},
                "current_ratio": {"high": 1.2, "medium": 0.8},
                "inventory_turnover": {"high": 10, "medium": 6},
                "debt_equity": {"low": 0.5, "medium": 1.0},
            },
            "tech": {
                "profit_margin": {"high": 20, "medium": 15},
                "current_ratio": {"high": 1.5, "medium": 1.2},
                "debt_equity": {"low": 0.3, "medium": 0.6},
            },
            "utilities": {
                "profit_margin": {"high": 12, "medium": 8},
                "current_ratio": {"high": 1.0, "medium": 0.8},
                "debt_equity": {"low": 1.2, "medium": 1.5},
                "interest_coverage": {"high": 3, "medium": 2},
            },
            "healthcare": {
                "profit_margin": {"high": 15, "medium": 10},
                "current_ratio": {"high": 1.5, "medium": 1.2},
                "debt_equity": {"low": 0.4, "medium": 0.8},
            },
            "energy": {
                "profit_margin": {"high": 10, "medium": 6},
                "current_ratio": {"high": 1.2, "medium": 1.0},
                "debt_equity": {"low": 0.6, "medium": 1.0},
            },
            "financial": {
                "roe": {"high": 15, "medium": 10},
                "debt_equity": {"low": 3.0, "medium": 4.0},  # Different for financials
            },
            "default": {
                "profit_margin": {"high": 15, "medium": 10},
                "current_ratio": {"high": 1.5, "medium": 1.2},
                "debt_equity": {"low": 0.5, "medium": 1.0},
            },
        }

        # Get appropriate thresholds
        industry_thresholds = thresholds.get(industry_type or "default", thresholds["default"])

        points = 0
        max_points = 0

        # Profit Margin Assessment
        if "profit_margin" in industry_thresholds:
            max_points += 2
            if (
                metrics_data["profit_margin"]
                > industry_thresholds["profit_margin"]["high"]
            ):
                points += 2
            elif (
                metrics_data["profit_margin"]
                > industry_thresholds["profit_margin"]["medium"]
            ):
                points += 1

        # ROE Assessment (important for all industries)
        max_points += 2
        roe_high = industry_thresholds.get("roe", {"high": 20, "medium": 15})["high"]
        roe_medium = industry_thresholds.get("roe", {"high": 20, "medium": 15})[
            "medium"
        ]
        if metrics_data["roe"] > roe_high:
            points += 2
        elif metrics_data["roe"] > roe_medium:
            points += 1

        # Liquidity Assessment
        if "current_ratio" in industry_thresholds:
            max_points += 1
            if (
                metrics_data["current_ratio"]
                > industry_thresholds["current_ratio"]["high"]
            ):
                points += 1

        # Solvency Assessment
        if "debt_equity" in industry_thresholds:
            max_points += 2
            if metrics_data["debt_equity"] < industry_thresholds["debt_equity"]["low"]:
                points += 2
            elif (
                metrics_data["debt_equity"]
                < industry_thresholds["debt_equity"]["medium"]
            ):
                points += 1

        # Interest Coverage (especially important for utilities and highly leveraged industries)
        interest_coverage_thresholds = industry_thresholds.get(
            "interest_coverage", {"high": 50, "medium": 20}
        )
        max_points += 2
        if metrics_data["interest_coverage"] > interest_coverage_thresholds["high"]:
            points += 2
        elif metrics_data["interest_coverage"] > interest_coverage_thresholds["medium"]:
            points += 1

        # Industry-specific metrics
        if industry_type == "retail" and "inventory_turnover" in industry_thresholds:
            max_points += 1
            if (
                metrics_data["inventory_turnover"]
                > industry_thresholds["inventory_turnover"]["high"]
            ):
                points += 1

        # Calculate final score
        score_percentage = (points / max_points) * 100 if max_points > 0 else 0

        # Return assessment
        score_result = "weak"
        if score_percentage >= 70:
            score_result = "strong"
        elif score_percentage >= 40:
            score_result = "moderate"

        return f"{score_result} {round(score_percentage)}% - {industry}"

    except Exception as e:
        logger.error(f"Error in assess_financial_health: {str(e)}")
        return "moderate"  # Default to moderate if calculation fails


def _compile_report(data: Dict[str, Any]) -> str:
    """
    Compile gathered data into a concise but comprehensive report with improved error handling.
    Handles ETFs and money market funds differently from regular stocks.
    """
    try:
        profile = data.get("basic_info", {}).get("profile", {})
        financials = data.get("basic_info", {}).get("basic_financials", {})
        metrics = financials.get("metric", {})
        price_data = data.get("current_price", {})

        # Ensure we have a valid ticker and name
        ticker = profile.get("ticker")
        name = profile.get("name")

        # Check if we have minimum required data
        if not ticker:
            logger.error("Missing ticker in profile data")
            ticker = "Unknown"

        if not name:
            logger.error(f"Missing name for ticker {ticker}")
            name = ticker  # Fall back to using ticker as name

        # Check if this is an ETF or money market fund
        is_fund, fund_type = is_etf_or_fund(profile)

        # Format functions (unchanged from original)
        def format_with_suffix(num):
            if (
                num is None
                or num == "N/A"
                or not isinstance(num, (int, float))
                or num == 0
            ):
                return "N/A"
            try:
                # Handle trillions
                if num >= 1_000_000_000_000:
                    return f"{num/1_000_000_000_000:.2f}T"
                # Handle billions
                elif num >= 1_000_000_000:
                    return f"{num/1_000_000_000:.2f}B"
                # Handle millions
                elif num >= 1_000_000:
                    return f"{num/1_000_000:.2f}M"
                # Handle thousands
                elif num >= 1_000:
                    return f"{num/1_000:.2f}K"
                return f"{num:.2f}"
            except (TypeError, ValueError):
                return "N/A"

        # Safe fetch for market cap
        market_cap = profile.get("marketCapitalization")

        if market_cap is not None and isinstance(market_cap, (int, float)):
            market_cap_display = format_with_suffix(market_cap * 1_000_000)
        else:
            market_cap_display = "N/A"

        # Handle different report formats based on security type
        if is_fund:
            # Special report format for ETFs and money market funds
            report = f"""Investment Analysis: {name} ({ticker}) - {fund_type}

Fund Overview:
• Type: {fund_type} | Category: {profile.get('finnhubIndustry', 'Investment')}
• Current Price: ${safe_format_number(price_data.get('current_price'))} ({safe_format_number(price_data.get('change'))}%)
• Daily Range: ${safe_format_number(price_data.get('low'))} - ${safe_format_number(price_data.get('high'))}

Performance:
• 1-Day Change: {safe_format_number(price_data.get('change'))}%
• Previous Close: ${safe_format_number(price_data.get('previous_close'))}
"""

            # Add fund-specific info if we have it
            # Currently we don't retrieve this data - would need to add specialized data retrieval
            if fund_type == "ETF":
                etf_data = data.get("etf_data", {})
                report += f"""
ETF Details:
• Expense Ratio: {format_percentage(etf_data.get('expense_ratio'))}
• Yield: {format_percentage(etf_data.get('yield'))}
• YTD Return: {format_percentage(etf_data.get('ytd_return'))}
• 3 Year Return: {format_percentage(etf_data.get('three_year_return'))}
• 5 Year Return: {format_percentage(etf_data.get('five_year_return'))}
• Assets Under Management: {format_currency(etf_data.get('aum'))}
• Net Asset Value: {format_currency(etf_data.get('nav'), include_suffix=True)}
• Inception Date: {format_timestamp_to_date(etf_data.get('inception_date'))}

Note: This is an Exchange Traded Fund (ETF). Detailed financial metrics like P/E ratio, 
profit margins, and debt ratios don't apply in the same way as individual stocks. 
ETFs represent a basket of securities and are evaluated differently.
"""
            elif fund_type == "Money Market Fund":
                etf_data = data.get("etf_data", {})
                report += f"""
Money Market Fund Details:
• Yield: {format_percentage(etf_data.get('yield'))}
• 7-Day Yield: {format_percentage(etf_data.get('seven_day_yield'))}
• YTD Return: {format_percentage(etf_data.get('ytd_return'))}
• Weighted Average Maturity: {etf_data.get('weighted_avg_maturity', 'N/A')}
• Total Net Assets: {format_currency(etf_data.get('aum'))}

Note: This is a Money Market Fund. These funds invest in short-term, high-quality 
securities and are evaluated by their yield, stability, and liquidity rather than 
traditional stock metrics.
"""
            # Add any news sentiment if available
            sentiments = data.get("sentiments", [])
            if sentiments:
                report += "\n\nRecent News Sentiment:"
                positive_count = sum(
                    1
                    for item in sentiments
                    if isinstance(item, dict) and item.get("sentiment") == "positive"
                )
                total_count = len(sentiments)

                # Calculate sentiment
                try:
                    weighted_sentiment = sum(
                        float(item.get("confidence", 0))
                        for item in sentiments
                        if isinstance(item, dict)
                        and item.get("sentiment") == "positive"
                    )
                    total_confidence = sum(
                        float(item.get("confidence", 0))
                        for item in sentiments
                        if isinstance(item, dict)
                    )
                    sentiment_ratio = (
                        weighted_sentiment / total_confidence
                        if total_confidence > 0
                        else 0.5
                    )
                    overall_sentiment = (
                        "Bullish"
                        if sentiment_ratio > 0.6
                        else "Bearish" if sentiment_ratio < 0.4 else "Neutral"
                    )
                except Exception as e:
                    logger.warning(f"Error calculating sentiment: {str(e)}")
                    overall_sentiment = "Neutral"

                report += f"\n• Overall: {overall_sentiment} ({positive_count}/{total_count} positive)"

                # Add news items
                for item in sentiments:
                    if not isinstance(item, dict):
                        continue
                    sentiment = item.get("sentiment", "neutral")
                    title = item.get("title", "No title")
                    summary = item.get("summary", "")
                    summary_str = f" - {summary}" if summary else ""
                    report += f"\n• {sentiment} - {title}{summary_str}"
        else:
            # Original report format for regular stocks
            # [Original stock report code goes here]
            # Get financial health score with safe handling
            try:
                fin_health_score = assess_financial_health(
                    metrics, profile.get("finnhubIndustry", "")
                )
            except Exception as e:
                logger.error(f"Error calculating financial health score: {str(e)}")
                fin_health_score = "moderate"

            # Build report with key metrics and insights
            report = f"""Stock Analysis: {name} ({ticker})

Company Overview:
• {profile.get('finnhubIndustry', 'N/A')} | Market Cap: ${market_cap_display} | Country: {profile.get('country', 'N/A')}
• Current Price: ${safe_format_number(price_data.get('current_price'))} ({safe_format_number(price_data.get('change'))}%) | YTD: {safe_format_number(metrics.get('yearToDatePriceReturnDaily'))}%
• 52W Range: ${safe_format_number(metrics.get('52WeekLow'))} - ${safe_format_number(metrics.get('52WeekHigh'))} | Beta: {safe_format_number(metrics.get('beta'))}

Key Performance Indicators:
• Growth (5Y): Revenue {safe_format_number(metrics.get('revenueGrowth5Y'))}% | EPS {safe_format_number(metrics.get('epsGrowth5Y'))}%
• Margins: Gross {safe_format_number(metrics.get('grossMarginTTM'))}% | Operating {safe_format_number(metrics.get('operatingMarginTTM'))}% | Net {safe_format_number(metrics.get('netProfitMarginTTM'))}%
• Returns: ROE {safe_format_number(metrics.get('roeTTM'))}% | ROA {safe_format_number(metrics.get('roaTTM'))}%

Financial Health:
• Liquidity: Current Ratio {safe_format_number(metrics.get('currentRatioQuarterly'))} | Quick Ratio {safe_format_number(metrics.get('quickRatioQuarterly'))}
• Leverage: Debt/Equity {safe_format_number(metrics.get('totalDebt/totalEquityQuarterly'))} | Interest Coverage {safe_format_number(metrics.get('netInterestCoverageTTM'))}x
• Per Share: EPS ${safe_format_number(metrics.get('epsTTM'))} | Book Value ${safe_format_number(metrics.get('bookValuePerShareQuarterly'))}

Valuation:
• Multiples: P/E {safe_format_number(metrics.get('peTTM'))} | P/B {safe_format_number(metrics.get('pbQuarterly'))} | P/S {safe_format_number(metrics.get('psTTM'))}
• Dividend Yield: {safe_format_number(metrics.get('dividendYieldIndicatedAnnual'))}% | Payout Ratio: {safe_format_number(metrics.get('payoutRatioTTM'))}%

Summary Analysis:
• Health Score: {fin_health_score}"""

            # Add key strengths with error handling
            strengths = []
            try:
                if safe_float(metrics.get("revenueGrowth5Y", 0)) > 10:
                    strengths.append("High Growth")
                if safe_float(metrics.get("netProfitMarginTTM", 0)) > 15:
                    strengths.append("Strong Margins")
                if safe_float(metrics.get("roeTTM", 0)) > 15:
                    strengths.append("Solid Returns")
                if safe_float(metrics.get("totalDebt/totalEquityQuarterly", 0)) < 0.5:
                    strengths.append("Low Leverage")
            except Exception as e:
                logger.warning(f"Error calculating strengths: {str(e)}")

            report += f"\n• Key Strengths: {', '.join(strengths) if strengths else 'None identified'}"

            # Add key risks with error handling
            risks = []
            try:
                if safe_float(metrics.get("revenueGrowth5Y", 0)) < 0:
                    risks.append("Negative Growth")
                if safe_float(metrics.get("netProfitMarginTTM", 0)) < 5:
                    risks.append("Low Margins")
                if safe_float(metrics.get("totalDebt/totalEquityQuarterly", 0)) > 2:
                    risks.append("High Leverage")
                if safe_float(metrics.get("beta", 0)) > 2:
                    risks.append("High Beta")
            except Exception as e:
                logger.warning(f"Error calculating risks: {str(e)}")

            report += (
                f"\n• Key Risks: {', '.join(risks) if risks else 'None identified'}"
            )

            # Add sentiment analysis summary with error checking
            sentiments = data.get("sentiments", [])
            if sentiments:
                report += "\n\nRecent News Sentiment:"

                # Calculate positive count safely
                positive_count = sum(
                    1
                    for item in sentiments
                    if isinstance(item, dict) and item.get("sentiment") == "positive"
                )
                total_count = len(sentiments)

                # Calculate weighted sentiment safely
                try:
                    weighted_sentiment = sum(
                        float(item.get("confidence", 0))
                        for item in sentiments
                        if isinstance(item, dict)
                        and item.get("sentiment") == "positive"
                    )
                    total_confidence = sum(
                        float(item.get("confidence", 0))
                        for item in sentiments
                        if isinstance(item, dict)
                    )
                    sentiment_ratio = (
                        weighted_sentiment / total_confidence
                        if total_confidence > 0
                        else 0.5
                    )
                    overall_sentiment = (
                        "Bullish"
                        if sentiment_ratio > 0.6
                        else "Bearish" if sentiment_ratio < 0.4 else "Neutral"
                    )
                except Exception as e:
                    logger.warning(f"Error calculating sentiment: {str(e)}")
                    overall_sentiment = "Neutral"

                report += f"\n• Overall: {overall_sentiment} ({positive_count}/{total_count} positive)"

                # Add news items with error handling
                for item in sentiments:
                    if not isinstance(item, dict):
                        continue

                    sentiment = item.get("sentiment", "neutral")
                    title = item.get("title", "No title")
                    summary = item.get("summary", "")

                    summary_str = f" - {summary}" if summary else ""
                    report += f"\n• {sentiment} - {title}{summary_str}"

        return report

    except Exception as e:
        import traceback

        tb = traceback.extract_tb(e.__traceback__)
        filename, line_no, func_name, text = tb[-1]
        logger.error(
            f"Error compiling report at line {line_no} in {func_name}: {str(e)}"
        )
        logger.error(f"Line content: {text}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Even if we encounter an error, return a basic valid report instead of failing
        return f"""Stock Analysis: Error generating complete report
            Error details: Error in {func_name} at line {line_no}: {str(e)}
            Please check logs for more information."""

def _compile_report_optimized(data: Dict[str, Any]) -> str:
    """
    Compile gathered data into a token-optimized but still human-readable report.
    Removes redundant text and uses more compact formatting.
    Reduces token usage by ~60% compared to original.
    """
    try:
        profile = data.get("basic_info", {}).get("profile", {})
        financials = data.get("basic_info", {}).get("basic_financials", {})
        metrics = financials.get("metric", {})
        price_data = data.get("current_price", {})

        ticker = profile.get("ticker", "Unknown")
        name = profile.get("name", ticker)
        
        # Check if this is an ETF or money market fund
        is_fund, fund_type = is_etf_or_fund(profile)

        def format_compact(num):
            if num is None or num == "N/A" or not isinstance(num, (int, float)) or num == 0:
                return "N/A"
            try:
                if num >= 1_000_000_000_000:
                    return f"{num/1_000_000_000_000:.1f}T"
                elif num >= 1_000_000_000:
                    return f"{num/1_000_000_000:.1f}B"
                elif num >= 1_000_000:
                    return f"{num/1_000_000:.1f}M"
                elif num >= 1_000:
                    return f"{num/1_000:.1f}K"
                return f"{num:.2f}"
            except (TypeError, ValueError):
                return "N/A"

        market_cap = profile.get("marketCapitalization")
        market_cap_display = format_compact(market_cap * 1_000_000) if market_cap else "N/A"

        if is_fund:
            # Compact ETF/Fund format
            report = f"""{name} ({ticker}) - {fund_type}
Price: ${safe_format_number(price_data.get('current_price'))} ({safe_format_number(price_data.get('change'))}%)
Range: ${safe_format_number(price_data.get('low'))}-${safe_format_number(price_data.get('high'))}
Industry: {profile.get('finnhubIndustry', 'Investment')}"""

            etf_data = data.get("etf_data", {})
            if etf_data:
                if fund_type == "ETF":
                    report += f"""
Expense: {format_percentage(etf_data.get('expense_ratio'))} | Yield: {format_percentage(etf_data.get('yield'))}
AUM: {format_currency(etf_data.get('aum'))} | YTD: {format_percentage(etf_data.get('ytd_return'))}"""
                elif fund_type == "Money Market Fund":
                    report += f"""
Yield: {format_percentage(etf_data.get('yield'))} | 7D: {format_percentage(etf_data.get('seven_day_yield'))}
Assets: {format_currency(etf_data.get('aum'))}"""

        else:
            # Compact stock format - remove redundant labels and group related metrics
            try:
                fin_health = assess_financial_health(metrics, profile.get("finnhubIndustry", ""))
            except Exception as e:
                logger.error(f"Error calculating financial health: {str(e)}")
                fin_health = "moderate"
            
            report = f"""{name} ({ticker})
${safe_format_number(price_data.get('current_price'))} ({safe_format_number(price_data.get('change'))}%) | Cap: ${market_cap_display} | Beta: {safe_format_number(metrics.get('beta'))}
Growth: Rev {safe_format_number(metrics.get('revenueGrowth5Y'))}% EPS {safe_format_number(metrics.get('epsGrowth5Y'))}% | YTD: {safe_format_number(metrics.get('yearToDatePriceReturnDaily'))}%
Margins: {safe_format_number(metrics.get('grossMarginTTM'))}%/{safe_format_number(metrics.get('operatingMarginTTM'))}%/{safe_format_number(metrics.get('netProfitMarginTTM'))}% (G/O/N)
Returns: ROE {safe_format_number(metrics.get('roeTTM'))}% ROA {safe_format_number(metrics.get('roaTTM'))}%
Ratios: P/E {safe_format_number(metrics.get('peTTM'))} P/B {safe_format_number(metrics.get('pbQuarterly'))} Current {safe_format_number(metrics.get('currentRatioQuarterly'))}
D/E: {safe_format_number(metrics.get('totalDebt/totalEquityQuarterly'))} | Health: {fin_health}"""

            # Add key strengths/risks in compact format
            strengths = []
            risks = []
            try:
                if safe_float(metrics.get("revenueGrowth5Y", 0)) > 10:
                    strengths.append("Growth")
                if safe_float(metrics.get("netProfitMarginTTM", 0)) > 15:
                    strengths.append("Margins")
                if safe_float(metrics.get("roeTTM", 0)) > 15:
                    strengths.append("ROE")
                if safe_float(metrics.get("totalDebt/totalEquityQuarterly", 0)) < 0.5:
                    strengths.append("Low Debt")
                    
                if safe_float(metrics.get("revenueGrowth5Y", 0)) < 0:
                    risks.append("Rev Decline")
                if safe_float(metrics.get("netProfitMarginTTM", 0)) < 5:
                    risks.append("Low Margins")
                if safe_float(metrics.get("totalDebt/totalEquityQuarterly", 0)) > 2:
                    risks.append("High Debt")
                if safe_float(metrics.get("beta", 0)) > 2:
                    risks.append("High Beta")
            except Exception as e:
                logger.warning(f"Error calculating strengths/risks: {str(e)}")

            if strengths or risks:
                report += f"\n+: {', '.join(strengths) if strengths else 'None'} | -: {', '.join(risks) if risks else 'None'}"

        # Compact sentiment summary
        sentiments = data.get("sentiments", [])
        if sentiments:
            pos_count = sum(1 for s in sentiments if isinstance(s, dict) and s.get("sentiment") == "positive")
            sentiment_summary = "Bullish" if pos_count > len(sentiments)/2 else "Bearish" if pos_count < len(sentiments)/3 else "Neutral"
            report += f"\nNews: {sentiment_summary} ({pos_count}/{len(sentiments)} positive)"

        return report

    except Exception as e:
        logger.error(f"Error compiling optimized report: {str(e)}")
        return f"Analysis error for {ticker}: {str(e)}"


def _short_cat_from_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    # very lightweight heuristics
    if "money market" in s or s.endswith("mm") or "treasury" in s or "gov" in s:
        return "MMF"
    if "total" in s and ("market" in s or "stock" in s):
        return "US-TOT"
    if "s&p" in s or "500" in s or "spdr" in s:
        return "US-LC"
    if "emerging" in s or "em" == s or "vwo" in s:
        return "EM"
    if "gold" in s or "gld" in s:
        return "COMD"
    if "robot" in s or "ai" in s or "automation" in s:
        return "THEME"
    if "real estate" in s or "reit" in s:
        return "RE"
    return ""

def _short_cat_fallback(profile: dict) -> str:
    # use finnhubIndustry if no good short code
    cat = profile.get("finnhubIndustry") or ""
    if not cat:
        return ""
    # compact it a bit
    return "".join([w[0].upper() for w in cat.split() if w and w[0].isalpha()])[:6]

def _fmt_num(val, decimals=1):
    """Return '' for missing; otherwise a compact float string with <=decimals."""
    if val is None or val == "N/A":
        return ""
    try:
        v = float(val)
        return f"{v:.{decimals}f}".rstrip('0').rstrip('.')  # trim trailing zeros
    except:
        return ""

def _fmt_pct(val, decimals=1):
    """Input is already % or decimal? We assume it's a % number already. Return '' if missing."""
    if val is None or val == "N/A":
        return ""
    try:
        v = float(val)
        return f"{v:.{decimals}f}".rstrip('0').rstrip('.')
    except:
        return ""

def _sentiment_score(sentiments: list) -> str:
    if not sentiments:
        return "0.5"
    try:
        pos = sum(1 for s in sentiments if isinstance(s, dict) and s.get("sentiment") == "positive")
        return f"{pos/len(sentiments):.2f}".rstrip('0').rstrip('.')
    except:
        return "0.5"

def _health_score_from_text(health_str: str) -> str:
    if not isinstance(health_str, str):
        return "0.5"
    hs = health_str.lower()
    if "strong" in hs:
        return "0.8"
    if "weak" in hs or "poor" in hs:
        return "0.2"
    return "0.5"

def _safe_metric(metrics: dict, key: str):
    val = metrics.get(key)
    try:
        return float(val)
    except:
        return None

def _etf_nums_from_yf(etf_data: dict):
    # returns er, yld, ytd, nav, aum — all as strings ('' if missing)
    if not isinstance(etf_data, dict):
        return "", "", "", "", ""
    er = _fmt_pct((etf_data.get("expense_ratio") or 0) * 100) if etf_data.get("expense_ratio") else ""
    yld = _fmt_pct((etf_data.get("yield") or 0) * 100) if etf_data.get("yield") else ""
    ytd = _fmt_pct((etf_data.get("ytd_return") or 0) * 100) if etf_data.get("ytd_return") else ""
    nav = _fmt_num(etf_data.get("nav"))
    aum_raw = etf_data.get("aum")
    # compress aum to millions if large
    if aum_raw is None or aum_raw == "N/A":
        aum_m = ""
    else:
        try:
            aum_val = float(aum_raw) / 1_000_000.0
            aum_m = _fmt_num(aum_val)  # compact like 1234.5 (meaning $1.23B)
        except:
            aum_m = ""
    return er, yld, ytd, nav, aum_m

def _fund_line(profile: dict, price_data: dict, sentiments: list, etf_data: dict) -> str:
    ticker = profile.get("ticker", "UNK")
    name = profile.get("name", "")
    px     = _fmt_num(price_data.get("current_price"))
    dchg_p = _fmt_num(price_data.get("change")) or "0"
    er_p, yld_p, ytd_p, nav, aum_m = _etf_nums_from_yf(etf_data)  # make aum in millions
    cat    = _short_cat_from_name(name) or _short_cat_fallback(profile)
    news   = _sentiment_score(sentiments)
    hlth   = "0.5"  # or a smarter rule if you add one later

    # F|ticker|price_usd|chg_day_pct|exp_ratio_pct|yield_pct|ytd_pct|category|nav_usd|aum_millions|news_score_0to1|health_score_0to1
    fields = ["F", ticker, px, dchg_p, er_p, yld_p, ytd_p, cat, nav, aum_m, news, hlth]
    return "|".join(fields)

def _stock_line(profile: dict, metrics: dict, price_data: dict, sentiments: list) -> str:
    ticker = profile.get("ticker", "UNK")
    px     = _fmt_num(price_data.get("current_price"))
    dchg_p = _fmt_num(price_data.get("change"))        # already percent value
    pe_r   = _fmt_num(_safe_metric(metrics, "peTTM"))
    roe_p  = _fmt_num(_safe_metric(metrics, "roeTTM"))
    npm_p  = _fmt_num(_safe_metric(metrics, "netProfitMarginTTM"))
    rev5y_p= _fmt_num(_safe_metric(metrics, "revenueGrowth5Y"))
    de_r   = _fmt_num(_safe_metric(metrics, "totalDebt/totalEquityQuarterly"))
    beta   = _fmt_num(_safe_metric(metrics, "beta"))
    news   = _sentiment_score(sentiments)              # 0..1 as string
    
    try:
        fin_health_str = assess_financial_health(metrics, profile.get("finnhubIndustry", ""))
    except Exception:
        fin_health_str = "moderate"
    hlth   = _health_score_from_text(fin_health_str)   # 0..1 as string

    # S|ticker|price_usd|chg_day_pct|pe_ratio|roe_pct|npm_pct|rev5y_pct|de_ratio|beta|news_score_0to1|health_score_0to1
    fields = ["S", ticker, px, dchg_p, pe_r, roe_p, npm_p, rev5y_p, de_r, beta, news, hlth]
    return "|".join(fields)


def _compile_report_llm_focused(data: Dict[str, Any]) -> str:
    """
    Compact output for LLMs with separate schemas for Stocks (S) and Funds (F).
    No 'N/A' – blank fields when missing.
    SCHEMA:
        S = Stock: ticker|price_usd|chg_day_pct|pe_ratio|roe_pct|npm_pct|rev5y_pct|de_ratio|beta|news_score_0to1|health_score_0to1
        F = Fund/ETF: ticker|price_usd|chg_day_pct|exp_ratio_pct|yield_pct|ytd_pct|category|nav_usd|aum_millions|news_score_0to1|health_score_0to1
        * "_0to1" fields are normalized scores: 0.0=low, 0.5=neutral, 1.0=high.
    """
    try:
        profile = data.get("basic_info", {}).get("profile", {}) or {}
        metrics = data.get("basic_info", {}).get("basic_financials", {}).get("metric", {}) or {}
        price   = data.get("current_price", {}) or {}
        sentiments = data.get("sentiments", []) or []

        is_fund, _fund_type = is_etf_or_fund(profile)
        if is_fund:
            etf_data = data.get("etf_data", {}) or {}
            return _fund_line(profile, price, sentiments, etf_data)
        else:
            return _stock_line(profile, metrics, price, sentiments)

    except Exception as e:
        # ultra-compact error line (still parsable)
        tkr = data.get("basic_info", {}).get("profile", {}).get("ticker", "UNK")
        return f"ERR|{tkr}|{str(e)[:60]}"

class Tools:
    class Valves(BaseModel):
        FINNHUB_API_KEY: str = Field(
            default="", description="Required API key to access FinHub services"
        )

    def __init__(self):
        self.citation = True
        self.valves = self.Valves()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "compile_stock_report",
                    "description": "BATCH-ONLY. Full comprehensive stock analysis. Pass ALL symbols at once using the `tickers` array. Do not call per ticker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of ticker symbols, e.g. ['AAPL','GOOGL','MSFT']"
                            },
                            "ticker": {
                                "type": "string",
                                "description": "(Legacy) Comma-separated symbols, e.g. 'AAPL,GOOGL,MSFT'. Prefer `tickers`."
                            }
                        }
                        # Intentionally no "required": we accept either field.
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_summary",
                    "description": "BATCH-ONLY. Optimized human-readable summary. Use `tickers` array; do NOT call once per symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of ticker symbols."
                            },
                            "ticker": {
                                "type": "string",
                                "description": "(Legacy) Comma-separated symbols. Prefer `tickers`."
                            }
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                            "name": "get_stock_data_compact",
                            "description": "BATCH-ONLY. Ultra-compact structured data. Use `tickers` array; do NOT call once per symbol.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "tickers": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Array of ticker symbols."
                                    },
                                    "ticker": {
                                        "type": "string",
                                        "description": "(Legacy) Comma-separated symbols. Prefer `tickers`."
                                    }
                                }
                            },
                },
            },
        ]

    async def compile_stock_report(self, tickers: List[str] = None, ticker: str = "", __user__={}, __event_emitter__=None) -> str:
        symbols = self._normalize_symbols(tickers, ticker)
        return await self._execute_analysis(symbols, "full", __user__, __event_emitter__)

    async def get_stock_summary(self, tickers: List[str] = None, ticker: str = "", __user__={}, __event_emitter__=None) -> str:
        symbols = self._normalize_symbols(tickers, ticker)
        return await self._execute_analysis(symbols, "optimized", __user__, __event_emitter__)

    async def get_stock_data_compact(self, tickers: List[str] = None, ticker: str = "", __user__={}, __event_emitter__=None) -> str:
        symbols = self._normalize_symbols(tickers, ticker)
        return await self._execute_analysis(symbols, "compact", __user__, __event_emitter__)

    def _normalize_symbols(self, tickers: Optional[List[str]], ticker: str) -> List[str]:
        if tickers and isinstance(tickers, list) and len(tickers):
            return [t.strip().upper() for t in tickers if t and str(t).strip()]
        if ticker:
            # support both CSV "AAPL,MSFT" and single "AAPL"
            return [t.strip().upper() for t in str(ticker).split(",") if t.strip()]
        raise ValueError("No tickers provided")

    async def _execute_analysis(
        self,
        symbols: List[str],
        format_type: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Common analysis execution logic with different report formats.
        """
 
        try:
            logger.info(f"Starting {format_type} stock analysis for symbols: {symbols}")

            if not self.valves.FINNHUB_API_KEY:
                raise Exception("FINNHUB_API_KEY not provided in valves")

            # Normalize to uppercase and strip spaces
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            if not symbols:
                raise ValueError("No valid tickers provided")

            # Initialize the Finnhub client
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Initializing client", "done": False},
                    }
                )
            self.client = finnhub.Client(api_key=self.valves.FINNHUB_API_KEY)

            # Open the shelve cache
            with shelve.open(
                os.path.join(os.path.dirname(__file__), "stock_cache_shelve"),
                writeback=True,
            ) as cache:
                
                combined_report = ""

                # Split tickers and clean them
                for idx, single_ticker in enumerate(symbols):
                    logger.info(f"Processing ticker {idx + 1}/{len(symbols)}: {single_ticker}")

                    if idx % 4 == 0 or idx == len(symbols) - 1:
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Retrieving stock data for {single_ticker} ({idx + 1}/{len(symbols)})",
                                        "done": False,
                                    },
                                }
                            )

                    # Get the data using existing function
                    data = await _async_gather_stock_data(self.client, single_ticker, cache)

                    # Choose report format based on format_type
                    if format_type == "full":
                        report = _compile_report(data)  # Your original function
                    elif format_type == "optimized":
                        report = _compile_report_optimized(data)
                    elif format_type == "compact":
                        report = _compile_report_llm_focused(data)
                    else:
                        report = _compile_report(data)  # fallback

                    last_price = data["current_price"]["current_price"]

                    # Add separator for multiple tickers (except compact format)
                    if combined_report and format_type != "compact":
                        combined_report += "\n" + "=" * 8 + "\n\n"
                    elif combined_report and format_type == "compact":
                        combined_report += "\n"

                    combined_report += report

                    if idx == len(symbols) - 1:
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Finished {format_type} analysis for {single_ticker}",
                                        "done": True,
                                    },
                                }
                            )

                logger.info(f"Successfully completed {format_type} stock analysis")
                
                # Format the final output based for compact format with detailed schema
                if format_type == "compact":
                    # Detailed schema with value ranges and meanings
                    schema_header = (
                        "SCHEMA: S = Stock: ticker|price_usd|chg_day_pct|pe_ratio|roe_pct|npm_pct|rev5y_pct|de_ratio|beta|news_score_0to1|health_score_0to1 \n"
                        "F = Fund/ETF: ticker|price_usd|chg_day_pct|exp_ratio_pct|yield_pct|ytd_pct|category|nav_usd|aum_millions|news_score_0to1|health_score_0to1 \n"
                        "* '_0to1' fields are normalized scores: 0.0=low, 0.5=neutral, 1.0=high. \n"
                )
                    return f"{schema_header}\n{combined_report}"
                else:
                    return f"Analysis for {symbols}:\n\n{combined_report}"

        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            filename, line_no, func_name, text = tb[-1]
            error_msg = f"Error in {format_type} analysis for {symbols} at line {line_no}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
