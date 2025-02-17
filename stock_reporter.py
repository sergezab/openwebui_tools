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


def _get_basic_info(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company information from Finnhub API.
    Handles failures gracefully by providing default values.
    """
    result = {
        "profile": {"ticker": ticker},  # Minimal profile with just the ticker
        "basic_financials": {"metric": {}},  # Empty financials
        "peers": [],  # Empty peers list
    }

    try:
        # Try to get profile
        try:
            profile = client.company_profile2(symbol=ticker)
            if profile:
                result["profile"] = profile
        except Exception as e:
            logger.warning(
                f"Could not fetch profile for {ticker}, using minimal profile: {str(e)}"
            )

        # Try to get financials
        try:
            basic_financials = client.company_basic_financials(ticker, "all")
            if basic_financials and "metric" in basic_financials:
                result["basic_financials"] = basic_financials
        except Exception as e:
            logger.warning(
                f"Could not fetch financials for {ticker}, using empty financials: {str(e)}"
            )

        # Try to get peers
        try:
            peers = client.company_peers(ticker)
            if peers:
                result["peers"] = peers
        except Exception as e:
            logger.warning(
                f"Could not fetch peers for {ticker}, using empty peers list: {str(e)}"
            )

        return result

    except Exception as e:
        logger.error(f"Critical error in _get_basic_info for {ticker}: {str(e)}")
        return result  # Return default structure even on critical error


def _get_current_price(client: finnhub.Client, ticker: str) -> Dict[str, float]:
    """
    Fetch current price and daily change from Finnhub API.
    Returns default values if API call fails.
    """
    try:
        quote = client.quote(ticker)
        return {
            "current_price": quote["c"],
            "change": quote["dp"],
            "change_amount": quote["d"],
            "high": quote["h"],
            "low": quote["l"],
            "open": quote["o"],
            "previous_close": quote["pc"],
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
) -> Dict[str, Union[str, float]]:
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

        logger.info(
            f"Keyword-based sentiment: {base_sentiment} (confidence: {base_confidence})"
        )

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

        logger.info(f"Combined text: {combined_text}")
        # logger.info(f"Logits: {outputs.logits.tolist()}")
        # logger.info(f"Scaled logits: {scaled_logits.tolist()}")
        # logger.info(f"Probabilities: {sentiment_scores}")
        logger.info(
            f"Model sentiment: {model_sentiment} (confidence: {model_confidence})"
        )

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


async def _async_gather_stock_data(
    client: finnhub.Client, ticker: str, cache: shelve.Shelf
) -> Dict[str, Any]:
    """
    Gather all stock data with graceful error handling.
    Returns minimal valid data structure even if some parts fail.
    Caches current price if within 2 hours of request or outside trading hours.
    """
    try:
        # Get the ticker's cache data; if not present, start with an empty dict
        ticker_cache = cache.get(ticker, {})

        # Initialize result with default values
        result = {
            "basic_info": {
                "profile": {"ticker": ticker},
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
        }

        try:
            # Get basic info (use cache if valid)
            if not _is_cache_valid(ticker_cache, "basic_info"):
                logger.info(f"Fetching fresh basic info for {ticker}")
                basic_info = _get_basic_info(client, ticker)
                # Check if the basic info looks valid (e.g. has a 'name')
                if basic_info.get("profile", {}).get("name"):
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
                    # Optionally, if previous cached data exists, you might keep that.
                    basic_info = ticker_cache.get("basic_info", {}).get(
                        "data", basic_info
                    )
            else:
                logger.info(f"Using cached basic info for {ticker}")
                basic_info = ticker_cache["basic_info"]["data"]
            result["basic_info"] = basic_info
        except Exception as e:
            logger.error(f"Error getting basic info for {ticker}: {str(e)}")

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
                # Get fresh price data
                logger.info(f"Fetching fresh price for {ticker}")
                current_price = _get_current_price(client, ticker)
                if current_price is None:
                    # If the API call failed (e.g. due to rate limit), log and optionally update status.
                    error_msg = f"Failed to fetch fresh price for {ticker}. Using cached price if available."
                    logger.error(error_msg)
                    # Optionally: await __event_emitter__({...})
                    if "current_price" in ticker_cache:
                        current_price = ticker_cache["current_price"]["data"]
                        logger.info(f"Using previously cached price for {ticker}.")
                    else:
                        # No cached price available; use a fallback value or decide how to handle it.
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
            result["current_price"] = current_price

        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")

        try:
            # Get news and process sentiments
            if not _is_cache_valid(ticker_cache, "sentiments"):
                logger.info(f"Processing fresh news and sentiments for {ticker}")
                try:
                    news_items = _get_company_news(client, ticker)
                    if news_items:

                        effective_summaries = [
                            (
                                ""
                                if is_similar(item["title"], item["summary"])
                                else item["summary"]
                            )
                            for item in news_items
                        ]

                        # Create sentiment analysis tasks directly using the summary and title
                        # sentiment_tasks = [
                        #    _async_sentiment_analysis(effective_summary, item["title"])
                        #    for effective_summary, item in zip(
                        #        effective_summaries, news_items
                        #    )
                        # ]
                        # sentiments = await asyncio.gather(*sentiment_tasks)

                        # Wrap the CPU-bound work in a thread
                        sentiment_tasks = [
                            _async_sentiment_analysis(effective_summary, item["title"])
                            for effective_summary, item in zip(
                                effective_summaries, news_items
                            )
                        ]
                        sentiments = await asyncio.gather(*sentiment_tasks)

                        sentiment_results = [
                            {
                                "url": item["url"],
                                "title": item["title"],
                                "summary": effective_summary,
                                "sentiment": sentiment["sentiment"],
                                "confidence": sentiment["confidence"],
                                "probabilities": sentiment["probabilities"],
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
            else:
                logger.info(f"Using cached sentiments for {ticker}")
                result["sentiments"] = ticker_cache["sentiments"]["data"]
        except Exception as e:
            logger.error(f"Error in sentiment processing for {ticker}: {str(e)}")

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


def interpret_current_ratio(ratio: float, industry: str = None) -> str:
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


def assess_financial_health(metrics: Dict[str, Any], industry: str = None) -> str:
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
        industry_thresholds = thresholds.get(industry_type, thresholds["default"])

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
    Compile gathered data into a concise but comprehensive report.
    """
    try:
        profile = data["basic_info"]["profile"]
        financials = data["basic_info"]["basic_financials"]
        metrics = financials.get("metric", {})
        price_data = data["current_price"]

        def format_with_suffix(num):
            if num is None or num == 0:
                return "N/A"
            if num >= 1_000_000:
                return f"{num/1_000_000:.2f}B"
            elif num >= 1_000:
                return f"{num/1_000:.2f}M"
            return f"{num:.2f}"

        fin_health_score = assess_financial_health(
            metrics, profile.get("finnhubIndustry", "")
        )

        # Build report with key metrics and insights
        report = f"""Stock Analysis: {profile.get('name')} ({profile.get('ticker')})

Company Overview:
• {profile.get('finnhubIndustry', 'N/A')} | Market Cap: ${format_with_suffix(profile.get('marketCapitalization', 0) * 1_000_000)} | Country: {profile.get('country', 'N/A')}
• Current Price: ${safe_format_number(price_data['current_price'])} ({safe_format_number(price_data['change'])}%) | YTD: {safe_format_number(metrics.get('yearToDatePriceReturnDaily'))}%
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
• Health Score: {fin_health_score}
• Key Strengths: {', '.join([s for s in [
    'High Growth' if safe_float(metrics.get('revenueGrowth5Y', 0)) > 10 else None,
    'Strong Margins' if safe_float(metrics.get('netProfitMarginTTM', 0)) > 15 else None,
    'Solid Returns' if safe_float(metrics.get('roeTTM', 0)) > 15 else None,
    'Low Leverage' if safe_float(metrics.get('totalDebt/totalEquityQuarterly', 0)) < 0.5 else None
    ] if s is not None]) or 'None identified'}
• Key Risks: {', '.join([r for r in [
    'Negative Growth' if safe_float(metrics.get('revenueGrowth5Y', 0)) < 0 else None,
    'Low Margins' if safe_float(metrics.get('netProfitMarginTTM', 0)) < 5 else None,
    'High Leverage' if safe_float(metrics.get('totalDebt/totalEquityQuarterly', 0)) > 2 else None,
    'High Beta' if safe_float(metrics.get('beta', 0)) > 2 else None
    ] if r is not None]) or 'None identified'}

Recent News Sentiment:"""

        # Add sentiment analysis summary
        positive_count = sum(
            1 for item in data["sentiments"] if item["sentiment"] == "positive"
        )
        negative_count = sum(
            1 for item in data["sentiments"] if item["sentiment"] == "negative"
        )
        total_count = len(data["sentiments"])

        # Updated sentiment summary logic
        if total_count > 0:
            weighted_sentiment = sum(
                float(item["confidence"])
                for item in data["sentiments"]
                if item["sentiment"] == "positive"
            )
            total_confidence = sum(
                float(item["confidence"]) for item in data["sentiments"]
            )
            sentiment_ratio = (
                weighted_sentiment / total_confidence if total_confidence > 0 else 0.5
            )
            overall_sentiment = (
                "Bullish"
                if sentiment_ratio > 0.6
                else "Bearish" if sentiment_ratio < 0.4 else "Neutral"
            )
            report += f"\n• Overall: {overall_sentiment} ({positive_count}/{total_count} positive)"

            # Add all recent news
            for item in data["sentiments"]:
                title = item["title"]
                summary = item["summary"]
                if summary:
                    summary_str = f" - {summary}"
                else:
                    summary_str = ""
                report += f"\n• {item['sentiment']} - {title}{summary_str}"

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
        raise


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
                    "description": "Compile a comprehensive stock analysis report using Finnhub data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol(s) to analyze (e.g., 'AAPL' or 'AAPL,GOOGL,MSFT')",
                            }
                        },
                        "required": ["ticker"],
                    },
                },
            }
        ]

    async def compile_stock_report(
        self,
        ticker: str,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Perform a comprehensive stock analysis and compile a detailed report for given ticker(s).

        :param ticker: The stock ticker symbol(s) as a string (e.g., "AAPL" or "AAPL,GOOGL,MSFT") or list of strings.
        :return: A comprehensive analysis report of the stock(s).
        """
        ticker_query = ""
        try:
            logger.info(f"Starting stock report compilation for ticker: {ticker}")

            if not self.valves.FINNHUB_API_KEY:
                raise Exception("FINNHUB_API_KEY not provided in valves")

            # Handle different input types for ticker
            if isinstance(ticker, list):
                logger.info(f"Converting ticker list to string: {ticker}")
                ticker_str = ",".join(str(t) for t in ticker)
            else:
                logger.info(f"Using ticker string directly: {ticker}")
                ticker_str = str(ticker)

            ticker_query = ticker_str

            # Initialize the Finnhub client
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
                # Split tickers and clean them
                tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]
                if not tickers:
                    raise ValueError("No valid tickers provided")
                logger.info(f"Processing tickers: {tickers}")

                combined_report = ""

                # Process each ticker
                for idx, single_ticker in enumerate(tickers):
                    logger.info(
                        f"Processing ticker {idx + 1}/{len(tickers)}: {single_ticker}"
                    )

                    if idx % 4 == 0 or idx == len(tickers) - 1:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Retrieving stock data for {single_ticker} ({idx + 1}/{len(tickers)})",
                                    "done": False,
                                },
                            }
                        )

                    # Pass the shared shelve cache to the data gathering function
                    data = await _async_gather_stock_data(
                        self.client, single_ticker, cache
                    )

                    report = _compile_report(data)
                    last_price = data["current_price"]["current_price"]

                    # Add separator between reports if this isn't the first report
                    # if combined_report:
                    combined_report += "\n" + "=" * 8 + "\n\n"

                    combined_report += report

                    if idx == len(tickers) - 1:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Finished report for {single_ticker} - latest price: {str(last_price)}",
                                    "done": idx == len(tickers) - 1,
                                },
                            }
                        )

                logger.info("Successfully completed stock report compilation")
                return f"Tickers {tickers}\n\n Combined Report:\n{combined_report}"

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            filename, line_no, func_name, text = tb[-1]
            error_msg = (
                f"Error in compile_stock_report for tickers {ticker_query} at line {line_no}: {str(e)}\n"
                f"Line content: {text}"
            )
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
