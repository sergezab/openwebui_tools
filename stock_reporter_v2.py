"""
title: Stock Market Helper (Refactored)
description: A comprehensive stock analysis tool that gathers data from Finnhub and Yahoo Finance APIs, with robust caching and multiple report formats.
author: Sergii Zabigailo (Refactored by Gemini)
author_url: https://github.com/sergezab/
github: https://github.com/sergezab/openwebui_tools/
version: 0.3.1
license: MIT
requirements: finnhub-python,pytz,transformers,torch,pydantic,aiohttp,yfinance
"""

import asyncio
import difflib
import json
import logging
import os
import random
import shelve
import time
import traceback
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import (Any, Awaitable, Callable, Dict, List, Optional, Tuple,
                    TypedDict, Union)

import aiohttp
import finnhub
import pytz
import torch
import yfinance as yf
from pydantic import BaseModel, Field
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

# region: --- Logging Configuration ---
# ==============================================================================
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if the script is reloaded
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Setup file handler
    try:
        log_dir = os.path.expanduser("~")
        log_file = os.path.join(log_dir, "stock_reporter.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to configure file logging: {e}")
# ==============================================================================
# endregion

# region: --- Type Definitions ---
# ==============================================================================
class NewsItem(TypedDict):
    """Represents a single news article."""
    url: str
    title: str
    summary: str
# ==============================================================================
# endregion

# region: --- Decorators and API Wrappers ---
# ==============================================================================
def retry_with_backoff(max_retries=3, initial_backoff=1, max_backoff=10):
    """
    A decorator to retry a function with exponential backoff.
    Handles common transient API errors like rate limiting (429), server errors (5xx),
    and connection issues.
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
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "api limit reached" in error_str
                    is_transient_error = any(
                        code in error_str
                        for code in ["502", "504", "timeout", "connection"]
                    )

                    if not (is_rate_limit or is_transient_error):
                        raise  # Not a retryable error

                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
                        raise

                    if is_rate_limit:
                        sleep_time = min(backoff * 3, 30) + random.uniform(0, 1)
                        log_msg = f"API rate limit hit. Waiting {sleep_time:.2f}s before retry {retry_count}/{max_retries} for {func.__name__}"
                    else:
                        sleep_time = backoff + random.uniform(0, 1)
                        log_msg = f"Retrying {func.__name__} in {sleep_time:.2f}s after error: {e}"

                    logger.warning(log_msg)
                    time.sleep(sleep_time)
                    backoff = min(backoff * 2, max_backoff)
            return None # Should not be reached
        return wrapper
    return decorator

@retry_with_backoff()
def _api_call(api_func: Callable, *args, **kwargs) -> Any:
    """Generic wrapper for all Finnhub API calls to apply retry logic."""
    return api_func(*args, **kwargs)
# ==============================================================================
# endregion

# region: --- Caching and Time Utilities ---
# ==============================================================================
EST = pytz.timezone("US/Eastern")

def get_last_trading_day(current_time: datetime) -> datetime:
    """Get the most recent trading day, excluding weekends and current day before market opens."""
    if current_time.tzinfo is None:
        current_time = EST.localize(current_time)

    last_day = current_time
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    if current_time < market_open:
        last_day -= timedelta(days=1)

    while last_day.weekday() >= 5:  # Saturday or Sunday
        last_day -= timedelta(days=1)
    return last_day

def is_cache_stale(cache_timestamp: str) -> bool:
    """Check if cached data is stale based on last valid trading day and market hours."""
    try:
        cache_time = datetime.fromisoformat(cache_timestamp)
        if cache_time.tzinfo is None:
            cache_time = EST.localize(cache_time)
    except (ValueError, TypeError):
        return True # Invalid timestamp format, treat as stale

    current_time = datetime.now(EST)
    last_trading_day = get_last_trading_day(current_time)

    if cache_time.date() < last_trading_day.date():
        return True

    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_open = market_open <= current_time <= market_close and current_time.weekday() < 5

    if is_market_open and (current_time - cache_time).total_seconds() > 3600: # 60 minutes
        return True

    if cache_time.date() == last_trading_day.date():
        last_market_close = cache_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return cache_time < last_market_close

    return False

def _is_cache_valid(cached_data: Dict[str, Any], data_type: str) -> bool:
    """Check if a specific data type in the cache is from today."""
    if not cached_data or data_type not in cached_data:
        return False
    try:
        timestamp = cached_data[data_type].get("timestamp")
        if not timestamp:
            return False
        cache_time = datetime.fromisoformat(timestamp)
        if cache_time.tzinfo is None:
            cache_time = EST.localize(cache_time)
        return cache_time.date() == datetime.now(EST).date()
    except Exception as e:
        logger.error(f"Error validating cache for {data_type}: {e}")
        return False

def _update_cache(cache: shelve.Shelf, ticker: str, key: str, data: Any):
    """Updates the cache for a given ticker and key with a timestamp."""
    if ticker not in cache:
        cache[ticker] = {}
    cache[ticker][key] = {
        "data": data,
        "timestamp": datetime.now(EST).isoformat(),
    }
# ==============================================================================
# endregion

# region: --- Data Fetching ---
# ==============================================================================
def _get_basic_info(client: finnhub.Client, ticker: str) -> Dict[str, Any]:
    """Fetch company profile, financials, and peers, handling failures gracefully."""
    result = {
        "profile": {"ticker": ticker, "name": ticker},
        "basic_financials": {"metric": {}},
        "peers": [],
    }

    # Fetch Profile
    try:
        profile = _api_call(client.company_profile2, symbol=ticker)
        if isinstance(profile, dict) and profile:
            result["profile"] = {"ticker": ticker, "name": ticker, **profile}
        else:
            logger.warning(f"Invalid profile response for {ticker}: {profile}")
    except Exception as e:
        logger.warning(f"Could not fetch profile for {ticker}: {e}")

    # Fetch Financials
    try:
        financials = _api_call(client.company_basic_financials, ticker, "all")
        if isinstance(financials, dict) and "metric" in financials:
            result["basic_financials"] = financials
    except Exception as e:
        logger.warning(f"Could not fetch financials for {ticker}: {e}")

    # Fetch Peers
    try:
        peers = _api_call(client.company_peers, ticker)
        if isinstance(peers, list):
            result["peers"] = peers
    except Exception as e:
        logger.warning(f"Could not fetch peers for {ticker}: {e}")

    return result

def _get_current_price(client: finnhub.Client, ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch current stock price and check for suspicious data."""
    try:
        quote = _api_call(client.quote, ticker)
        if not isinstance(quote, dict) or 'c' not in quote:
             logger.warning(f"Invalid quote response for {ticker}")
             return None

        current_price = quote.get("c", 0.0)
        prev_close = quote.get("pc", 0.0)
        daily_percent = quote.get("dp", 0.0)

        # Check for potentially stale or delisted stock data
        if abs(daily_percent) > 1000 or prev_close < 0.10:
            logger.warning(f"Suspicious price data for {ticker}: Price=${current_price}, PrevClose=${prev_close}, Change={daily_percent}%")
            quote['dp'] = 0.0
            quote['data_warning'] = "Potentially stale/delisted"

        return {
            "current_price": current_price,
            "change": quote.get("dp", 0.0),
            "change_amount": quote.get("d", 0.0),
            "high": quote.get("h", 0.0),
            "low": quote.get("l", 0.0),
            "open": quote.get("o", 0.0),
            "previous_close": prev_close,
            "data_warning": quote.get("data_warning")
        }
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {e}")
        return None

def _get_company_news(client: finnhub.Client, ticker: str) -> List[NewsItem]:
    """Fetch recent news articles for the company."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        news = _api_call(
            client.company_news,
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
        if not isinstance(news, list):
            return []
        return [
            {"url": item["url"], "title": item["headline"], "summary": item["summary"]}
            for item in news[:10] if item.get("headline")
        ]
    except Exception as e:
        logger.warning(f"Could not fetch news for {ticker}: {e}")
        return []

def _get_etf_data_from_yfinance(ticker: str) -> Dict[str, Any]:
    """Supplement ETF data using Yahoo Finance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info:
            logger.warning(f"yfinance returned no info for ETF {ticker}")
            return {}
            
        data = {
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "category": info.get("category"),
            "aum": info.get("totalAssets"),
            "nav": info.get("navPrice"),
            "inception_date": info.get("fundInceptionDate"),
            "yield": info.get("yield"),
            "ytd_return": info.get("ytdReturn"),
            "three_year_return": info.get("threeYearAverageReturn"),
            "five_year_return": info.get("fiveYearAverageReturn"),
        }
        # Money market fund specific fields
        if "money market" in (info.get("category", "") or "").lower():
            data["seven_day_yield"] = info.get("sevenDayYield")
            data["weighted_avg_maturity"] = info.get("weightedAverageMaturity")
            
        return data
    except Exception as e:
        logger.warning(f"Error getting supplemental yfinance data for {ticker}: {e}")
        return {}
# ==============================================================================
# endregion

# region: --- Sentiment Analysis ---
# ==============================================================================
# Global cache for the sentiment model to ensure it's loaded only once.
_sentiment_model_cache = {}

def _get_sentiment_model():
    """Load and cache the sentiment analysis model and tokenizer using a global dict."""
    # Return the cached model if it's already loaded.
    if "model" in _sentiment_model_cache:
        return _sentiment_model_cache["tokenizer"], _sentiment_model_cache["model"]

    logger.info("Loading sentiment analysis model for the first time...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for sentiment model.")
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using device_map is the modern way to handle device placement with Accelerate.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            device_map=device
        )

        # Store the loaded model in the global cache.
        _sentiment_model_cache["tokenizer"] = tokenizer
        _sentiment_model_cache["model"] = model
        logger.info("Sentiment model loaded and cached successfully.")
        return tokenizer, model
    except Exception as e:
        logger.critical(f"Fatal error: Failed to load sentiment model: {e}", exc_info=True)
        # Return None to indicate failure, preventing the tool from proceeding with a broken model.
        return None, None

def _get_sentiment_from_keywords(text: str) -> Tuple[str, float]:
    """Perform a quick sentiment analysis based on financial keywords."""
    text_lower = text.lower()
    negative_keywords = [
        "weak", "concern", "risk", "overvalued", "bearish", "sell", "short",
        "downgrade", "warning", "caution", "negative", "decline", "drop",
        "fall", "loses", "plunge", "crash", "bubble", "expensive", "correction",
        "miss", "fails", "failed", "disappointing",
    ]
    positive_keywords = [
        "strong", "buy", "bullish", "undervalued", "opportunity", "upgrade",
        "outperform", "beat", "exceeded", "growth", "positive", "potential",
        "momentum", "successful", "gain", "rally", "surge", "breakthrough",
        "innovative", "leading", "dominant",
    ]
    neg_count = sum(1 for word in negative_keywords if word in text_lower)
    pos_count = sum(1 for word in positive_keywords if word in text_lower)

    total = neg_count + pos_count
    if total == 0:
        return "neutral", 0.5
    if neg_count > pos_count:
        return "negative", min(0.9, neg_count / (total + 1))
    return "positive", min(0.9, pos_count / (total + 1))

async def _analyze_sentiment(title: str, summary: str) -> Dict[str, Any]:
    """Perform sentiment analysis combining keyword and model-based approaches."""
    try:
        tokenizer, model = _get_sentiment_model()
        # If model loading failed, we can't proceed.
        if not model or not tokenizer:
            raise RuntimeError("Sentiment model is not available.")

        device = model.device
        combined_text = f"Headline: {title}\nSummary: {summary}"
        
        # Keyword-based analysis
        kw_sentiment, kw_confidence = _get_sentiment_from_keywords(combined_text)

        # Model-based analysis
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to model's device

        with torch.no_grad():
            logits = model(**inputs).logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu() # Move to CPU
        scores = probabilities.tolist()[0]
        confidence = max(scores)
        
        id2label = model.config.id2label
        predicted_index = torch.argmax(probabilities, dim=-1).item()
        model_sentiment = id2label[predicted_index].lower()

        # Combine results
        if kw_sentiment == model_sentiment:
            final_sentiment = model_sentiment
            final_confidence = (kw_confidence + confidence) / 2
        else: # Disagreement, prefer model but with slight penalty
            final_sentiment = model_sentiment
            final_confidence = confidence * 0.8
        
        return {
            "sentiment": final_sentiment,
            "confidence": final_confidence,
        }
    except Exception as e:
        logger.warning(f"Error in sentiment analysis for '{title}': {e}")
        return {"sentiment": "neutral", "confidence": 0.0}
# ==============================================================================
# endregion

# region: --- Classification Helpers ---
# ==============================================================================
COMMON_MMFS = {"snsxx","vusxx"}
COMMON_ETFS = {"spy","qqq","voo","dia","iwm","vti","gld","xlf","xle","xlu","vnq","kweb","ihi","ewz","bnd","vwo","botz"}

def classify_security(profile: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (is_fund, type_str)"""
    if not profile:
        return False, "Stock"
    name = (profile.get("name") or "").lower()
    ticker = (profile.get("ticker") or "").lower()
    stype = (profile.get("finnhubIndustry") or "").lower() # Use finnhubIndustry as a proxy for type
    if ticker in COMMON_MMFS or any(k in name for k in ["money market","cash reserves","liquidity fund","treasury","govt"]):
        return True, "Money Market Fund"
    if ticker in COMMON_ETFS or any(k in name for k in ["etf", "exchange traded fund", "index fund", "index trust"]) or ticker.endswith("etf"):
        return True, "ETF"
    if stype and ("etf" in stype or "fund" in stype or "investment" in stype):
        return True, "ETF"
    return False, "Stock"
# ==============================================================================
# endregion

# region: --- Core Data Gathering Logic ---
# ==============================================================================
async def _async_gather_stock_data(
    client: finnhub.Client, ticker: str, cache: shelve.Shelf
) -> Dict[str, Any]:
    """
    Main orchestrator to gather all stock data for a single ticker, using cache where possible.
    """
    ticker_cache = cache.get(ticker, {})

    # 1. Get Basic Info (Profile, Financials, Peers)
    if not _is_cache_valid(ticker_cache, "basic_info"):
        logger.info(f"Fetching fresh basic info for {ticker}")
        basic_info = _get_basic_info(client, ticker)
        _update_cache(cache, ticker, "basic_info", basic_info)
    else:
        logger.info(f"Using cached basic info for {ticker}")
        basic_info = ticker_cache["basic_info"]["data"]

    # 2. Get Current Price
    use_cached_price = "current_price" in ticker_cache and not is_cache_stale(ticker_cache["current_price"]["timestamp"])
    if use_cached_price:
        logger.info(f"Using cached price for {ticker}")
        current_price = ticker_cache["current_price"]["data"]
    else:
        logger.info(f"Fetching fresh price for {ticker}")
        current_price = _get_current_price(client, ticker)
        if current_price:
             _update_cache(cache, ticker, "current_price", current_price)
        elif "current_price" in ticker_cache: # Fallback to old cache if fetch fails
             current_price = ticker_cache["current_price"]["data"]

    # 3. Get News and Sentiments
    if not _is_cache_valid(ticker_cache, "sentiments"):
        logger.info(f"Fetching fresh news and sentiments for {ticker}")
        news_items = _get_company_news(client, ticker)
        sentiment_tasks = [
            _analyze_sentiment(item["title"], item.get("summary", "")) for item in news_items
        ]
        sentiments = await asyncio.gather(*sentiment_tasks)
        sentiment_results = [
            {**news, **sentiment} for news, sentiment in zip(news_items, sentiments)
        ]
        _update_cache(cache, ticker, "sentiments", sentiment_results)
    else:
        logger.info(f"Using cached sentiments for {ticker}")
        sentiment_results = ticker_cache["sentiments"]["data"]
    
    # 4. Handle ETFs
    is_fund, fund_type = classify_security(basic_info.get("profile", {}))
    etf_data = {}
    if is_fund:
        if not _is_cache_valid(ticker_cache, "etf_data"):
             logger.info(f"Fetching fresh ETF data for {ticker} from yfinance")
             etf_data = _get_etf_data_from_yfinance(ticker)
             _update_cache(cache, ticker, "etf_data", etf_data)
        else:
             logger.info(f"Using cached ETF data for {ticker}")
             etf_data = ticker_cache["etf_data"]["data"]


    return {
        "basic_info": basic_info,
        "current_price": current_price or {},
        "sentiments": sentiment_results,
        "is_etf": is_fund,
        "fund_type": fund_type,
        "etf_data": etf_data,
    }
# ==============================================================================
# endregion

# region: --- Financial Health Assessment ---
# ==============================================================================
def _safe_float(value, default=0.0):
    """Safely convert a value to float, returning a default on failure."""
    if value is None: return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def assess_financial_health(metrics: Dict[str, Any], industry: str = "") -> str:
    """Assess the financial health of a company based on key metrics and industry benchmarks."""
    industry = industry.lower()
    score = 0
    max_score = 0
    
    roe = _safe_float(metrics.get("roeTTM"))
    current_ratio = _safe_float(metrics.get("currentRatioQuarterly"))
    net_margin = _safe_float(metrics.get("netProfitMarginTTM"))
    de_ratio = _safe_float(metrics.get("totalDebt/totalEquityQuarterly"))

    # Profitability
    max_score += 2
    if net_margin > 20: score += 2
    elif net_margin > 10: score += 1

    # Efficiency
    max_score += 2
    if roe > 20: score += 2
    elif roe > 15: score += 1

    # Liquidity
    max_score += 1
    if "tech" in industry or "retail" in industry:
        if current_ratio > 1.2: score +=1
    else:
        if current_ratio > 1.5: score += 1

    # Leverage
    max_score += 2
    if de_ratio < 0.5: score += 2
    elif de_ratio < 1.0: score += 1

    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    if percentage >= 75: return "Strong"
    if percentage >= 40: return "Moderate"
    return "Weak"
# ==============================================================================
# endregion

# region: --- Report Compilation ---
# ==============================================================================
# Helper functions for formatting
def _fmt(val, decimals=2, suffix=""):
    if val is None: return "N/A"
    try:
        return f"{_safe_float(val):,.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"
        
def _fmt_compact(val):
    if val is None: return "N/A"
    num = _safe_float(val)
    if num >= 1e12: return f"{num/1e12:.1f}T"
    if num >= 1e9: return f"{num/1e9:.1f}B"
    if num >= 1e6: return f"{num/1e6:.1f}M"
    if num >= 1e3: return f"{num/1e3:.1f}K"
    return f"{num:.2f}"

def _compile_full_report(data: Dict[str, Any]) -> str:
    """Compiles a detailed, human-readable report."""
    profile = data.get("basic_info", {}).get("profile", {})
    metrics = data.get("basic_info", {}).get("basic_financials", {}).get("metric", {})
    price = data.get("current_price", {})
    ticker, name = profile.get("ticker", "N/A"), profile.get("name", "N/A")

    if data["is_etf"]:
        etf_data = data.get("etf_data", {})
        report = [
            f"Investment Analysis: {name} ({ticker}) - {data['fund_type']}",
            "---",
            f"Fund Overview:",
            f"  - Type: {data['fund_type']} | Category: {profile.get('finnhubIndustry', 'N/A')}",
            f"  - Current Price: ${_fmt(price.get('current_price'))} ({_fmt(price.get('change'))}%)",
            f"  - Daily Range: ${_fmt(price.get('low'))} - ${_fmt(price.get('high'))}",
            "---",
            "ETF Details:",
            f"  - Expense Ratio: {_fmt(etf_data.get('expense_ratio'), 2, '%')}",
            f"  - Yield: {_fmt(etf_data.get('yield'), 2, '%')}",
            f"  - AUM: ${_fmt_compact(etf_data.get('aum'))}",
            f"  - YTD Return: {_fmt(etf_data.get('ytd_return'), 2, '%')}"
        ]
    else:
        health = assess_financial_health(metrics, profile.get("finnhubIndustry", ""))
        market_cap = profile.get("marketCapitalization", 0) * 1_000_000
        report = [
            f"Stock Analysis: {name} ({ticker})",
            "---",
            f"Company Overview:",
            f"  - Industry: {profile.get('finnhubIndustry', 'N/A')} | Market Cap: ${_fmt_compact(market_cap)}",
            f"  - Price: ${_fmt(price.get('current_price'))} ({_fmt(price.get('change'))}%) | 52W Range: ${_fmt(metrics.get('52WeekLow'))} - ${_fmt(metrics.get('52WeekHigh'))}",
            "---",
            "Key Metrics:",
            f"  - P/E: {_fmt(metrics.get('peTTM'))} | P/B: {_fmt(metrics.get('pbQuarterly'))} | Beta: {_fmt(metrics.get('beta'))}",
            f"  - ROE: {_fmt(metrics.get('roeTTM'), 2, '%')} | Net Margin: {_fmt(metrics.get('netProfitMarginTTM'), 2, '%')}",
            f"  - D/E: {_fmt(metrics.get('totalDebt/totalEquityQuarterly'))} | Health: {health}",
        ]
    
    sentiments = data.get("sentiments", [])
    if sentiments:
        report.append("\nRecent News:")
        
        # Calculate overall sentiment
        try:
            positive_confidence = sum(
                item.get("confidence", 0.0) for item in sentiments if item.get("sentiment") == "positive"
            )
            total_confidence = sum(item.get("confidence", 0.0) for item in sentiments)
            sentiment_ratio = positive_confidence / total_confidence if total_confidence > 0 else 0.5
            
            if sentiment_ratio > 0.6:
                overall_sentiment = "Bullish"
            elif sentiment_ratio < 0.4:
                overall_sentiment = "Bearish"
            else:
                overall_sentiment = "Neutral"
            
            positive_count = sum(1 for item in sentiments if item.get("sentiment") == "positive")
            report.append(f"  - Overall Sentiment: {overall_sentiment} ({positive_count}/{len(sentiments)} positive articles)")
        except Exception as e:
            logger.warning(f"Could not calculate overall sentiment for {ticker}: {e}")


        for item in sentiments[:3]: # Show top 3
            report.append(f"  - [{item.get('sentiment', 'N/A').capitalize()}] {item.get('title', 'No Title')}")

    return "\n".join(report)

def _compile_compact_report(data: Dict[str, Any]) -> str:
    """Compiles a compact, token-optimized report suitable for LLMs."""
    profile = data.get("basic_info", {}).get("profile", {})
    metrics = data.get("basic_info", {}).get("basic_financials", {}).get("metric", {})
    price = data.get("current_price", {})
    sentiments = data.get("sentiments", [])
    ticker = profile.get("ticker", "UNK")

    def _get_news_score(sentiments: List[Dict[str, Any]]) -> str:
        """Calculates a news sentiment score from 0.0 to 1.0."""
        if not sentiments:
            return "0.5"  # Neutral default
        try:
            positive_count = sum(1 for item in sentiments if item.get("sentiment") == "positive")
            score = positive_count / len(sentiments)
            return f"{score:.2f}"
        except (ZeroDivisionError, TypeError):
            return "0.5"

    def _get_health_score_numeric(health_str: str) -> str:
        """Converts health string to a numeric score."""
        if "Strong" in health_str: return "0.8"
        if "Weak" in health_str: return "0.2"
        return "0.5"

    news_score = _get_news_score(sentiments)

    if data["is_etf"]:
        etf_data = data.get("etf_data", {})
        health_score = "0.5" # Default for funds
        fields = [
            "F", ticker,
            _fmt(price.get('current_price'), 2), _fmt(price.get('change'), 1),
            _fmt(etf_data.get('expense_ratio'), 2), _fmt(etf_data.get('yield'), 2),
            _fmt(etf_data.get('ytd_return'), 2), profile.get('finnhubIndustry', '')[:10],
            health_score,
            news_score
        ]
    else:
        health_str = assess_financial_health(metrics, profile.get("finnhubIndustry", ""))
        health_score = _get_health_score_numeric(health_str)
        fields = [
            "S", ticker,
            _fmt(price.get('current_price'), 2), _fmt(price.get('change'), 1),
            _fmt(metrics.get('peTTM'), 1), _fmt(metrics.get('roeTTM'), 1),
            _fmt(metrics.get('netProfitMarginTTM'), 1), _fmt(metrics.get('totalDebt/totalEquityQuarterly'), 2),
            health_score,
            news_score
        ]
    return "|".join(str(f) for f in fields)
# ==============================================================================
# endregion

# region: --- Main Tool Class ---
# ==============================================================================
class Tools:
    class Valves(BaseModel):
        FINNHUB_API_KEY: str = Field(
            default="", description="Required API key to access Finnhub services."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client = None
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_analysis",
                    "description": "BATCH-ONLY. Get a comprehensive analysis for multiple stock or ETF tickers. Provide all tickers in the `tickers` array.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "An array of ticker symbols, e.g., ['AAPL', 'GOOGL', 'SPY']",
                            },
                             "report_format": {
                                 "type": "string",
                                 "enum": ["full", "compact"],
                                 "description": "'full' for a detailed human-readable report, 'compact' for a token-optimized format.",
                                 "default": "full",
                            }
                        },
                        "required": ["tickers"],
                    },
                },
            }
        ]

    def _initialize_client(self):
        """Initializes the Finnhub client if not already done."""
        if self.client is None:
            if not self.valves.FINNHUB_API_KEY:
                raise ValueError("FINNHUB_API_KEY is not set in valves.")
            self.client = finnhub.Client(api_key=self.valves.FINNHUB_API_KEY)

    async def get_stock_analysis(
        self,
        tickers: List[str],
        report_format: str = "full",
        __user__={},
        __event_emitter__=None,
    ) -> str:
        """The main tool function exposed to the model."""
        self._initialize_client()
        
        if not tickers:
            return "Error: No tickers provided. Please provide a list of stock symbols."

        unique_tickers = sorted(list(set([t.strip().upper() for t in tickers if t.strip()])))
        logger.info(f"Starting analysis for: {unique_tickers} (Format: {report_format})")
        
        cache_path = os.path.join(os.path.dirname(__file__), "stock_cache.db")
        
        all_reports = []
        try:
            with shelve.open(cache_path, writeback=True) as cache:
                total_tickers = len(unique_tickers)
                for i, ticker in enumerate(unique_tickers):
                    if __event_emitter__ and (i % 4 == 0 or i == total_tickers - 1):
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Processing {ticker} ({i+1}/{total_tickers})...", "done": False}
                        })
                    
                    try:
                        data = await _async_gather_stock_data(self.client, ticker, cache)
                        if report_format == "compact":
                            report = _compile_compact_report(data)
                        else:
                            report = _compile_full_report(data)
                        all_reports.append(report)
                    except Exception as e:
                        logger.error(f"Failed to process ticker {ticker}: {e}\n{traceback.format_exc()}")
                        all_reports.append(f"Error processing {ticker}: {e}")

        except Exception as e:
            logger.critical(f"A critical error occurred during analysis: {e}\n{traceback.format_exc()}")
            return f"An unexpected error occurred: {e}"
            
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Analysis complete.", "done": True}})

        separator = "\n\n" + "=" * 20 + "\n\n" if report_format == "full" else "\n"
        
        final_output = separator.join(all_reports)

        if report_format == "compact":
            schema = (
                "SCHEMA:\n"
                "S|ticker|price|day_chg%|pe|roe%|net_margin%|d/e|health_score|news_score\n"
                "F|ticker|price|day_chg%|expense%|yield%|ytd%|category|health_score|news_score\n---\n"
            )
            return schema + final_output

        return final_output

# Example of how to run this tool if executed directly (for testing)
async def main():
    tools = Tools()
    # IMPORTANT: Replace with your actual API key for testing
    tools.valves.FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_API_KEY_HERE") 
    
    test_tickers = ["AAPL", "GOOGL", "SPY", "NONEXISTENTTICKER", "VTSAX"]
    
    print("--- FULL REPORT ---")
    full_report = await tools.get_stock_analysis(tickers=test_tickers, report_format="full")
    print(full_report)
    
    print("\n\n--- COMPACT REPORT ---")
    compact_report = await tools.get_stock_analysis(tickers=test_tickers, report_format="compact")
    print(compact_report)

if __name__ == "__main__":
    # To run this for testing, ensure you have an environment variable
    # FINNHUB_API_KEY set, or replace the placeholder in main().
    asyncio.run(main())
# ==============================================================================
# endregion

