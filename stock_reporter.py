"""
title: Stock Market Helper
description: A comprehensive stock analysis tool that gathers data from Finnhub API and compiles a detailed report.
author: Sergii Zabigailo
author_url: https://github.com/christ-offer/
github: https://github.com/christ-offer/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.9
license: MIT
requirements: finnhub-python,pytz
"""

import finnhub
import requests
import aiohttp
import asyncio
import os
import json
import logging
import pytz
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "stock_reporter.log")
        ),
    ],
)
logger = logging.getLogger(__name__)


def _format_date(date: datetime) -> str:
    """Helper function to format date for Finnhub API"""
    return date.strftime("%Y-%m-%d")


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
        "peers": []  # Empty peers list
    }
    
    try:
        # Try to get profile
        try:
            profile = client.company_profile2(symbol=ticker)
            if profile:
                result["profile"] = profile
        except Exception as e:
            logger.warning(f"Could not fetch profile for {ticker}, using minimal profile: {str(e)}")
    
        # Try to get financials
        try:
            basic_financials = client.company_basic_financials(ticker, "all")
            if basic_financials and "metric" in basic_financials:
                result["basic_financials"] = basic_financials
        except Exception as e:
            logger.warning(f"Could not fetch financials for {ticker}, using empty financials: {str(e)}")
    
        # Try to get peers
        try:
            peers = client.company_peers(ticker)
            if peers:
                result["peers"] = peers
        except Exception as e:
            logger.warning(f"Could not fetch peers for {ticker}, using empty peers list: {str(e)}")
    
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
        # Return default values that won't break the report
        return {
            "current_price": 0.0,
            "change": 0.0,
            "change_amount": 0.0,
            "high": 0.0,
            "low": 0.0,
            "open": 0.0,
            "previous_close": 0.0,
        }


def _get_company_news(client: finnhub.Client, ticker: str) -> List[Dict[str, str]]:
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
        return [{"url": item["url"], "title": item["headline"]} for item in news_items]
    except Exception as e:
        logger.warning(f"Could not fetch news for {ticker}, using empty news list: {str(e)}")
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
async def _async_sentiment_analysis(content: str) -> Dict[str, Union[str, float]]:
    """
    Perform sentiment analysis on text content.
    Returns neutral sentiment with 0 confidence if analysis fails.
    """
    try:
        tokenizer, model = _get_sentiment_model()
        inputs = tokenizer(
            content, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_scores = probabilities.tolist()[0]

        # Update sentiment labels to match the new model's output
        sentiments = ["Neutral", "Positive", "Negative"]
        sentiment = sentiments[sentiment_scores.index(max(sentiment_scores))]
        confidence = max(sentiment_scores)

        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        logger.warning(f"Error in sentiment analysis, using neutral sentiment: {str(e)}")
        return {"sentiment": "Neutral", "confidence": 0.0}


# Asynchronous data gathering
def _load_cache(cache_file: str) -> Dict[str, Any]:
    """Load cached stock data from file"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading cache from {cache_file}: {str(e)}")
            return {}
    return {}


def _save_cache(cache_file: str, cache_data: Dict[str, Any]) -> None:
    """Save cached stock data to file"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving cache to {cache_file}: {str(e)}")
        raise


def _is_cache_valid(cached_data: Dict[str, Any], data_type: str) -> bool:
    """Check if cached data is from today"""
    if not cached_data or data_type not in cached_data:
        return False
    try:
        cache_time = datetime.fromisoformat(cached_data[data_type]["timestamp"])
        return cache_time.date() == datetime.now().date()
    except Exception as e:
        logger.error(f"Error validating cache for {data_type}: {str(e)}")
        return False


def _is_market_hours() -> bool:
    """Check if current time is within market hours (9:00 AM - 4:00 PM EST)"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
        
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

async def _async_gather_stock_data(
    client: finnhub.Client, ticker: str
) -> Dict[str, Any]:
    """
    Gather all stock data with graceful error handling.
    Returns minimal valid data structure even if some parts fail.
    Caches current price if within 2 hours of request or outside trading hours.
    """
    try:
        # Initialize cache
        cache_file = os.path.join(os.path.dirname(__file__), "stock_cache.json")
        cache = _load_cache(cache_file)
        ticker_cache = cache.get(ticker, {})

        # Initialize result with default values
        result = {
            "basic_info": {"profile": {"ticker": ticker}, "basic_financials": {"metric": {}}, "peers": []},
            "current_price": {
                "current_price": 0.0,
                "change": 0.0,
                "change_amount": 0.0,
                "high": 0.0,
                "low": 0.0,
                "open": 0.0,
                "previous_close": 0.0,
            },
            "sentiments": []
        }

        try:
            # Get basic info (use cache if valid)
            if not _is_cache_valid(ticker_cache, "basic_info"):
                logger.info(f"Fetching fresh basic info for {ticker}")
                basic_info = _get_basic_info(client, ticker)
                ticker_cache["basic_info"] = {
                    "data": basic_info,
                    "timestamp": datetime.now().isoformat(),
                }
                _save_cache(cache_file, cache)
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
                cache_time = datetime.fromisoformat(ticker_cache["current_price"]["timestamp"])
                time_diff = datetime.now() - cache_time
                
                # Use cache if:
                # 1. Outside trading hours
                # 2. Within 2 hours of last cache
                if not _is_market_hours() or time_diff <= timedelta(hours=2):
                    use_cached_price = True
                    logger.info(f"Using cached price for {ticker} ({'outside trading hours' if not _is_market_hours() else 'within 2 hours'})")
            
            if use_cached_price:
                current_price = ticker_cache["current_price"]["data"]
            else:
                # Get fresh price data
                logger.info(f"Fetching fresh price for {ticker}")
                current_price = _get_current_price(client, ticker)
                # Cache the new price data
                ticker_cache["current_price"] = {
                    "data": current_price,
                    "timestamp": datetime.now().isoformat()
                }
                cache[ticker] = ticker_cache
                _save_cache(cache_file, cache)
            
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
                        async with aiohttp.ClientSession() as session:
                            scrape_tasks = [
                                _async_web_scrape(session, item["url"]) for item in news_items
                            ]
                            contents = await asyncio.gather(*scrape_tasks)

                        sentiment_tasks = [
                            _async_sentiment_analysis(content) for content in contents if content
                        ]
                        sentiments = await asyncio.gather(*sentiment_tasks)

                        sentiment_results = [
                            {
                                "url": news_items[i]["url"],
                                "title": news_items[i]["title"],
                                "sentiment": sentiment["sentiment"],
                                "confidence": sentiment["confidence"],
                            }
                            for i, sentiment in enumerate(sentiments)
                            if contents[i]
                        ]

                        # Cache sentiment results
                        ticker_cache["sentiments"] = {
                            "data": sentiment_results,
                            "timestamp": datetime.now().isoformat(),
                        }
                        cache[ticker] = ticker_cache
                        _save_cache(cache_file, cache)
                        result["sentiments"] = sentiment_results
                except Exception as e:
                    logger.error(f"Error processing news/sentiments for {ticker}: {str(e)}")
            else:
                logger.info(f"Using cached sentiments for {ticker}")
                result["sentiments"] = ticker_cache["sentiments"]["data"]
        except Exception as e:
            logger.error(f"Error in sentiment processing for {ticker}: {str(e)}")

        return result

    except Exception as e:
        logger.error(f"Critical error gathering stock data for {ticker}: {str(e)}")
        return result  # Return default structure even on critical error


def _compile_report(data: Dict[str, Any]) -> str:
    """
    Compile gathered data into a comprehensive structured report.
    """
    try:
        profile = data["basic_info"]["profile"]
        financials = data["basic_info"]["basic_financials"]
        metrics = financials.get("metric", {})
        peers = data["basic_info"]["peers"]
        price_data = data["current_price"]

        # Handle cases where profile might not have all fields (e.g., for ETFs)
        ticker = profile.get('ticker', 'Unknown')
        name = profile.get('name', ticker)  # Use ticker as fallback if name is not available

        report = f"""
        Comprehensive Stock Analysis Report for {name} ({ticker})

        Basic Information:
        Industry: {profile.get('finnhubIndustry', 'N/A')}
        Market Cap: {profile.get('marketCapitalization', 0):,.0f} M USD
        Share Outstanding: {profile.get('shareOutstanding', 0):,.0f} M
        Country: {profile.get('country', 'N/A')}
        Exchange: {profile.get('exchange', 'N/A')}
        IPO Date: {profile.get('ipo', 'N/A')}

        Current Trading Information:
        Current Price: ${price_data['current_price']:.2f}
        Daily Change: {price_data['change']:.2f}% (${price_data['change_amount']:.2f})
        Day's Range: ${price_data['low']:.2f} - ${price_data['high']:.2f}
        Open: ${price_data['open']:.2f}
        Previous Close: ${price_data['previous_close']:.2f}

        Key Financial Metrics:
        52 Week High: ${financials['metric'].get('52WeekHigh', 'N/A')}
        52 Week Low: ${financials['metric'].get('52WeekLow', 'N/A')}
        P/E Ratio: {financials['metric'].get('peBasicExclExtraTTM', 'N/A')}
        EPS (TTM): ${financials['metric'].get('epsBasicExclExtraItemsTTM', 'N/A')}
        Return on Equity: {financials['metric'].get('roeRfy', 'N/A')}%
        Debt to Equity: {financials['metric'].get('totalDebtToEquityQuarterly', 'N/A')}
        Current Ratio: {financials['metric'].get('currentRatioQuarterly', 'N/A')}
        Dividend Yield: {financials['metric'].get('dividendYieldIndicatedAnnual', 'N/A')}%

        Peer Companies: {', '.join(peers[:5])}

        Detailed Financial Analysis:

        1. Valuation Metrics:
        P/E Ratio: {metrics.get('peBasicExclExtraTTM', 'N/A')}
        - Interpretation: {'High (may be overvalued)' if float(metrics.get('peBasicExclExtraTTM', 0) or 0) > 25 else 'Moderate' if 15 <= float(metrics.get('peBasicExclExtraTTM', 0) or 0) <= 25 else 'Low (may be undervalued)'}

        P/B Ratio: {metrics.get('pbQuarterly', 'N/A')}
        - Interpretation: {'High' if float(metrics.get('pbQuarterly', 0) or 0) > 3 else 'Moderate' if 1 <= float(metrics.get('pbQuarterly', 0) or 0) <= 3 else 'Low'}

        2. Profitability Metrics:
        Return on Equity: {metrics.get('roeRfy', 'N/A')}%
        - Interpretation: {'Excellent' if float(metrics.get('roeRfy', 0) or 0) > 20 else 'Good' if 15 <= float(metrics.get('roeRfy', 0) or 0) <= 20 else 'Average' if 10 <= float(metrics.get('roeRfy', 0) or 0) < 15 else 'Poor'}

        Net Profit Margin: {metrics.get('netProfitMarginTTM', 'N/A')}%
        - Interpretation: {'Excellent' if float(metrics.get('netProfitMarginTTM', 0) or 0) > 20 else 'Good' if 10 <= float(metrics.get('netProfitMarginTTM', 0) or 0) <= 20 else 'Average' if 5 <= float(metrics.get('netProfitMarginTTM', 0) or 0) < 10 else 'Poor'}

        3. Liquidity and Solvency:
        Current Ratio: {metrics.get('currentRatioQuarterly', 'N/A')}
        - Interpretation: {'Strong' if float(metrics.get('currentRatioQuarterly', 0) or 0) > 2 else 'Healthy' if 1.5 <= float(metrics.get('currentRatioQuarterly', 0) or 0) <= 2 else 'Adequate' if 1 <= float(metrics.get('currentRatioQuarterly', 0) or 0) < 1.5 else 'Poor'}

        Debt-to-Equity Ratio: {metrics.get('totalDebtToEquityQuarterly', 'N/A')}
        - Interpretation: {'Low leverage' if float(metrics.get('totalDebtToEquityQuarterly', 0) or 0) < 0.5 else 'Moderate leverage' if 0.5 <= float(metrics.get('totalDebtToEquityQuarterly', 0) or 0) <= 1 else 'High leverage'}

        4. Dividend Analysis:
        Dividend Yield: {metrics.get('dividendYieldIndicatedAnnual', 'N/A')}%
        - Interpretation: {'High yield' if float(metrics.get('dividendYieldIndicatedAnnual', 0) or 0) > 4 else 'Moderate yield' if 2 <= float(metrics.get('dividendYieldIndicatedAnnual', 0) or 0) <= 4 else 'Low yield'}

        5. Market Performance:
        52-Week Range: ${metrics.get('52WeekLow', 'N/A')} - ${metrics.get('52WeekHigh', 'N/A')}
        Current Price Position: {((price_data['current_price'] - metrics.get('52WeekLow', price_data['current_price'])) / max(0.0001, metrics.get('52WeekHigh', price_data['current_price']) - metrics.get('52WeekLow', price_data['current_price'])) * 100):.2f}% of 52-Week Range

        Beta: {metrics.get('beta', 'N/A')}
        - Interpretation: {'More volatile than market' if metrics.get('beta', 1) > 1 else 'Less volatile than market' if metrics.get('beta', 1) < 1 else 'Same volatility as market'}

    Overall Analysis:
    {
        f"{name} shows {'strong' if float(metrics.get('roeRfy', 0) or 0) > 15 and float(metrics.get('currentRatioQuarterly', 0) or 0) > 1.5 else 'moderate' if float(metrics.get('roeRfy', 0) or 0) > 10 and float(metrics.get('currentRatioQuarterly', 0) or 0) > 1 else 'weak'} financial health with {'high' if float(metrics.get('peBasicExclExtraTTM', 0) or 0) > 25 else 'moderate' if 15 <= float(metrics.get('peBasicExclExtraTTM', 0) or 0) <= 25 else 'low'} valuation metrics. The company's profitability is {'excellent' if float(metrics.get('netProfitMarginTTM', 0) or 0) > 20 else 'good' if float(metrics.get('netProfitMarginTTM', 0) or 0) > 10 else 'average' if float(metrics.get('netProfitMarginTTM', 0) or 0) > 5 else 'poor'}, and it has {'low' if float(metrics.get('totalDebtToEquityQuarterly', 0) or 0) < 0.5 else 'moderate' if float(metrics.get('totalDebtToEquityQuarterly', 0) or 0) < 1 else 'high'} financial leverage. Investors should consider these factors along with their investment goals and risk tolerance."
        if any(metrics.get(key) is not None for key in ['roeRfy', 'currentRatioQuarterly', 'peBasicExclExtraTTM', 'netProfitMarginTTM', 'totalDebtToEquityQuarterly'])
        else f"Note: Traditional financial metrics are not applicable for {name} as it appears to be an ETF, money market fund, or other non-traditional security. Please refer to the fund's prospectus and other fund-specific metrics for a more appropriate analysis."
    }


        Recent News and Sentiment Analysis:
        """

        for item in data["sentiments"]:
            report += f"""
        Title: {item['title']}
        URL: {item['url']}
        Sentiment Analysis: {item['sentiment']} (Confidence: {item['confidence']:.2f})

        """
        return report
    except Exception as e:
        logger.error(f"Error compiling report: {str(e)}")
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
                    "description": "Search the web using Perplexity AI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The ticker to perform a comprehensive stock analysis",
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

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Retrieving stock data for {single_ticker} ({idx + 1}/{len(tickers)})",
                            "done": False,
                        },
                    }
                )
                data = await _async_gather_stock_data(self.client, single_ticker)

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Compiling stock report for {single_ticker}",
                            "done": False,
                        },
                    }
                )
                report = _compile_report(data)
                last_price = data["current_price"]["current_price"]

                # Add separator between reports if this isn't the first report
                # if combined_report:
                combined_report += "\n" + "=" * 8 + "\n\n"

                combined_report += report

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
            error_msg = (
                f"Error in compile_stock_report for tickers {ticker_query} : {str(e)}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)
