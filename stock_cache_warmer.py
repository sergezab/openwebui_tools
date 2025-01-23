#!/usr/bin/env python3
"""
Stock Cache Warmer

This script pre-caches stock data for a list of tickers to speed up subsequent queries.
Run this script once per day (e.g., via cron job) before market hours to ensure fresh cache.
"""

import asyncio
import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from stock_reporter import Tools, logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'stock_cache_warmer.log'))
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS_FILE = os.path.join(os.path.dirname(__file__), 'default_tickers.json')

def save_default_tickers(tickers: List[str]) -> None:
    """Save the list of tickers to the default tickers file"""
    try:
        with open(DEFAULT_TICKERS_FILE, 'w') as f:
            json.dump(tickers, f)
        logger.info(f"Saved {len(tickers)} tickers to {DEFAULT_TICKERS_FILE}")
    except Exception as e:
        logger.error(f"Error saving default tickers: {str(e)}")
        raise

def load_default_tickers() -> List[str]:
    """Load the list of tickers from the default tickers file"""
    if os.path.exists(DEFAULT_TICKERS_FILE):
        try:
            with open(DEFAULT_TICKERS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading default tickers: {str(e)}")
            return []
    return []

async def warm_cache(api_key: str, tickers: List[str]) -> None:
    """
    Pre-cache data for all specified tickers
    
    :param api_key: Finnhub API key
    :param tickers: List of stock tickers to cache data for
    """
    tools = Tools()
    tools.valves.FINNHUB_API_KEY = api_key
    
    start_time = datetime.now()
    logger.info(f"Starting cache warming for {len(tickers)} tickers at {start_time}")
    
    # Process tickers in chunks to avoid overwhelming the API
    chunk_size = 5
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(tickers) + chunk_size - 1)//chunk_size}: {chunk}")
        
        try:
            # Join tickers with commas for the API call
            ticker_str = ','.join(chunk)
            await tools.compile_stock_report(
                ticker_str,
                __event_emitter__=lambda x: None  # Dummy event emitter
            )
            logger.info(f"Successfully cached data for: {chunk}")
        except Exception as e:
            logger.error(f"Error processing chunk {chunk}: {str(e)}")
            continue
            
        # Small delay between chunks to be nice to the API
        if i + chunk_size < len(tickers):
            await asyncio.sleep(2)
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Finished cache warming at {end_time} (Duration: {duration})")

def main():
    parser = argparse.ArgumentParser(description='Warm up stock data cache for faster queries')
    parser.add_argument('--api-key', help='Finnhub API key')
    parser.add_argument('--tickers', nargs='+', help='List of stock tickers to cache')
    parser.add_argument('--save-tickers', action='store_true', 
                      help='Save provided tickers as default for future runs')
    parser.add_argument('--file', help='File containing list of tickers (one per line)')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('FINNHUB_API_KEY')
    if not api_key:
        raise ValueError("Finnhub API key must be provided via --api-key or FINNHUB_API_KEY environment variable")
    
    # Get tickers from args, file, or default file
    tickers = []
    if args.tickers:
        tickers = args.tickers
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading tickers file {args.file}: {str(e)}")
            return
    else:
        tickers = load_default_tickers()
        
    if not tickers:
        logger.error("No tickers provided and no default tickers found")
        return
        
    # Save as default if requested
    if args.save_tickers:
        save_default_tickers(tickers)
    
    # Run the cache warmer
    asyncio.run(warm_cache(api_key, tickers))

if __name__ == '__main__':
    main()
