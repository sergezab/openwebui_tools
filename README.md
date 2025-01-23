# Open WebUI Tools: Stock Reporter

This repository contains the **Stock Reporter** tool, an enhanced version of the original concept by [christ-offer](https://github.com/christ-offer/open-webui-tools). The Stock Reporter is a comprehensive stock analysis tool that gathers data from the Finnhub API and compiles detailed reports.

## Enhancements

I have significantly improved the original tool with the following features:

- **Caching**: Implemented stock price caching to:
  - Store current price data when outside trading hours (9 AM–4 PM EST) or on weekends.
  - Cache current price data when the request is within 2 hours of the last cache update.
  - Added timezone handling using `pytz` to accurately determine market hours in EST.
  - Detailed logging to track when cached prices are used and why.

  These improvements enhance efficiency by reducing unnecessary API calls during non-trading hours and reusing recent price data within a 2-hour window.

- **Logging**: Added detailed logging to monitor:
  - Cache usage and reasons for using cached data.
  - Progress tracking for each component.
  - Clear distinction between warnings and errors.

- **Error Handling**: Enhanced error handling to ensure the tool works reliably with all types of tickers:
  - Returns neutral sentiment instead of failing during sentiment analysis.
  - Logs warnings instead of errors and continues processing remaining news items.
  - Starts with a valid default data structure and handles each component independently.
  - Provides default values for all data types, ensuring graceful degradation of functionality.

- **Multiple Ticker Support**: Added support for multiple tickers in a single query, allowing users to analyze several stocks simultaneously.

- **Improved Security Handling**: The tool now better handles different types of securities:
  - For traditional stocks, it provides detailed financial analysis with metrics like ROE, P/E ratio, and leverage.
  - For ETFs, money market funds, and other non-traditional securities, it displays an appropriate message indicating that traditional financial metrics are not applicable.

- **Bug Fixes**: Resolved issues in the stock reporter script:
  - Fixed the float division by zero error in the Current Price Position calculation.
  - Properly handled `None` values in metrics to prevent `NoneType` comparison errors.

## Installation

The Stock Reporter is written in Python and can be easily integrated into your own WebUI.

1. **Install the `finnhub-python` package**:

   ```bash
   pip install finnhub-python
   ```

   If you are running OpenWebUI through Docker, install it using the following command:

   ```bash
   docker exec -it owui bash -c "pip install finnhub-python"
   ```

2. **Set up the Finnhub API key**:

   - Provide your Finnhub API key in the Valves section of your WebUI settings.
   - You can also set up the Finnhub API key through the OpenWebUI interface.

## Usage

To use the Stock Reporter tool:

1. **Ensure the Finnhub API key is configured** as described in the installation steps.

2. **Run the tool** through your WebUI, specifying the ticker(s) you wish to analyze. The tool supports multiple tickers in a single query, separated by commas (e.g., `AAPL, GOOGL, MSFT`).

3. **View the generated report**, which will include comprehensive stock analysis, financial metrics, and recent news sentiment analysis.

## Cache Warming System

I have implemented a complete cache warming system that pre-caches your stock data daily:

- **Cache Warmer Script (`stock_cache_warmer.py`)**:
  - Pre-caches data for multiple tickers in efficient chunks (5 tickers at a time).
  - Includes detailed logging in `stock_cache_warmer.log`.
  - Supports multiple ways to provide tickers: command line arguments, text file (one ticker per line), or saved default tickers.
  - Handles API rate limiting with delays between chunks and saves progress in case of errors.

- **Shell Script (`warm_cache.sh`)**:
  - Makes the cache warmer executable.
  - Provides examples of different ways to run the warmer: direct ticker list, save and use default tickers, or read tickers from a file.
  - Uses an environment variable for the API key.

- **Crontab Setup (`crontab_example.txt`)**:
  - Runs the cache warmer at 7:00 AM before the market opens.
  - Only runs on market days (Monday–Friday).
  - Logs output and errors to `cron.log`.

**To set up daily cache warming**:

1. **First-time setup**:

   ```bash
   chmod +x openwebui_tools/warm_cache.sh
   # Save your default tickers
   ./openwebui_tools/warm_cache.sh
   ```

2. **Set up the cron job**:

   ```bash
   crontab -e
   # Add the line from crontab_example.txt (modified for your paths)
   ```

This system ensures your stock data is pre-cached each morning before the market opens, making the model's responses much faster since it will use the cached data from that day.

## License

This project is licensed under the MIT License.

---

*Original idea and all credits belong to [christ-offer](https://github.com/christ-offer/open-webui-tools).*

