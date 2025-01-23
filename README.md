Open WebUI Tools

This repository contains an enhanced version of the Stock Reporter tool for Open WebUI. The Stock Reporter provides comprehensive stock analysis by gathering data from the Finnhub API and compiling detailed reports.

Original Idea and Credits: The original concept and implementation were developed by christ-offer.

Enhancements

I have made significant modifications to the original tool, including:
	•	Stock Price Caching:
	•	Implemented caching of current price data when outside trading hours (9 AM–4 PM EST) or on weekends.
	•	Cached current price data is used if the request is within 2 hours of the last cache update.
	•	Added timezone handling using pytz to accurately determine market hours in EST.
	•	Introduced detailed logging to track when cached prices are being used and the reasons for it.
	•	Improved Efficiency:
	•	Reduced unnecessary API calls during non-trading hours.
	•	Reused recent price data within a 2-hour window.
	•	Provided clear logging about cache usage.
	•	Enhanced Security Handling:
	•	For traditional stocks, the tool continues to display detailed financial analysis with metrics like ROE, P/E ratio, and leverage.
	•	For ETFs, money market funds, and other non-traditional securities (e.g., GLD and VUSXX), it now displays an appropriate message:
“Note: Traditional financial metrics are not applicable for [name] as it appears to be an ETF, money market fund, or other non-traditional security. Please refer to the fund’s prospectus and other fund-specific metrics for a more appropriate analysis.”
	•	Error Handling Improvements:
	•	Fixed the float division by zero error in the Current Price Position calculation by ensuring a non-zero denominator.
	•	Handled NoneType comparison errors by properly managing None values in metrics, ensuring all comparisons are done with numeric values.
	•	Implemented comprehensive error handling to ensure the tool works reliably with all types of tickers, including:
	•	Returning neutral sentiment instead of failing during sentiment analysis.
	•	Logging warnings instead of errors and continuing processing remaining news items.
	•	Starting with a valid default data structure and handling each component independently.
	•	Continuing if some parts fail and returning valid data even on critical errors.
	•	Providing default values for all data types and graceful degradation of functionality.
	•	Logging Improvements:
	•	Added detailed logs for debugging.
	•	Provided clear distinctions between warnings and errors.
	•	Included progress tracking for each component.
	•	Logged cache hit/miss events.
	•	Cache Warming System:
	•	Created a complete cache warming system that pre-caches stock data daily.
	•	Introduced a Cache Warmer Script (stock_cache_warmer.py) that pre-caches data for multiple tickers in efficient chunks (5 tickers at a time), includes detailed logging in stock_cache_warmer.log, and supports multiple ways to provide tickers:
	•	Command line arguments.
	•	Text file (one ticker per line).
	•	Saved default tickers.
	•	Handled API rate limiting with delays between chunks and saved progress in case of errors.
	•	Provided a Shell Script (warm_cache.sh) to make the cache warmer executable and offered examples of different ways to run the warmer:
	•	Direct ticker list.
	•	Save and use default tickers.
	•	Read tickers from a file.
	•	Used environment variables for the API key.
	•	Included a Crontab Setup (crontab_example.txt) to run the cache warmer at 7:00 AM before market opens, only on market days (Monday–Friday), and logged output and errors to cron.log.

Usage

To use this tool, follow these steps:
	1.	Install the finnhub-python Package:
Ensure that the finnhub-python package is installed. You can install it using pip:

pip install finnhub-python

If you are running Open WebUI through Docker, you can install it using the following command:

docker exec -it open-webui bash -c "pip install finnhub-python"


	2.	Provide the Finnhub API Key:
Set up your Finnhub API key through the Open WebUI interface:
	•	Navigate to the “Workspace” tab in your Open WebUI instance.
	•	Click on the “Tools” section.
	•	Select the Stock Reporter tool.
	•	In the “Valves” section, enter your Finnhub API key.
	3.	Integrate the Tool into Open WebUI:
Follow the manual installation process as outlined in the Open WebUI documentation:
	•	Navigate to the “Workspace” tab in your Open WebUI instance.
	•	Click on the “Tools” section.
	•	Click the “+” button to add a new tool.
