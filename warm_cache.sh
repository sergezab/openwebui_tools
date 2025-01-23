#!/bin/bash

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make the script executable if it isn't already
chmod +x "$SCRIPT_DIR/stock_cache_warmer.py"

# Example of running with direct ticker list:
# python3 "$SCRIPT_DIR/stock_cache_warmer.py" --api-key "your-api-key" --tickers AAPL MSFT GOOGL META AMZN NVDA AMD INTC TSLA F GM NFLX DIS CSCO ORCL IBM ADBE CRM PYPL SQ COIN HOOD SHOP SNAP TWTR FB BABA JD PDD TCEHY BIDU NIO XPEV LI BILI TME DIDI GRAB SE CPNG LAZR LCID RIVN

# Example of saving default tickers and running:
python3 "$SCRIPT_DIR/stock_cache_warmer.py" \
  --api-key "${FINNHUB_API_KEY}" \
  --tickers AAPL MSFT GOOGL META AMZN NVDA AMD INTC TSLA F GM NFLX DIS CSCO ORCL IBM ADBE CRM PYPL SQ COIN HOOD SHOP SNAP TWTR FB BABA JD PDD TCEHY BIDU NIO XPEV LI BILI TME DIDI GRAB SE CPNG LAZR LCID RIVN \
  --save-tickers

# After saving tickers, you can just run:
# python3 "$SCRIPT_DIR/stock_cache_warmer.py" --api-key "${FINNHUB_API_KEY}"

# Or you can create a file with tickers (one per line) and use:
# python3 "$SCRIPT_DIR/stock_cache_warmer.py" --api-key "${FINNHUB_API_KEY}" --file "$SCRIPT_DIR/my_tickers.txt"
