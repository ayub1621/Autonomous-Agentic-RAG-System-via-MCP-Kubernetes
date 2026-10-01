import yfinance as yf
from typing import Dict, Any
from fastmcp import FastMCP

# 1. Initialize the FastMCP server
mcp = FastMCP("Live Financial Data")

# 2. Expose functions as MCP Tools
@mcp.tool
def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Fetch the current market price and basic info for a given stock ticker symbol (e.g., AAPL, GOOGL)."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # yfinance returns a large dictionary; extract only what the agent needs
        return {
            "symbol": symbol,
            "name": info.get("shortName", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "N/A")
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {symbol}. {str(e)}"}

@mcp.tool
def get_market_trend(symbol: str) -> Dict[str, Any]:
    """Fetch the 5-day historical trend for a stock to determine if it is up or down."""
    try:
        stock = yf.Ticker(symbol)
        # Get the last 5 days of historical data
        hist = stock.history(period="5d")
        
        if hist.empty:
            return {"error": f"No historical data found for {symbol}"}
            
        closing_prices = hist['Close'].tolist()
        trend = "Upward" if closing_prices[-1] > closing_prices[0] else "Downward"
        
        return {
            "symbol": symbol,
            "5_day_trend": trend,
            "latest_close": closing_prices[-1]
        }
    except Exception as e:
        return {"error": f"Could not fetch trend for {symbol}. {str(e)}"}

if __name__ == "__main__":
    # 3. Run the server using STDIO transport
    print("Starting Financial MCP Server on STDIO...")
    mcp.run()