import pandas as pd
import numpy as np

from langchain.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

from api_helper import get_price_change

ds = pd.read_csv('tickers.csv', index_col=0)

my_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker_price",
            "description": "Get the price of a cryptocurrency or stock of choice",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "enum": ds["Ticker"],
                        "description": "Type of asset, e.g. BTC",
                    },
                    "timeframe_format": {
                        "type": "string",
                        "enum": {"day", "week", "month"},
                        "description": "Unit of time measure, e.g. day",
                    },
                    "timeframe_number": {
                        "type": "integer",
                        "description": "How much of the chosen timeframes has passed, e.g. 5",
                    }
                },
                "required": ["ticker", "timeframe_format", "timeframe_number"],
            },
        }
    },
]

class PriceChangeInput(BaseModel):
    days: int = Field(description="How many days ago to look"),
    ticker: str=Field(description="Which symbol to look up")
    
get_price_change_tool = StructuredTool.from_function(
    func=get_price_change,
    name="get price change",
    description="mandatory if you need to answer questions about cryptocurrency price change",
    args_schema=PriceChangeInput
)

if __name__ == "__main__":
    print(get_price_change_tool.args)