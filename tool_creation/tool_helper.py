import pandas as pd
import numpy as np

from langchain.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

from api_helper import get_price_change

class PriceChangeInput(BaseModel):
    days:   int = Field(description="How many days ago to look")
    ticker: str=Field(description="Which symbol to look up",)
    
get_price_change_tool = StructuredTool.from_function(
    func=get_price_change,
    name="get price change",
    description="mandatory if you need to answer questions about cryptocurrency price change",
    args_schema=PriceChangeInput
)

if __name__ == "__main__":
    print(get_price_change_tool.run({"days" : 1, "ticker": "BTC"}))