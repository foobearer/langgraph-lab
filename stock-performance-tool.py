import os
from typing import Annotated
import pandas as pd
from langchain_core.tools import tool

@tool 
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."],
    num_days: Annotated[int, "The number of days of stock data required to respond to the user query."]
) -> str:
    
    """
        Use this to lookup stock performance data for the companies to retrieve a table from a CSV. You may need to convert
        company names into ticker symbols to call this function, e.g Apple Inc. --> AAPL, and you may need to convert weeks,
        months, and years into days. 
    """

    file_path = f"data/{company_ticker}.cvg"

    if not os.path.exists(file_path):
        return f"Sorry, but the data for the company {company_ticker} is not available. Please try Apple, Amazon, Meta"
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    stock_df.index = stock_df.index.date

    max_num_days = (stock_df.index.max() - stock_df.index.min()).days

    if num_days > max_num_days:
        return "Sorry, this period exceeds the data available. Reduce it to continue."
    
    final_date = stock_df.index.max()
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]


    return f"We've successfully retrieved the last '{num_days}' of data for {company_ticker}: \n\n {filtered_df.to_markdown()}"


#Invoking the tool and printing the result.

retrieved_data = stock_data_tool.invoke({"company_ticker": "AMZN", "num_days": 5})
print(retrieved_data)


