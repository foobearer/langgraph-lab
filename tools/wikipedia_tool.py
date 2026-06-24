from typing import Annotated
import wikipedia
from langchain_core.tools import tool

wikipedia.set_user_agent("langgraph-lab/1.0 (https://github.com/)")

@tool 
def wikipedia_tool(
    query: Annotated[str, "The wikipedia search to execute to find key summary information"],
):
    """Use this to search Wikipedia for actual information"""

    try: 

        results = wikipedia.search(query)

        if not results:
            return "No results found on Wikipedia."
        
        title = results[0]

        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)


    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"

# For tool testing uncomment the codes below
# company_name = "Warner bros."
# wiki_summary = wikipedia_tool.invoke(f"{company_name}")
# print(wiki_summary)
