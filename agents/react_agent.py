#Setup OPEN_API_KEY
import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

from tools.python_repl_tool import python_repl_tool
from tools.wikipedia_tool import wikipedia_tool
from tools.stock_performance_tool import stock_data_tool

from utils.pretty_print import pretty_print_messages
from utils.load_env_file import load_env_file
load_env_file()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#Build the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
tools = [python_repl_tool, stock_data_tool, wikipedia_tool]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_API_KEY)


prompt = """
You are an assistant for research and analysis of Fortune 500 companies. You have access to three tools:
- A Wikipedia tool for retrieving factual summary information about companies
- A stock performance data tool for retrieving stock price information from local CSV files
- A Python tool for executing Python code, which is to be used for creating stock performance visualizations
"""

# Create an agent using the create_react_agent function
agent = create_react_agent(
    llm,
    tools=tools,
    name="finance_assistant",
    prompt=prompt
)

print(agent.get_graph().draw_mermaid())

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}
):
    pretty_print_messages(chunk)