import os
import sys
from typing import Annotated
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI

from tools.python_repl_tool import python_repl_tool
from tools.stock_performance_tool import stock_data_tool
from tools.wikipedia_tool import wikipedia_tool


#Setup OPEN_API_KEY
from utils.load_env_file import load_env_file
load_env_file()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#Build the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
tools = [python_repl_tool, stock_data_tool, wikipedia_tool]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_API_KEY)

