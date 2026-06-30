import os
import sys
from typing import Annotated
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI

from tools.python_repl_tool import python_repl_tool
from tools.stock_performance_tool import stock_data_tool
from tools.wikipedia_tool import wikipedia_tool

from utils.pretty_print import pretty_print_messages


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

llm_with_tools = llm.bind_tools(tools)

def llm_nodes(state:State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#Define the agent workflow

# Create the llm and tools nodes
graph_builder.add_node("llm", llm_nodes)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add the edges
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition, ["tools", END])
graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()


print(graph.get_graph().draw_mermaid())

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}
):
    pretty_print_messages(chunk)