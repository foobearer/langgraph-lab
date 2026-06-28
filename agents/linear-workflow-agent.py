import sys
import os
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import ToolNode

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.python_repl_tool import python_repl_tool
from tools.stock_performance_tool import stock_data_tool
from tools.wikipedia_tool import wikipedia_tool

#Setup OPENAI_API_KEY
from utils.load_env_file import load_env_file
load_env_file()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#Loop through the toolsand extract name and description attributes
for tool in [python_repl_tool, stock_data_tool, wikipedia_tool]:
    print(f"\n\nThe Description for {tool.name}: \n\n {tool.description}\n")

#Building the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
tools = [python_repl_tool, stock_data_tool, wikipedia_tool]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_API_KEY)

#Tell llm which tools it can call
llm_with_tools = llm.bind_tools(tools)

#Invoke llm_with_tools in the llm_nodes() function
def llm_nodes(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#Defining the agent linear workflow

#LLM node
graph_builder.add_node("llm", llm_nodes)

#Tools node
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

#Create the edges to represent the linear workflow

graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", "tools")
graph_builder.add_edge("tools", END)

graph = graph_builder.compile()

print(graph.get_graph().draw_mermaid())