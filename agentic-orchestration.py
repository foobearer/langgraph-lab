import os
import warnings
from typing import Annotated
from typing_extensions import TypedDict

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from load_env_file import load_env_file


load_env_file()

# Define and initialize the OpenAI LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your shell or in a local .env file before running this script."
    )

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Generate state to capture the message
class State(TypedDict): 
    messages = Annotated[list, add_messages]

# Create a graph state
graph_builder = StateGraph(State)

# Create a method that takes the state and appends the new messages to it
def llm_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

#create a node name "llm" and calls llm_node function
graph_builder.add_node("llm", llm_node)

# Connect the llm to the start and end graph
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

# Compile the graph
graph = graph_builder.compile()

# Render the graph as Mermaid text
print(graph.get_graph().draw_mermaid())

# To visualise the output you can navigate to mermaid.live

