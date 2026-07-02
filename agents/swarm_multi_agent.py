import os
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_swarm import create_handoff_tool, create_swarm

from tools.wikipedia_tool import wikipedia_tool
from tools.stock_performance_tool import stock_data_tool
from tools.python_repl_tool import python_repl_tool

from utils.pretty_print import pretty_print_messages
from utils.load_env_file import load_env_file
load_env_file()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPEN_API_KEY)

#Create handoff researcher tool
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or load stock performance data from CSV files.",
)

#Create handoff analyst tool
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations of provided data."
)

#Create agents

research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
    name="researcher"
)

analyst_agent = create_react_agent(
    llm,
    tools=[python_repl_tool, transfer_to_researcher],
    prompt="You generate plots of stock performance data provided by another assistant.",
    name="analyst"
)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

swarm = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher",
).compile(checkpointer=checkpointer)

print(swarm.get_graph().draw_mermaid())


for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]},
    config=config,
):
    pretty_print_messages(chunk)