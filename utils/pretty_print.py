"""
Utility module for formatting and displaying LangGraph stream output.

This module provides functions to pretty-print messages from LangGraph agent
streams, making it easier to visualize AI messages, tool calls, and tool
responses during agent execution.

Example:
    >>> from langgraph import LangGraph
    >>> from utils.pretty_print import pretty_print_messages
    >>>
    >>> graph = LangGraph(...)
    >>> for chunk in graph.stream({"messages": [...]}):
    ...     pretty_print_messages(chunk)
"""


def pretty_print_messages(chunk):
    """
    Pretty print messages from a LangGraph stream chunk.
    
    Formats and displays messages from graph nodes with proper indentation
    and separators. Handles AI messages with tool calls and tool messages
    with execution results.
    
    Args:
        chunk (dict): A dictionary with node names as keys and node outputs
                      as values. Expected structure: 
                      {node_name: {"messages": [message_obj, ...]}}
    
    Returns:
        None: Output is printed to stdout.
    
    Raises:
        AttributeError: If message objects don't have expected attributes
                       (type, content, tool_calls, name).
    """
    for node, output in chunk.items():
        print(f"\nUpdate from node {node}:\n")
        
        for message in output.get("messages", []):
            if message.type == "ai":
                print("=" * 82 + " Ai Message " + "=" * 82)
                
                if message.tool_calls:
                    print("Tool Calls:")
                    for tool_call in message.tool_calls:
                        print(f"  {tool_call['name']} ({tool_call['id']})")
                        print(f" Call ID: {tool_call['id']}")
                        print("  Args:")
                        for key, value in tool_call['args'].items():
                            print(f"    {key}: {value}")
                
                if message.content:
                    print(f"\n{message.content}")
            
            elif message.type == "tool":
                print("=" * 81 + " Tool Message " + "=" * 81)
                print(f"Name: {message.name}\n")
                print(message.content)
