# llm/llm.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, add_messages
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_agent

# -----------------------------
# Import your custom tools
# -----------------------------
from llm.tools_ import (
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
)

# -----------------------------
# Define state structure
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------
# Initialize LLM and Tools
# -----------------------------
llm = ChatOllama(
    model="llama3",   # or mistral, phi3, codellama, etc.
    temperature=0.7
)


tools = [
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
]

# -----------------------------
# Create ReAct Agent Node
# -----------------------------
agent_node = create_agent(llm, tools)

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(AgentState)
graph.add_node("react_agent", agent_node)
graph.set_entry_point("react_agent")
graph.add_edge("react_agent", END)

# Compile the executable graph (equivalent to old AgentExecutor)
agent_executor = graph.compile()

# -----------------------------
# Ask Agent Function (for Flask)
# -----------------------------
def ask_agent(question: str) -> str:
    """Send user input to the LangGraph agent and return response."""
    try:
        result = agent_executor.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"⚠️ Error: {e}"
