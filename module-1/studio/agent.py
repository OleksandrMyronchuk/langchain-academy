import os
import logging
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
if load_dotenv():
    logger.info("Loaded environment variables from .env file.")
else:
    logger.warning("No .env file found or failed to load environment variables.")

# Define tools with @tool decorator
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    logger.debug("add() called with a=%s, b=%s", a, b)
    result = a + b
    logger.debug("add() result: %s", result)
    return result

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    logger.debug("multiply() called with a=%s, b=%s", a, b)
    result = a * b
    logger.debug("multiply() result: %s", result)
    return result

@tool
def divide(a: int, b: int) -> float:
    """Divide two integers and return the quotient."""
    logger.debug("divide() called with a=%s, b=%s", a, b)
    if b == 0:
        logger.error("Division by zero: a=%s, b=%s", a, b)
        raise ZeroDivisionError("Cannot divide by zero")
    result = a / b
    logger.debug("divide() result: %s", result)
    return result

# Collect tools
tools = [add, multiply, divide]
logger.info("Collected tools: %s", [t.name for t in tools])

# Initialize the LLM with bound tools
model_name = "gemini-2.5-flash-preview-04-17"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
logger.info("Initialized ChatGoogleGenerativeAI with model %s", model_name)

llm_with_tools = llm.bind_tools(tools)
logger.info("Tools bound to LLM instance.")

# System message for the assistant
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
logger.info("System message defined: %s", sys_msg.content)

# Assistant node definition
def assistant(state: MessagesState):
    logger.info("Assistant node invoked; current messages: %s", state["messages"])
    try:
        response_msg = llm_with_tools.invoke([sys_msg] + state["messages"])
        logger.info("LLM response: %s", response_msg.content)
    except Exception as e:
        logger.error("Error during LLM invocation: %s", e, exc_info=True)
        raise
    return {"messages": [response_msg]}

# Build the state graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
logger.info("Graph nodes and edges configured.")

# Compile the graph
graph = builder.compile()
logger.info("Graph compiled successfully.")

# Define the initial user message
initial_message = {"messages": [HumanMessage(content="What is 3 plus 5, multiplied by 2, and then divided by 2? Execute this immediately without asking additional questions")]}
logger.info("Initial message prepared: %s", initial_message["messages"][0].content)

# Invoke the graph and print the result
try:
    response = graph.invoke(initial_message)
    logger.info("Graph invocation completed. Full messages: %s", response["messages"])
    final = response["messages"][-1].content
    logger.info("Final assistant response: %s", final)
    print(final)
except Exception as e:
    logger.error("Error invoking the graph: %s", e, exc_info=True)
    raise
