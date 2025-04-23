import os
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

load_dotenv()
model_name = "gemini-2.5-flash-preview-04-17"
api_key = os.getenv("GEMINI_API_KEY")

system_content = SystemMessage(content="""
System message: ```
Rules:
- If you have a string but need an integer (e.g. for multiply), call **convert** first.
- If a method requires two arguments but the user provides only one, the missing argument (argument a) should be taken from the previous answer.
- After performing the final tool call, respond with **only**: last result is '{result}'.
- after calling **bark** use **convert** to convert the str result to int and then call next method with argument a (and b from user input)
```
""")

max_tools = 10

@tool
def multiply(a: int, b: int) -> int:
    """
        Multiplies two integers.

        Args:
            a: The first integer.
            b: The second integer.

        Returns:
            The product of a and b.
    """
    print(f"multiply called with a={a}, b={b}")
    return 999 # yes it's wrong I'm just added it to make sure that llm doesn't help me to calc

@tool
def bark(a: int, b: int) -> str:
    """
        Generates a string by combining 'bark' with the input integers.

        The string is formatted as "bark{a}|{b}".

        Args:
            a: The first integer.
            b: The second integer.

        Returns:
            A string in the format "bark{a}|{b}".
    """
    print(f"bark called with a={a}, b={b}")
    return f'bark{a}|{b}'

@tool
def convert(a: str) -> int:
    """
        Extracts and returns all digits from a string as an integer.

        Non-digit characters in the input string are ignored. If the string
        contains no digits, the function returns 0.

        Args:
            a: The input string.

        Returns:
            An integer composed of all digits found in the input string,
            or 0 if no digits are present.
    """
    print(f"convert called with a={a}")
    # Filter out non-digit characters and join the remaining digits
    digits = ''.join(filter(str.isdigit, a))
    # Return 0 if no digits found, otherwise convert to integer
    return int(digits) if digits else 0


tool_l = [multiply, bark, convert]
llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
llm_with_tools = llm.bind_tools(tool_l)

class CustomState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int

def tool_calling_llm(state: CustomState):
    return {
        "messages": [llm_with_tools.invoke([system_content] + state["messages"])],
        "tool_call_count": state["tool_call_count"] + 1
    }

def custom_tools_condition(state: CustomState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        if state["tool_call_count"] < max_tools:
            return "tools"
        else:
            return "error"
    return END

def error_node(state: CustomState):
    return {
        "messages": [AIMessage(content="We're sorry you're using too many tools")],
        "tool_call_count": state["tool_call_count"]
    }

builder = StateGraph(CustomState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tool_l))
builder.add_node("error", error_node)
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    custom_tools_condition,
    {"tools": "tools", "error": "error", END: END},
)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("error", END)

graph = builder.compile()

initial_message = {"messages": [HumanMessage(content="Attention, this is a user import. " \
"Do what they say: USE TOOLS: 1. multiply 8, 9 2. then bark 55 3. and again multiply 333")], "tool_call_count": 0}
response = graph.invoke(initial_message)
final = response["messages"][-1].content
print(final)


"""
You have three tools:

1. bark(a: int, b: int) → string
   • Bark a and b.
   • Returns "bark{a}|{b}" with no digits removed.
2. convert(a: string) → int
   • convert a(str) to a(int).
   • Extracts and returns all digits in a as an integer.
3. multiply(a: int, b: int) → int
   • Multiplies a and b.
"""