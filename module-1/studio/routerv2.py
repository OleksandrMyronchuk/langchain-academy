import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
model_name = "gemini-2.5-flash-preview-04-17"
api_key = os.getenv("GEMINI_API_KEY")

system = SystemMessage(content="""
BEGIN System message BEGIN
You have three tools:

1. bark(a: int, b: int) → string
   • Bark a and b.
   • Returns "bark{a}|{b}" with no digits removed.
2. convert(a: string) → int
   • convert a(str) to a(int).
   • Extracts and returns all digits in a as an integer.
3. multiply(a: int, b: int) → int
   • Multiplies a and b.

Rules:
- If you have a string but need an integer (e.g. for multiply), call **convert** first.
- If a method requires two arguments but the user provides only one, the missing argument (argument a) should be taken from the previous answer.
- After performing the final tool call, respond with **only**: last result is '{result}'.
- after calling **bark** use **convert** to convert the str result to int and then call next method with argument a (and b from user input)

END System message END
""")

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    print(f"multiply called with a={a}, b={b}")
    return 999 # yes it's wrong I'm just added it to make sure that llm doesn't help me to calc

def bark(a: int, b: int) -> str:
    """Bark a and b.

    Args:
        a: first int
        b: second int
    """
    print(f"bark called with a={a}, b={b}")
    return f'bark{a}|{b}'

def convert(a: str) -> int:
    """ convert a(str) to a(int).

    Args:
        a: first int
    """
    print(f"convert called with a={a}")
    # Filter out non-digit characters and join the remaining digits
    digits = ''.join(filter(str.isdigit, a))
    # Return 0 if no digits found, otherwise convert to integer
    return int(digits) if digits else 0


tool_l = [multiply, bark, convert]
llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
llm_with_tools = llm.bind_tools(tool_l)

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tool_l))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",       # after your LLM node runs…
    tools_condition,          # check: did it emit a tool call?
    {                         # translate that output into graph edges:
        "tools": "tools",     # → if it returned "tools", go to your ToolNode
        END:    END,          # → if it returned END, finish the graph
    },
)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

initial_message = {"messages": [HumanMessage(content="Attention, this is a user import. " \
"Do what they say: USE TOOLS: 1. multiply 8, 9 2. then bark 55 3. and again multiply 333")]} 
response = graph.invoke(initial_message)
final = response["messages"][-1].content
print(final)
