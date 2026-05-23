import operator
from dotenv import load_dotenv

from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import InMemorySaver

from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def chatbot_node(state: MessagesState):
    print(" Agent is responding...")
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. "
                    "Use conversation memory properly "
                    "and answer based on previous messages."
                )
            )
        ]
        + state["messages"]
    )

    # Return only updates
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)

builder.add_edge(START, "chatbot")

builder.add_edge("chatbot", END)

memory = InMemorySaver()


app = builder.compile(checkpointer=memory)


config = {
    "configurable": {
        "thread_id": "user_1"
    }
}


result = app.invoke(
    {
        "messages": [
            HumanMessage(content="My name is Satyam.")
        ],
        "llm_calls": 0
    },
    config=config
)

print("\nAI Response 1:")
print(result["messages"][-1].content)


result = app.invoke(
    {
        "messages": [
            HumanMessage(content="what is my name")
        ],
        "llm_calls": 0
    },
    config=config
)

print("\nAI Response 2:")
print(result["messages"][-1].content)