import operator
from typing import Annotated, TypedDict, Union, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- 1. SETUP ---
from rag import setup_rag_pipeline
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found! Check your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key
)

# Initialize RAG
retriever = setup_rag_pipeline()

# --- 2. DEFINE TOOLS ---
@tool
def retrieve_knowledge(query: str):
    """Useful for answering questions about AutoStream pricing, features, and policies."""
    # This connects the RAG pipeline to the agent
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


@tool
def mock_lead_capture(name: str, email: str, platform: str):
    """Call this ONLY when you have collected the user's Name, Email, and Content Platform."""
    # The exact function required by the problem statement
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return "Lead saved successfully."


tools = [retrieve_knowledge, mock_lead_capture]
llm_with_tools = llm.bind_tools(tools)


# --- 3. DEFINE STATE ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# --- 4. DEFINE NODES ---
def chatbot(state: AgentState):
    """The main decision-making node."""

    # System Prompt: The Instructions
    system_instruction = """You are a helpful sales assistant for AutoStream, a video editing SaaS.

    YOUR GOALS:
    1. Answer questions about pricing/policies using the 'retrieve_knowledge' tool.
    2. If a user shows HIGH INTENT (wants to sign up/buy), you must collect:
       - Name
       - Email
       - Content Platform (YouTube/Instagram)

    RULES:
    - Do NOT call 'mock_lead_capture' until you have ALL three details.
    - If details are missing, ask for them one by one naturally.
    - Be polite and professional.
    """

    # Combine system instructions with conversation history
    messages = [SystemMessage(content=system_instruction)] + state["messages"]

    # Invoke the LLM
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# --- 5. BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", chatbot)
workflow.add_node("tools", ToolNode(tools))  # Prebuilt node to run tools

# Add Edges
workflow.set_entry_point("agent")


# Conditional Edge: If LLM wants to use a tool, go to "tools". Otherwise, END.
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")  # Loop back to agent after tool execution

# Compile with Memory (State Persistence)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 6. TESTING (Optional Run Block) ---
def clean_print(message):
    """Helper to handle both text strings and list-based responses from Gemini."""
    content = message.content
    if isinstance(content, list):
        # If it's a list (Gemini's raw format), join the text parts
        text = " ".join([block['text'] for block in content if 'text' in block])
        print(f"Agent: {text}")
    else:
        # If it's just a string
        print(f"Agent: {content}")

if __name__ == "__main__":
    # Unique thread ID for this conversation (keeps memory alive)
    config = {"configurable": {"thread_id": "demo_user_final_1"}}

    print("\n--- STARTING DEMO ---\n")

    # TURN 1: Casual Greeting
    print("User: Hi!")
    inputs = {"messages": [HumanMessage(content="Hi!")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])

    # TURN 2: RAG / Pricing Question
    print("\nUser: Tell me about your pricing.")
    inputs = {"messages": [HumanMessage(content="Tell me about your pricing.")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])
        if "tools" in event:
            print("(Agent is checking knowledge base...)")

    # TURN 3: Intent Shift (Agent should ask for Name)
    print("\nUser: That sounds good. I want to sign up for the Pro plan.")
    inputs = {"messages": [HumanMessage(content="That sounds good. I want to sign up for the Pro plan.")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])

    # TURN 4: DEVIATION! (User ignores question, asks about Support)
    print("\nUser: Wait, does the Pro plan actually have 24/7 support?")
    inputs = {"messages": [HumanMessage(content="Wait, does the Pro plan actually have 24/7 support?")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])
        if "tools" in event:
             print("(Agent is checking knowledge base...)")

    # TURN 5: Back on Track (User finally gives Name)
    print("\nUser: Okay cool. My name is Samarth.")
    inputs = {"messages": [HumanMessage(content="Okay cool. My name is Samarth.")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])

    # TURN 6: Slot Filling (Email)
    print("\nUser: My email is sam@test.com.")
    inputs = {"messages": [HumanMessage(content="My email is sam@test.com.")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])

    # TURN 7: Completion (Platform -> Tool Trigger)
    print("\nUser: I create content for YouTube.")
    inputs = {"messages": [HumanMessage(content="I create content for YouTube.")]}
    for event in app.stream(inputs, config=config):
        if "agent" in event:
            clean_print(event['agent']['messages'][-1])
        if "tools" in event:
            print(">>> SUCCESS: Lead Capture Tool Called! <<<")