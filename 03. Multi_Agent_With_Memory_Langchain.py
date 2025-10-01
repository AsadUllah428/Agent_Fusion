from langchain_groq import ChatGroq
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
import os
import signal
import sys

# Configuration
os.environ["GROQ_API_KEY"] = "Your_Groq_API_Key_Here"

# Initialize LLM - using LLaMA3
llm = ChatGroq(
    temperature=0.7,
    model_name="Your_Model_Name_Here",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Memory
memory = ConversationBufferWindowMemory(k=6, return_messages=True)

# System prompt
SYSTEM_PROMPT = """You are an expert AI assistant. Be helpful, concise, and friendly. 
Respond professionally without revealing your internal thinking process.
Keep responses under 3 sentences unless detailed explanation is needed."""

# State definition
class State(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    should_end: bool

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\nAssistant: Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Nodes
def get_user_input(state: State):
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit"]:
        return {"messages": [SystemMessage(content="SESSION_END")], "should_end": True}
    return {"messages": [HumanMessage(content=user_input)], "should_end": False}

def chatbot(state: State):
    try:
        # Check if we should end first
        if state.get("should_end", False):
            return state
            
        # Prepare conversation history
        chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
        chat_history.extend(memory.load_memory_variables({})["history"] if memory.chat_memory.messages else [])
        chat_history.extend(state['messages'])
        
        # Get response
        response = llm.invoke(chat_history)
        
        # Update memory
        if len(state['messages']) > 0 and isinstance(state['messages'][-1], HumanMessage):
            memory.save_context(
                {"input": state['messages'][-1].content},
                {"output": response.content}
            )
        
        return {"messages": [AIMessage(content=response.content)], "should_end": False}
    except Exception as e:
        print(f"Error: {e}")
        return {"messages": [AIMessage(content="Sorry, I encountered an error. Please try again.")], "should_end": False}

# Graph construction
def should_continue(state: State):
    # Check if there's a SESSION_END message or the should_end flag is True
    if state.get("should_end", False):
        return END
        
    # Check for SESSION_END message
    if len(state['messages']) > 0 and isinstance(state['messages'][-1], SystemMessage):
        if state['messages'][-1].content == "SESSION_END":
            return END
            
    return "get_input"

# Build workflow
workflow = StateGraph(State)
workflow.add_node("get_input", get_user_input)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "get_input")
workflow.add_edge("get_input", "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "get_input": "get_input",
        END: END
    }
)
graph = workflow.compile()

# Main loop
def main():
    print("Chatbot initialized. Type 'quit', 'Quit' or 'exit' to exit.")
    
    try:
        for event in graph.stream({"messages": [], "should_end": False}):
            for key, value in event.items():
                if key == "chatbot" and "messages" in value and len(value['messages']) > 0:
                    last_msg = value['messages'][-1]
                    if isinstance(last_msg, AIMessage):
                        print(f"\nAssistant: {last_msg.content}")
                
                # Check if we should end
                if value.get("should_end", False):
                    print("\nAssistant: Goodbye!")
                    return
    except StopIteration:
        print("\nAssistant: Goodbye!")

if __name__ == "__main__":
    main()
