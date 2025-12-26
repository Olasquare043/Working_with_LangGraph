# importing the neccessary libraries
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from dotenv import load_dotenv
load_dotenv()
print("🔃 Loading LLM...")
# initializing the llm
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# prepare the system messages
sys_promp="""You are a Customer Support Assistant for Olasquare Gadget and Laptop Company.

Your primary responsibility is to assist customers with any issues, questions, or concerns related to gadgets or laptops they purchased from Olasquare Gadgets Limited.

Guidelines:
1. Always greet the customer politely when appropriate and clearly identify yourself as customer support for Olasquare Gadgets Limited.
2. Maintain a respectful, friendly, and professional tone at all times.
3. Show empathy and understanding, especially when customers report problems or dissatisfaction.
4. Ask relevant follow-up questions when necessary to fully understand the customer’s issue before offering a solution.
5. Provide clear, simple, and practical explanations using layman’s terms. Avoid technical jargon unless the customer requests it.
6. Offer step-by-step solutions when applicable and ensure instructions are easy to follow.
7. Aim to fully resolve the customer’s issue and confirm that the customer is satisfied with the assistance provided before ending the conversation.

Objective:
Deliver helpful, clear, and empathetic support that leaves the customer confident and satisfied.

"""
sys_msg = SystemMessage(content=sys_promp)

# function for assistance
def assistance(state:MessagesState):
    """Node for assistance- process the message and generte response"""
    messages = [sys_msg] + state["messages"]

    # generate response
    response= llm.invoke(messages)
    return {"messages":[AIMessage(content=response.content)]}

def build_graph()-> StateGraph:
    """Building graph with neccessary nodes, edges and memory"""
    # let create our LangGraph
    builder= StateGraph(MessagesState)

    # let build our node
    builder.add_node("assistance",assistance)

    # let define our flow - chart
    # START → assistance → END
    builder.add_edge(START, "assistance")
    builder.add_edge("assistance", END)

    # let build our agent with memory
    memory=MemorySaver()
    agent = builder.compile(checkpointer=memory)
    return agent
print("🔃 Building agent...")
agent= build_graph()
def flowchart_gen(agent:StateGraph=agent):
    try:
        display(Image(agent.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Could not display graph: {e}")

def single_run_chat(user_input:str,thread_id="user001"):
    """Get user input and return response"""
    
    result= agent.invoke({"messages":[HumanMessage(content=user_input)]},
                            config={"configurable":{"thread_id":thread_id}})
    # print the conversation
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\n👤 User: {message.content}")
        else:
            print(f"🤖 Agent: {message.content}")

def interactive_chat():
    """
    Run an interactive chat session.
    Type 'exit' or 'quit' to stop.
    """
    print("\n" + "="*70)
    print("🤖 Welcome to Olasquare Gadget Limited")
    print("Type your message and press Enter. Type 'exit' to quit.")
    print("="*70 + "\n")
    
    thread_id = "user1"
    
    while True:
        user_input = input("\n👤 You:  ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        # Get response
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Print agent's response
        agent_message = result["messages"][-1]
        print(f"\n🤖 Agent:  {agent_message.content}")


if __name__ == "__main__":
    # runnig the app
    print("\n" + "="*70)
    print("🤖 Welcome to Olasquare Gadget Limited")
    print("1. Check the agent FlowChart")
    print("2. Run an intercative chat")
    print("3. Run single run chat")
    print("--"*30)
    print("")
    choice = input("Your choice here:  ")
    if choice=="1":
        flowchart_gen()
    elif choice=="2":
        interactive_chat()
    elif choice=="3":
        print("\n" + "="*70)
        print("🤖 Welcome to Olasquare Gadget Limited")
        user_input=input("Type your message and press Enter.")
        single_run_chat(user_input)
    else:
        print("❌ Invalid input")