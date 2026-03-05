from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.agent.graph import build_esg_graph

# Load environment variables (like GOOGLE_API_KEY) from your .env file
load_dotenv() 

def main():
    agent = build_esg_graph()
    
    print("🏦 Welcome to the ESG Banking Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() == 'exit':
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Run the agent
        final_state = agent.invoke(inputs)
        print(f"\nAnalyst:\n{final_state['messages'][-1].content}")

if __name__ == "__main__":
    main()

