from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import router_node, retrieve_node, generate_node

def decide_to_retrieve(state: AgentState):
    """
    This is a conditional edge function. 
    It reads the state and tells the graph which node to go to next.
    """
    if state.get("needs_search"):
        return "retrieve"
    return "generate"

def build_esg_graph():
    # 1. Initialize the Graph with our specific State schema
    workflow = StateGraph(AgentState)

    # 2. Add the Nodes (The "Doers")
    workflow.add_node("analyze", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 3. Define the Edges (The "Pathways")
    # Start at the router
    workflow.set_entry_point("analyze")
    
    # Conditional Routing: After analysis, do we retrieve or just generate?
    workflow.add_conditional_edges(
        "analyze",              # The node we are coming from
        decide_to_retrieve,     # The function that makes the decision
        {
            "retrieve": "retrieve", # If decision is 'retrieve', go to retrieve_node
            "generate": "generate"  # If decision is 'generate', skip to generate_node
        }
    )
    
    # If we retrieved, the next step is always to generate
    workflow.add_edge("retrieve", "generate")
    
    # After generating, the graph is finished
    workflow.add_edge("generate", END)

    # Compile the graph into an executable agent
    return workflow.compile()

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import router_node, retrieve_node, generate_node, audit_node

def decide_to_retrieve(state: AgentState):
    if state.get("needs_search"): return "retrieve"
    return "generate"

def assess_audit(state: AgentState):
    """Reads the auditor's verdict and decides whether to loop or end."""
    feedback = state.get("feedback")
    revisions = state.get("revision_count", 0)
    
    # If it passed, or we tried too many times, stop.
    if feedback == "PASS" or revisions >= 3:
        return "end"
    
    # Otherwise, loop back to generation to try again
    return "revise"

def build_esg_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("analyze", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("audit", audit_node) # NEW NODE

    # Build the Flow
    workflow.set_entry_point("analyze")
    
    workflow.add_conditional_edges(
        "analyze",              
        decide_to_retrieve,     
        {"retrieve": "retrieve", "generate": "generate"}
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "audit") # Send the draft to the auditor
    
    # Conditional Loop: Did it pass the audit?
    workflow.add_conditional_edges(
        "audit",
        assess_audit,
        {
            "revise": "generate", # Loop back!
            "end": END            # Deliver to user
        }
    )

    return workflow.compile()