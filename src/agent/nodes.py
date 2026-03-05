from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState, SearchFilters
from src.retrieval.retriever import hierarchical_retriever

# 1. Upgrade the Router Schema
class RouteQuery(BaseModel):
    needs_search: bool = Field(description="True if the query requires looking up specific ESG company data, False for general chat or greetings.")
    filters: Optional[SearchFilters] = Field(description="The extracted metadata filters, if a search is needed.")

# Initialize Gemini 1.5 Pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
structured_router = llm.with_structured_output(RouteQuery)

# 2. Define the Nodes
def router_node(state: AgentState):
    """Analyzes the query, extracts metadata, and decides if a database search is needed."""
    print("\n--- 🧠 ANALYZING & ROUTING ---")
    query = state["messages"][-1].content
    
    # Ask Gemini to structure the intent
    routing_decision = structured_router.invoke(query)
    
    print(f"Needs Search: {routing_decision.needs_search}")
    if routing_decision.filters:
        print(f"Extracted Filters: {routing_decision.filters.dict(exclude_none=True)}")
        
    # We pass 'needs_search' into the state so the Graph can use it to make a decision
    return {
        "filters": routing_decision.filters, 
        "needs_search": routing_decision.needs_search
    }

def retrieve_node(state: AgentState):
    """Searches the Hierarchical ChromaDB using the extracted filters."""
    print("--- 🔍 RETRIEVING PARENT DOCUMENTS ---")
    query = state["messages"][-1].content
    filters = state.get("filters")
    
    search_kwargs = {}
    if filters:
        filter_dict = {}
        if filters.company: filter_dict["company"] = filters.company
        if filters.year: filter_dict["year"] = str(filters.year)
        
        if filter_dict:
            # Tell ChromaDB to only search documents matching these exact tags
            search_kwargs = {"filter": filter_dict}
            
    # Retrieve the overarching Parent chunks based on the highly specific Child matches
    docs = hierarchical_retriever.invoke(query, **search_kwargs)
    
    # Combine the broad context
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    print(f"Retrieved {len(docs)} highly relevant parent section(s).")
    return {"context": context}

def generate_node(state: AgentState):
    """Generates the final analyst response."""
    print("--- ✍️ GENERATING RESPONSE ---")
    query = state["messages"][-1].content
    context = state.get("context", "")
    needs_search = state.get("needs_search", False)
    
    if needs_search and not context:
        response = llm.invoke([HumanMessage(content="Acknowledge that no specific ESG data was found for this query in the database.")])
        return {"messages": [response]}
        
    if not needs_search:
        # Standard conversation (no context injected)
        prompt = f"You are a professional investment banking AI assistant. Answer the user: {query}"
    else:
        # RAG conversation (context injected)
        prompt = f"""
        You are an expert investment banking analyst analyzing ESG reports. 
        Use ONLY the following retrieved context to answer the user's query. 
        Always cite the source if available in the context.
        If the answer is not in the context, state that clearly.
        
        Context: 
        {context}
        
        Query: {query}
        """
        
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# 1. Define the Grading Schema
class FactCheck(BaseModel):
    """Audits the generated response for accuracy."""
    is_grounded: bool = Field(description="True if all numbers and claims in the answer are strictly found in the context.")
    is_relevant: bool = Field(description="True if the answer directly addresses the user's question.")
    critique: str = Field(description="If there is a failure, explicitly state what number or fact is hallucinated or missing.")

fact_checker_llm = llm.with_structured_output(FactCheck)

# 2. Update the Generate Node to accept feedback and increment the counter
def generate_node(state: AgentState):
    print("--- ✍️ GENERATING / REVISING RESPONSE ---")
    query = state["messages"][0].content # The original question
    context = state.get("context", "")
    feedback = state.get("feedback", "")
    revisions = state.get("revision_count", 0)
    
    prompt = f"""
    You are an expert investment banking analyst. 
    Use ONLY the following retrieved context to answer the user's query. 
    
    Context: {context}
    Query: {query}
    """
    
    # If the auditor rejected the previous answer, inject the critique
    if feedback:
        print(f"Applying Auditor Feedback: {feedback}")
        prompt += f"\n\nWARNING: Your previous attempt was rejected. Feedback: {feedback}. Fix the errors."
        
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [response], 
        "revision_count": revisions + 1 # Increment the retry counter
    }

# 3. Create the new Auditor Node
def audit_node(state: AgentState):
    """Checks the generated answer against the context to prevent hallucinations."""
    print("--- 🕵️ AUDITING ANSWER ---")
    query = state["messages"][0].content
    context = state.get("context", "")
    generated_answer = state["messages"][-1].content
    needs_search = state.get("needs_search", False)
    
    # If it was just a general chat (no search), skip the audit
    if not needs_search:
        return {"feedback": "PASS"}

    grading_prompt = f"""
    You are a strict compliance auditor for an investment bank.
    Evaluate the GENERATED_ANSWER against the original QUERY and the SOURCE_CONTEXT.
    Ensure every number matches the context exactly.
    
    QUERY: {query}
    SOURCE_CONTEXT: {context}
    GENERATED_ANSWER: {generated_answer}
    """
    
    grade = fact_checker_llm.invoke(grading_prompt)
    
    if grade.is_grounded and grade.is_relevant:
        print("✅ Audit Passed: Answer is grounded and relevant.")
        return {"feedback": "PASS"}
    else:
        print(f"❌ Audit Failed: {grade.critique}")
        return {"feedback": grade.critique}