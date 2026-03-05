from typing import TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import BaseMessage

class SearchFilters(BaseModel):
    """Extraction of search parameters from a user query."""
    company: Optional[str] = Field(description="The name of the company, e.g., Apple, Microsoft")
    year: Optional[int] = Field(description="The specific reporting year mentioned")
    topic: Optional[str] = Field(description="The ESG topic, e.g., emissions, board diversity")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    filters: SearchFilters 
    needs_search: bool
    context: str
    revision_count: int  # NEW: To prevent infinite loops
    feedback: str


#This file defines the "memory" of our agent and the data structures it uses to think. 
# Have put the Pydantic Schemas and AgentState here