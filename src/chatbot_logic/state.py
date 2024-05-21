from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages] # List of messages in the conversation
    query: str # User query
    relevant: bool # Whether the user query is relevant to video game recommendations
    games: List[Dict] # List of games matching the user query
    details: Dict[str, Dict] # Details of the selected games