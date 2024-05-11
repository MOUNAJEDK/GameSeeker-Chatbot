from typing_extensions import TypedDict
from typing import Annotated, List, Optional
from langgraph.graph.message import AnyMessage, add_messages

class GameDetail(TypedDict, total=False):
    title: str
    description: str
    platforms: List[str]
    genre: str
    developer: str
    rating: float
    age_restriction: Optional[str]
    trailer_url: Optional[str]

class GameState(TypedDict):
    """
    GameState defines the state of the video game recommendation chatbot.
    It keeps track of user interactions and consolidates game search results 
    and details into a structured format.
    """
    # Stores messages between the user and the chatbot to provide context and history.
    messages: Annotated[List[AnyMessage], add_messages]

    # Current or last user interaction, which can include queries or feedback on recommendations.
    user_interaction: str

    # Consolidated list of games with their respective details.
    game_details: List[GameDetail]
