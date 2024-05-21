from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from chatbot_logic.state import AgentState
from chatbot_logic.agents import (
    input_assistant,
    game_search_assistant,
    game_description_assistant,
    game_platform_assistant,
    game_genre_assistant,
    game_developer_publisher_assistant,
    game_metacritic_assistant,
    game_age_restriction_assistant,
    game_trailer_assistant,
    output_assistant
)

# Define the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("input_assistant", input_assistant)
builder.add_node("game_search_assistant", game_search_assistant)
builder.add_node("game_description_assistant", game_description_assistant)
builder.add_node("game_platform_assistant", game_platform_assistant)
builder.add_node("game_genre_assistant", game_genre_assistant)
builder.add_node("game_developer_publisher_assistant", game_developer_publisher_assistant)
builder.add_node("game_metacritic_assistant", game_metacritic_assistant)
builder.add_node("game_age_restriction_assistant", game_age_restriction_assistant)
builder.add_node("game_trailer_assistant", game_trailer_assistant)
builder.add_node("output_assistant", output_assistant)

# Set entry point and edges
builder.set_entry_point("input_assistant")

# Define conditional edges
def should_continue(state):
    if state.get("relevant", False):
        return "continue"
    return "end"

builder.add_conditional_edges("input_assistant", should_continue, {"continue": "game_search_assistant", "end": END})

# Branching from game_search_assistant to run nodes in parallel
builder.add_edge("game_search_assistant", "game_description_assistant")
builder.add_edge("game_search_assistant", "game_platform_assistant")
builder.add_edge("game_search_assistant", "game_genre_assistant")
builder.add_edge("game_search_assistant", "game_developer_publisher_assistant")
builder.add_edge("game_search_assistant", "game_metacritic_assistant")
builder.add_edge("game_search_assistant", "game_age_restriction_assistant")
builder.add_edge("game_search_assistant", "game_trailer_assistant")

# All parallel nodes should merge to output assistant
builder.add_edge("game_description_assistant", "output_assistant")
builder.add_edge("game_platform_assistant", "output_assistant")
builder.add_edge("game_genre_assistant", "output_assistant")
builder.add_edge("game_developer_publisher_assistant", "output_assistant")
builder.add_edge("game_metacritic_assistant", "output_assistant")
builder.add_edge("game_age_restriction_assistant", "output_assistant")
builder.add_edge("game_trailer_assistant", "output_assistant")

# Ensure the output assistant is recognized as the final node
builder.set_finish_point("output_assistant")

# Persist state using SQLite
memory = SqliteSaver.from_conn_string(":memory:")
app = builder.compile(checkpointer=memory)