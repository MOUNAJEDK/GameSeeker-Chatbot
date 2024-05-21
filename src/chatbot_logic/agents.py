from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from chatbot_logic.state import AgentState

# Initialize the Tavily Search tool
tavily_search = TavilySearchResults(max_results=5)
tools = [tavily_search]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [{"role": "user", "content": "Respond with a real output."}]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Input Assistant
classification_prompt = ChatPromptTemplate.from_template(
    """
    You are part of a chatbot that provides personalized video game recommendations based on user preferences.
    Your task is to classify the following user query as either 'relevant' or 'irrelevant' to the topic of video game recommendations.

    A 'relevant' query includes:
    - Requests for game recommendations similar to a specific game (e.g., "What games are similar to Skyrim?")
    - Preferences for game genres (e.g., "I like open-world RPG games, can you recommend some?")
    - Inquiries about platform availability for games (e.g., "What games are available on PS5?")
    - Questions about game developers, publishers, or game details (e.g., "Who developed The Witcher 3?")

    An 'irrelevant' query includes:
    - General inquiries not related to video games (e.g., "What's the weather like today?")
    - Queries about other types of entertainment or products (e.g., "Can you recommend a good book?")
    - Vague questions without specific reference to video games (e.g., "Can you help me with something?")

    <question>
    {question} 
    </question>

    Classification:
    """
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
classification_chain = classification_prompt | llm | StrOutputParser()

def input_assistant(state: AgentState, config=None):
    input_text = state["messages"][-1]["content"]
    classification = classification_chain.invoke({"question": input_text})["output"]
    state["relevant"] = classification == "relevant"
    if not state["relevant"]:
        response = "Please provide a more specific query related to video games."
        state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Search Assistant
game_search_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides video game recommendations. Use the Tavily API to search for games based on the user's query.
    Only return the titles of the games found.

    User's query: {query}
    """
)

game_search_runnable = game_search_prompt | llm.bind_tools(tools)

def game_search_assistant(state: AgentState, config=None):
    assistant = Assistant(game_search_runnable)
    result = assistant(state, config)
    game_titles = [message['content'] for message in result['messages']]
    state["games"] = [{"title": title} for title in game_titles]
    response = f"Found {len(game_titles)} games based on your query."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Description Assistant
game_description_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides detailed descriptions of video games. Use the Tavily API to fetch descriptions for the following games:
    
    Games: {games}
    
    Only return the descriptions.
    """
)

game_description_runnable = game_description_prompt | llm.bind_tools(tools)

def game_description_assistant(state: AgentState, config=None):
    assistant = Assistant(game_description_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    descriptions = [message['content'] for message in result['messages']]
    for game, description in zip(state["games"], descriptions):
        game["description"] = description
    response = "Fetched descriptions for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Platform Assistant
game_platform_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides platform availability and store details for video games. Use the Tavily API to fetch platform details and store links for the following games:

    Games: {games}
    
    For each game, return the platforms it is available on and the stores where it can be purchased from, along with links to the store pages.

    Example format:
    [PC]
    Steam: *link to the game's page on Steam*
    GOG: *link to the game's page on GOG*

    [PlayStation]
    PlayStation Store: *link to the game's page on PlayStation Store*

    Games: {games}
    """
)

game_platform_runnable = game_platform_prompt | llm.bind_tools(tools)

def game_platform_assistant(state: AgentState, config=None):
    assistant = Assistant(game_platform_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    platform_details = [message['content'] for message in result['messages']]
    for game, platforms in zip(state["games"], platform_details):
        game["platforms"] = platforms
    response = "Fetched platform and store details for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Genre Assistant
game_genre_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides genre information for video games. Use the Tavily API to fetch genre details for the following games:
    
    Games: {games}
    
    Only return the genre information.
    """
)

game_genre_runnable = game_genre_prompt | llm.bind_tools(tools)

def game_genre_assistant(state: AgentState, config=None):
    assistant = Assistant(game_genre_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    genre_details = [message['content'] for message in result['messages']]
    for game, genres in zip(state["games"], genre_details):
        game["genres"] = genres
    response = "Fetched genre details for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Developer/Publisher Assistant
game_developer_publisher_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides developer and publisher information for video games. Use the Tavily API to fetch developer and publisher details for the following games:
    
    Games: {games}
    
    Only return the developer and publisher information.
    """
)

game_developer_publisher_runnable = game_developer_publisher_prompt | llm.bind_tools(tools)

def game_developer_publisher_assistant(state: AgentState, config=None):
    assistant = Assistant(game_developer_publisher_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    developer_publisher_details = [message['content'] for message in result['messages']]
    for game, details in zip(state["games"], developer_publisher_details):
        game["developer_publisher"] = details
    response = "Fetched developer and publisher details for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Metacritic Assistant
game_metacritic_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides Metacritic scores for video games. Use the Tavily API to fetch Metacritic details for the following games:
    
    Games: {games}
    
    Only return the Metacritic scores.
    """
)

game_metacritic_runnable = game_metacritic_prompt | llm.bind_tools(tools)

def game_metacritic_assistant(state: AgentState, config=None):
    assistant = Assistant(game_metacritic_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    metacritic_details = [message['content'] for message in result['messages']]
    for game, metacritic in zip(state["games"], metacritic_details):
        game["metacritic"] = metacritic
    response = "Fetched Metacritic scores for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Age Restriction Assistant
game_age_restriction_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides age restriction information for video games. Use the Tavily API to fetch age restriction details for the following games:
    
    Games: {games}
    
    Only return the age restriction information.
    """
)

game_age_restriction_runnable = game_age_restriction_prompt | llm.bind_tools(tools)

def game_age_restriction_assistant(state: AgentState, config=None):
    assistant = Assistant(game_age_restriction_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    age_restriction_details = [message['content'] for message in result['messages']]
    for game, age_restriction in zip(state["games"], age_restriction_details):
        game["age_restriction"] = age_restriction
    response = "Fetched age restriction details for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Game Trailer Assistant
game_trailer_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that provides trailer links for video games. Use the Tavily API to fetch trailer details for the following games:
    
    Games: {games}
    
    Only return the trailer links.
    """
)

game_trailer_runnable = game_trailer_prompt | llm.bind_tools(tools)

def game_trailer_assistant(state: AgentState, config=None):
    assistant = Assistant(game_trailer_runnable)
    games = [game["title"] for game in state["games"]]
    result = assistant({"messages": state["messages"], "games": games}, config)
    trailer_details = [message['content'] for message in result['messages']]
    for game, trailer in zip(state["games"], trailer_details):
        game["trailer"] = trailer
    response = "Fetched trailer links for the games."
    state["messages"].append({"role": "assistant", "content": response})
    return state

# Output Assistant
def compile_response(games):
    response = "Here are the details for the games you requested:\n"
    for game in games:
        response += f"Title: {game['title']}\n"
        response += f"Description: {game.get('description', 'N/A')}\n"
        response += f"Genres: {game.get('genres', 'N/A')}\n"
        response += f"Developer/Publisher: {game.get('developer_publisher', 'N/A')}\n"
        response += f"Metacritic Score: {game.get('metacritic', 'N/A')}\n"
        response += f"Age Restriction: {game.get('age_restriction', 'N/A')}\n"
        response += f"Trailer: {game.get('trailer', 'N/A')}\n"
        response += "Platforms and Stores:\n"
        platforms = game.get('platforms', {})
        for platform, stores in platforms.items():
            response += f"[{platform}]\n"
            for store, link in stores.items():
                response += f"{store}: {link}\n"
        response += "\n"
    return response

def output_assistant(state: AgentState, config=None):
    response = compile_response(state["games"])
    state["messages"].append({"role": "assistant", "content": response})
    return state