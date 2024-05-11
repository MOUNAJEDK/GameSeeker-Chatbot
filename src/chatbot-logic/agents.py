from typing import Any
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

class CoreAssistant:
    """
    CoreAssistant defines the core logic of the chatbot, it is responsible for:
    - Handling user interactions, which are then routed to the next appropriate agent, if they are relevant and clear, or back to the user if they are not.
    - Providing responses to user queries and feedback.
    """
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
