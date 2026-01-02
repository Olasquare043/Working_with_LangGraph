# # An agent with three custom tools:
# 1. Weather tool: Returns simulated weather for a given city
# 2. Dictionary tool: Looks up word definitions (simulate with a small dict)
# 3. Web search tool: Uses DuckDuckGo to search the web for information
# ===========================================================================

# Importing require libraries
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from typing import Literal
from langchain_openai import ChatOpenAI
import requests

# ===========================================================================

# load api key and intialize llm
load_dotenv()
llm= ChatOpenAI(model='gpt-4o-mini', temperature=0)
# ===========================================================================

# ---- Building nodes/tools----
@tool
def check_weather(city:str) -> str:
    """
    check the weather of a given city and return the weather details
    
    Args: city
    Return: weather details

    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params={
        "q":city,
        "appid": os.getenv("WEATHER_KEY"),
        "units": "metric"
    }
    response= requests.get(url, params=params)
    if response.status_code!="200":
        return f'Failed to get weather for {city}. Reason:{response.text}'
    data= response.json()
    return{
        f'Current weather in {city}:\n'
        
    }