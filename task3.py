# # An Agentic RAG System:
#Build an agentic RAG system on Computer Science domain
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
