import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from lib.agents.index import create_agent
from lib.twitter.index import send_tweet
import json

def approve(type, request):
    # switch case
    if type == "send_tweet":
       return send_tweet(request["tweet"], json.loads(os.environ["twitter_auth"])["access_token"])
    elif type == "not_approved":
        return "Posting tweet not approved."
    else:
        return "Error approving."
    
def run(message, history): 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    try:
        memory.chat_memory.clear()
        memory.chat_memory.messages = history
    except Exception as e:
        print(e)   
    try:
        # memory.buffer = history
        agent = create_agent(memory)
        
        return agent.run(message)
    except Exception as e:
        print(e)
        return "I'm sorry, I'm having trouble understanding you. Could you please rephrase?"


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["name"] = config["name"]
    os.environ["tagline"] = config["tagline"]
    os.environ["description"] = config["description"]
    os.environ["language"] = config["language"]
    os.environ["twitter_auth"] = config["twitter_auth"]