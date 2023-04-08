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

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def run(message, history):        
    agent = create_agent(memory)
    
    return agent.run(message)


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["name"] = config["name"]
    os.environ["tagline"] = config["tagline"]
    os.environ["description"] = config["description"]
    os.environ["language"] = config["language"]
    os.environ["twitter_auth"] = config["twitter_auth"]