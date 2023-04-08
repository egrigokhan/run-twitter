import os

from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper
from lib.twitter.index import send_tweet

class TwitterTool(BaseTool):
    name = "Twitter"
    description = "Send a tweet and then report to the user if the tweet was successfully sent or not"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return send_tweet(query, os.environ["twitter_auth"]["access_token"])
