from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
import os
from lib.tools.index import TwitterTool
from lib.twitter.index import send_tweet
import json

def create_agent(memory):
    # Define which tools the agent can use to answer user queries
    tools = [
        Tool(
            name="Suggest Tweet",
            func=lambda x: "Current Tweet suggestion: " + x,
            description="create a tweet based on the user's input, without posting it.",
            return_direct=True
        ),
        Tool(
            name="Iterate Tweet",
            func=lambda x: x,
            description="refine the latest tweet suggestion according to the user's feedback.",
        ),
        Tool(
            name="Post Tweet",
            func=lambda x: send_tweet(x, json.loads(os.environ["twitter_auth"])["access_token"]),
            description="post the latest tweet suggestion to the user's twitter account. ONLY CALL THIS WHEN THE USER SAYS EXPLICITLY TO POST THE TWEET.",
        ),
        Tool(
            name="Chat",
            func=lambda x: x,
            description="engage in conversation with the user.",
            return_direct=True
        )
    ]

    prefix = f"""You are SocialMediaGPT, and you are collaborating with the social media manager of {os.environ['name']}.
    
Your primary task is to craft compelling Tweets based on the scenarios and requirements provided by the manager. Avoid asking questions prior to suggesting a Tweet.

Maintain an internal record of the current tweet suggestion and display it to the manager whenever requested.

Product details:

Name: {os.environ['name']}

Tagline: {os.environ['tagline']}

Description: {os.environ['description']}
            
The manager has established the following rules to be followed without exception in all tweets:

{os.environ['language']}
            
You have access to only the following tools:"""

    suffix = f"""

    The tool you can select has to be one of ${[tool.name for tool in tools]}.
Begin! Keep in mind not to post a tweet without the manager's explicit permission. Always show the manager the current tweet you have crafted together.

{{chat_history}}
Question: {{input}}
{{agent_scratchpad}}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=False)

    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, memory=memory)

    return agent_chain
