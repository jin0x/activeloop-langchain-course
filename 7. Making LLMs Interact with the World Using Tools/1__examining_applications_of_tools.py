from dotenv import load_dotenv
load_dotenv()
import os

from langchain import OpenAI
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType
)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
tools = load_tools(['serpapi', 'requests_all'], llm=llm, serpapi_api_key=os.environ['SERPAPI_API_KEY'])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)

text = agent.run("tell me what is midjourney?")

#
tools = load_tools(["google-search"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print( agent("What is the national drink in Spain?") )