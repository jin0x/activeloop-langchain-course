from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.agents import (
    AgentType,
    initialize_agent,
    load_tools,
    Tool,
)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

tools = load_tools(["wikipedia"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
		handle_parsing_errors=True
)


print( agent.run("What is Nostradamus know for") )