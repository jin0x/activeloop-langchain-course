from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.utilities import PythonREPL
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
python_repl = PythonREPL()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

toolkit = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to search Google to answer questions about current events"
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. Useful for saving strings to files.",
        func=python_repl.run
    )
]

agent = initialize_agent(
	toolkit,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

agent.run("Find the birth date of Napoleon Bonaparte and save it to a file 'answer.txt'.")