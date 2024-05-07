from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.agents import (
    AgentType,
    initialize_agent,
    Tool,
)
from langchain_experimental.utilities import PythonREPL

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

agent = initialize_agent(
    [repl_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
  )

print( agent.run("Create a list of casual strings containing 4 letters, list should contain 30 examples, and sort the list alphabetically") )